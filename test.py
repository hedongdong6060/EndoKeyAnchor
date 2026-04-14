import cv2
import os
import argparse
import sys
import torch
import numpy as np
import time
from collections import deque

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QSlider, QVBoxLayout, QDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    from mainwindowui import Ui_MainWindow
    from login import LoginForm
except ImportError:
    class Ui_MainWindow(object):
        def setupUi(self, x): pass

    class LoginForm(QtWidgets.QWidget):
        login_successful = QtCore.pyqtSignal()

custom_modules_path = r'E:\a_shiyiyan\ultralytics\pose'
if custom_modules_path not in sys.path:
    sys.path.insert(0, custom_modules_path)

try:
    import ultralytics.nn.tasks as tasks
    from ultralytics import YOLO
    import custom_modules
    from custom_modules import HQIMIntegrationModule
    tasks.HQIMIntegrationModule = HQIMIntegrationModule
except ImportError as e:
    print(f"❌ 关键模块导入失败: {e}")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('--weights',
                    default=r"E:\a_shiyiyan\ultralytics\pose\runs\streamlined_experiment40\sar_enhanced_v40\weights\best.pt",
                    type=str, help='weights path')
parser.add_argument('--conf_thre', type=float, default=0.5, help='conf_thre')
parser.add_argument('--iou_thre', type=float, default=0.5, help='iou_thre')
opt = parser.parse_args()


class SmartKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1],
            [0, 0, 1, 0], [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0]
        ], dtype=np.float32)

        self.base_R_val = 0.01
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.base_R_val

        self.initialized = False
        self.missed_ticks = 0
        self.max_missed_ticks = 15

        self.last_observed_pos = None
        self.last_observed_tick = 0

    def update_with_sigma(self, measurement, sigma=None, conf=1.0, current_tick=0):
        if measurement[0] < 1.0 and measurement[1] < 1.0:
            return self.predict_with_instrument_guide(0, 0)

        if not self.initialized:
            self.kf.statePost = np.array([measurement[0], measurement[1], 0, 0], dtype=np.float32).reshape(-1, 1)
            self.initialized = True
            self.last_observed_pos = measurement
            self.last_observed_tick = current_tick
            return measurement

        self.kf.predict()

        if self.missed_ticks > 0 and self.last_observed_pos is not None:
            dt = current_tick - self.last_observed_tick
            if dt < 1: dt = 1.0
            dx = measurement[0] - self.last_observed_pos[0]
            dy = measurement[1] - self.last_observed_pos[1]
            self.kf.statePost[2, 0] = dx / dt
            self.kf.statePost[3, 0] = dy / dt
            self.kf.errorCovPost[2, 2] *= 0.1
            self.kf.errorCovPost[3, 3] *= 0.1

        if sigma is None:
            sigma = 0.5

        lambda_factor = 5.0
        r_scale = (sigma / (conf + 1e-6)) * lambda_factor
        dynamic_R = self.base_R_val * r_scale
        dynamic_R = max(1e-4, min(dynamic_R, 10.0))

        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * dynamic_R
        self.kf.correct(np.array([[measurement[0]], [measurement[1]]], dtype=np.float32))

        self.missed_ticks = 0
        self.last_observed_pos = measurement
        self.last_observed_tick = current_tick

        return (self.kf.statePost[0, 0], self.kf.statePost[1, 0])

    def predict_with_instrument_guide(self, inst_dx=0.0, inst_dy=0.0):
        if not self.initialized: return None

        self.missed_ticks += 1
        if self.missed_ticks > self.max_missed_ticks: return None

        self.kf.predict()

        guide_weight = 0.2

        if abs(inst_dx) > 2.0 or abs(inst_dy) > 2.0:
            self.kf.statePost[0, 0] += inst_dx * guide_weight
            self.kf.statePost[1, 0] += inst_dy * guide_weight

        damping = 0.6
        self.kf.statePost[2, 0] *= damping
        self.kf.statePost[3, 0] *= damping

        return (self.kf.statePost[0, 0], self.kf.statePost[1, 0])

    def predict_only(self):
        return self.predict_with_instrument_guide(0, 0)

    def reset(self):
        self.initialized = False
        self.missed_ticks = 0
        self.last_observed_pos = None
        self.last_observed_tick = 0


class Detector(object):
    def __init__(self, weight_path, conf_threshold=0.5, iou_threshold=0.5):
        print(f"Loading model from {weight_path}...")
        try:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model = YOLO(weight_path)
            self.model.to(self.device)
            self.conf_threshold = conf_threshold

            self.smart_tracker = SmartKalmanFilter()
            self._frame_counter = 0

            self.last_instrument_center = None

            self.stats = {
                'total_frames': 0,
                'has_papilla_detection': 0,
                'no_papilla_detection': 0,
                'use_keypoint': 0,
                'use_bbox_center': 0,
                'keypoint_high_conf': 0,
                'keypoint_mid_conf': 0,
                'keypoint_low_conf': 0,
                'no_keypoint_output': 0,
                'tracking': 0,
                'inst_guide': 0,
                'predicting': 0,
                'lost': 0,
                'filtered_by_zero_check': 0,
                'total_conf': 0.0,
                'min_conf': 1.0,
                'max_conf': 0.0,
            }

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def get_instrument_center(self, instruments):
        if not instruments: return None
        sum_x, sum_y, count = 0, 0, 0
        for box, conf, cls_id in instruments:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            sum_x += cx
            sum_y += cy
            count += 1
        if count == 0: return None
        return (sum_x / count, sum_y / count)

    def reset_all_memory(self):
        print("✓ [System] Memory Reset.")
        if hasattr(self.model.model, 'reset_memory'):
            self.model.model.reset_memory()
        self.smart_tracker.reset()
        self.last_instrument_center = None
        self._frame_counter = 0

    def reset_stats(self):
        for key in self.stats:
            if key in ['min_conf']:
                self.stats[key] = 1.0
            elif key in ['max_conf']:
                self.stats[key] = 0.0
            else:
                self.stats[key] = 0

    def _extract_sigma_from_model(self):
        try:
            for m in self.model.model.modules():
                if hasattr(m, 'get_last_prediction'):
                    pred = m.get_last_prediction()
                    if pred and 'sigma' in pred:
                        return float(pred['sigma'].mean().item())
        except:
            pass
        return None

    def draw_smart_visualization(self, img, x, y, sigma, status):
        overlay = img.copy()
        ix, iy = int(x), int(y)

        base_radius = 20
        scale_factor = 100.0
        current_sigma = sigma if sigma is not None else 0.5

        radius = int(base_radius + (current_sigma * scale_factor))
        radius = max(15, min(radius, 150))

        if 'Tracking' in status:
            halo_color = (0, 255, 100)
            center_color = (0, 255, 0)
        elif 'Inst-Guide' in status:
            halo_color = (0, 255, 255)
            center_color = (0, 165, 255)
        else:
            halo_color = (0, 100, 255)
            center_color = (0, 0, 255)

        cv2.circle(overlay, (ix, iy), radius, halo_color, -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.circle(img, (ix, iy), radius, halo_color, 1, cv2.LINE_AA)

        cv2.circle(img, (ix, iy), 4, (255, 255, 255), -1)
        cv2.drawMarker(img, (ix, iy), center_color, cv2.MARKER_CROSS, 15, 2)

        text = f"{status}"
        if sigma is not None:
            text += f" sigma:{sigma:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.5, 1)
        text_y = iy - 12
        if text_y < 20: text_y = iy + 30

        cv2.rectangle(img, (ix + 10, text_y - text_h - 3), (ix + 10 + text_w + 4, text_y + 3), (0, 0, 0), -1)
        cv2.putText(img, text, (ix + 12, text_y), font, 0.5, (255, 255, 255), 1)

    def detect_image(self, img_bgr):
        try:
            original_img = img_bgr.copy()
            self._frame_counter += 1

            self.stats['total_frames'] += 1

            results = self.model(img_bgr, verbose=False)

            has_papilla = False
            papilla_info = None
            instruments = []

            used_keypoint_this_frame = False

            if len(results[0].boxes) > 0:
                for idx, cls in enumerate(results[0].boxes.cls):
                    cls_id = int(cls)

                    if cls_id == 0:
                        temp_px, temp_py = 0, 0

                        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                            kpt = results[0].keypoints.data[idx].cpu().numpy()

                            if len(kpt) > 0:
                                kpt_x, kpt_y, kpt_conf = kpt[0][0], kpt[0][1], kpt[0][2]

                                if kpt_conf > 0.5:
                                    self.stats['keypoint_high_conf'] += 1
                                elif kpt_conf > 0.1:
                                    self.stats['keypoint_mid_conf'] += 1
                                else:
                                    self.stats['keypoint_low_conf'] += 1

                                if kpt_conf > 0.1:
                                    temp_px, temp_py = kpt_x, kpt_y
                                    used_keypoint_this_frame = True
                                    self.stats['use_keypoint'] += 1
                                else:
                                    bbox = results[0].boxes.xyxy[idx].cpu().numpy()
                                    temp_px, temp_py = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                                    self.stats['use_bbox_center'] += 1
                            else:
                                self.stats['no_keypoint_output'] += 1
                                bbox = results[0].boxes.xyxy[idx].cpu().numpy()
                                temp_px, temp_py = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                                self.stats['use_bbox_center'] += 1
                        else:
                            self.stats['no_keypoint_output'] += 1
                            bbox = results[0].boxes.xyxy[idx].cpu().numpy()
                            temp_px, temp_py = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                            self.stats['use_bbox_center'] += 1

                        if temp_px < 1.0 and temp_py < 1.0:
                            self.stats['filtered_by_zero_check'] += 1
                            continue

                        current_conf = float(results[0].boxes.conf[idx])
                        if not has_papilla or current_conf > papilla_info[2]:
                            papilla_info = (temp_px, temp_py, current_conf)
                            has_papilla = True

                            self.stats['total_conf'] += current_conf
                            self.stats['min_conf'] = min(self.stats['min_conf'], current_conf)
                            self.stats['max_conf'] = max(self.stats['max_conf'], current_conf)

                    else:
                        box = results[0].boxes.xyxy[idx].cpu().numpy()
                        conf = float(results[0].boxes.conf[idx])
                        instruments.append((box, conf, cls_id))

            if has_papilla:
                self.stats['has_papilla_detection'] += 1
            else:
                self.stats['no_papilla_detection'] += 1

            sigma = self._extract_sigma_from_model()

            curr_inst_center = self.get_instrument_center(instruments)
            inst_dx, inst_dy = 0.0, 0.0

            if self.last_instrument_center is not None and curr_inst_center is not None:
                inst_dx = curr_inst_center[0] - self.last_instrument_center[0]
                inst_dy = curr_inst_center[1] - self.last_instrument_center[1]

            if curr_inst_center is not None:
                self.last_instrument_center = curr_inst_center

            final_x, final_y = 0, 0
            status_text = ""

            if has_papilla:
                det_x, det_y, det_conf = papilla_info

                self.smart_tracker.update_with_sigma(
                    (det_x, det_y),
                    sigma=0.01,
                    conf=det_conf,
                    current_tick=self._frame_counter
                )

                final_x, final_y = det_x, det_y
                status_text = "Tracking"
                self.stats['tracking'] += 1

            else:
                pred = self.smart_tracker.predict_with_instrument_guide(inst_dx, inst_dy)

                if pred is not None:
                    final_x, final_y = pred

                    if abs(inst_dx) > 2.0 or abs(inst_dy) > 2.0:
                        status_text = "Inst-Guide"
                        self.stats['inst_guide'] += 1
                    else:
                        status_text = "Predicting"
                        self.stats['predicting'] += 1

                    if sigma is None: sigma = 0.5
                    sigma += 0.1 * self.smart_tracker.missed_ticks
                else:
                    status_text = "Lost"
                    self.stats['lost'] += 1

            result_img = img_bgr.copy()
            detection_boxes = []

            for box, conf, cls_id in instruments:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 191, 0), 2)

                label = f"Tool {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(result_img, (x1, y1 - 20), (x1 + t_size[0], y1), (255, 191, 0), -1)
                cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if status_text == "Inst-Guide" and (abs(inst_dx) > 1 or abs(inst_dy) > 1):
                cx, cy = int(final_x), int(final_y)
                cv2.arrowedLine(result_img, (cx, cy), (int(cx + inst_dx * 5), int(cy + inst_dy * 5)), (255, 255, 0), 2)

            if status_text != "Lost":
                self.draw_smart_visualization(result_img, final_x, final_y, sigma, status_text)

                detection_boxes.append({
                    'label': 'duodenal papilla',
                    'conf': papilla_info[2] if has_papilla else 0.5,
                    'center_x': final_x, 'center_y': final_y,
                    'sigma': sigma if sigma else 0.0,
                    'timestamp': time.time()
                })

            class_counts = {'papilla': 1 if status_text != "Lost" else 0, 'instrument': len(instruments)}

            return result_img, class_counts, status_text, original_img, detection_boxes

        except Exception as e:
            print(f"❌ Detect Error: {e}")
            import traceback
            traceback.print_exc()
            return img_bgr, {}, "Error", img_bgr, []

    def print_stats(self):
        total = self.stats['total_frames']
        if total == 0:
            print("⚠️ 尚未处理任何帧")
            return

        print(f"\n{'=' * 70}")
        print(f"📊 检测统计报告 (共 {total} 帧)")
        print(f"{'=' * 70}")

        print(f"\n【检测成功率】")
        detection_rate = self.stats['has_papilla_detection'] / total * 100
        print(f"  ✓ 成功检测到乳头: {self.stats['has_papilla_detection']} 帧 ({detection_rate:.1f}%)")
        print(f"  ✗ 未检测到乳头:   {self.stats['no_papilla_detection']} 帧 ({100 - detection_rate:.1f}%)")

        print(f"\n【定位来源分析】")
        if self.stats['has_papilla_detection'] > 0:
            kpt_rate = self.stats['use_keypoint'] / self.stats['has_papilla_detection'] * 100
            bbox_rate = self.stats['use_bbox_center'] / self.stats['has_papilla_detection'] * 100
            print(f"  🎯 使用关键点:     {self.stats['use_keypoint']} 次 ({kpt_rate:.1f}%)")

        print(f"\n【关键点质量分析】")
        total_kpt_attempts = (self.stats['keypoint_high_conf'] +
                              self.stats['keypoint_mid_conf'] +
                              self.stats['keypoint_low_conf'])
        if total_kpt_attempts > 0:
            high_rate = self.stats['keypoint_high_conf'] / total_kpt_attempts * 100
            mid_rate = self.stats['keypoint_mid_conf'] / total_kpt_attempts * 100
            low_rate = self.stats['keypoint_low_conf'] / total_kpt_attempts * 100
            print(f"  🟢 高置信度 (>0.5):  {self.stats['keypoint_high_conf']} 次 ({high_rate:.1f}%)")
            print(f"  🟡 中置信度 (0.1~0.5): {self.stats['keypoint_mid_conf']} 次 ({mid_rate:.1f}%)")
            print(f"  🔴 低置信度 (<0.1):  {self.stats['keypoint_low_conf']} 次 ({low_rate:.1f}%)")
        print(f"  ⚠️ 无关键点输出:    {self.stats['no_keypoint_output']} 次")

        print(f"\n【追踪状态分析】")
        tracking_rate = self.stats['tracking'] / total * 100
        guide_rate = self.stats['inst_guide'] / total * 100
        pred_rate = self.stats['predicting'] / total * 100
        lost_rate = self.stats['lost'] / total * 100
        print(f"  🟢 Tracking (检测算法): {self.stats['tracking']} 帧 ({tracking_rate:.1f}%)")
        print(f"  🟡 Inst-Guide (器械引导): {self.stats['inst_guide']} 帧 ({guide_rate:.1f}%)")
        print(f"  🟠 Predicting (惯性预测): {self.stats['predicting']} 帧 ({pred_rate:.1f}%)")
        print(f"  🔴 Lost (丢失):         {self.stats['lost']} 帧 ({lost_rate:.1f}%)")

        print(f"\n【检测框置信度分析】")
        if self.stats['has_papilla_detection'] > 0:
            avg_conf = self.stats['total_conf'] / self.stats['has_papilla_detection']
            print(f"  平均置信度: {avg_conf:.3f}")
            print(f"  最低置信度: {self.stats['min_conf']:.3f}")
            print(f"  最高置信度: {self.stats['max_conf']:.3f}")

        print(f"\n【异常情况统计】")
        print(f"  ⚠️ 被防零判断过滤: {self.stats['filtered_by_zero_check']} 次")

        print(f"\n【综合评分】")
        kpt_rate = self.stats['use_keypoint'] / self.stats['has_papilla_detection'] * 100 if self.stats['has_papilla_detection'] > 0 else 0
        score = tracking_rate * 0.5 + detection_rate * 0.3 + kpt_rate * 0.2 if self.stats['has_papilla_detection'] > 0 else 0
        print(f"  综合得分: {score:.1f}/100")
        if score >= 80:
            grade = "优秀 ⭐⭐⭐"
        elif score >= 60:
            grade = "良好 ⭐⭐"
        else:
            grade = "需改进 ⭐"
        print(f"  评级: {grade}")

        print(f"{'=' * 70}\n")

    def save_stats(self, filename="detection_stats.txt"):
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        self.print_stats()

        output = buffer.getvalue()
        sys.stdout = old_stdout

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(output)

        print(f"✓ 统计已保存到: {filename}")


class HeatmapAnalysisDialog(QDialog):
    def __init__(self, parent=None, heatmap_data=None, image_size=None):
        super().__init__(parent)
        self.setWindowTitle("Heatmap Analysis")
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        if heatmap_data:
            ax = self.figure.add_subplot(111)
            x = [d['center_x'] for d in heatmap_data]
            y = [d['center_y'] for d in heatmap_data]
            ax.scatter(x, y, c='r', alpha=0.1)
            ax.invert_yaxis()
            ax.set_title("Trajectory Heatmap")
            self.canvas.draw()


class EfficiencyAnalysisDialog(QDialog):
    def __init__(self, parent=None, detection_data=None):
        super().__init__(parent)
        self.setWindowTitle("Efficiency Analysis")
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        if detection_data:
            sigmas = [x['sigma'] for x in detection_data]
            ax = self.figure.add_subplot(111)
            ax.plot(sigmas, label='Uncertainty (Sigma)', color='orange')
            ax.set_title("Uncertainty Evaluation")
            ax.legend()
            self.canvas.draw()


class ImageDetectionApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("内镜智能感知系统")

        self.detector = Detector(opt.weights, opt.conf_thre, opt.iou_thre)

        self.video_path = None
        self.cap = None
        self.is_video_running = False
        self.detection_results = []
        self.video_speed = 1.0

        self.setup_ui_connections()

        if not hasattr(self, 'speed_slider'):
            self.setup_speed_control()

    def update_speed(self, value):
        if value < 1: value = 1
        self.video_speed = value / 10.0

    def setup_speed_control(self):
        self.speed_label = QtWidgets.QLabel("Speed:")
        self.statusbar.addWidget(self.speed_label)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 30)
        self.speed_slider.setValue(10)
        self.speed_slider.setFixedWidth(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        self.statusbar.addWidget(self.speed_slider)

    def setup_ui_connections(self):
        try:
            self.btn_select.clicked.disconnect()
            self.btn_detect_image.clicked.disconnect()
            self.btn_detect_video.clicked.disconnect()
            self.btn_detect_real_time.clicked.disconnect()
            self.btn_save.clicked.disconnect()
            self.btn_heatmap_analysis.clicked.disconnect()
            self.btn_efficiency_analysis.clicked.disconnect()
        except:
            pass

        self.btn_select.clicked.connect(self.open_file)
        self.btn_detect_image.clicked.connect(self.detect_image)
        self.btn_detect_video.clicked.connect(self.detect_video)
        self.btn_detect_real_time.clicked.connect(self.detect_real_time)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_heatmap_analysis.clicked.connect(self.show_heatmap_analysis)
        self.btn_efficiency_analysis.clicked.connect(self.show_efficiency_analysis)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Media (*.jpg *.png *.mp4 *.avi)")
        if not path: return

        if path.endswith(('.mp4', '.avi')):
            self.video_path = path
            self.btn_detect_video.setEnabled(True)
            self.btn_detect_image.setEnabled(False)
            self.label_detection_result.setText(f"Loaded Video: {os.path.basename(path)}")
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if ret: self.display_image(frame)
            cap.release()
        else:
            self.current_frame = cv2.imread(path)
            self.btn_detect_image.setEnabled(True)
            self.btn_detect_video.setEnabled(False)
            self.display_image(self.current_frame)
            self.label_detection_result.setText(f"Loaded Image: {os.path.basename(path)}")

    def detect_image(self):
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        self.detector.reset_all_memory()
        self.detector.reset_stats()

        result_img, class_counts, status, original, detections = self.detector.detect_image(self.current_frame)
        self.result_frame = result_img
        self.detection_results = detections

        self.display_image(result_img, target='result')
        self.update_detection_info(class_counts, status)

    def detect_video(self):
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please load a video first!")
            return

        if self.is_video_running:
            self.is_video_running = False
            self.btn_detect_video.setText("Detect Video")
            if self.cap:
                self.cap.release()
            self.detector.print_stats()
            return

        self.is_video_running = True
        self.btn_detect_video.setText("Stop")
        self.detection_results = []

        self.detector.reset_all_memory()
        self.detector.reset_stats()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open video file!")
            self.is_video_running = False
            return

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while self.is_video_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            result_img, class_counts, status, original, detections = self.detector.detect_image(frame)
            self.detection_results.extend(detections)

            self.display_image(result_img, target='result')
            self.update_detection_info(class_counts, status, frame_count, total_frames)

            delay = int(max(1, (1000 / fps) / self.video_speed))
            cv2.waitKey(delay)
            QApplication.processEvents()

        self.cap.release()
        self.is_video_running = False
        self.btn_detect_video.setText("Detect Video")

        self.detector.print_stats()
        self.label_detection_result.setText(f"Video detection completed! Processed {frame_count} frames.")

    def detect_real_time(self):
        if self.is_video_running:
            self.is_video_running = False
            self.btn_detect_real_time.setText("Real-time Detection")
            if self.cap:
                self.cap.release()
            self.detector.print_stats()
            return

        self.is_video_running = True
        self.btn_detect_real_time.setText("Stop")
        self.detection_results = []

        self.detector.reset_all_memory()
        self.detector.reset_stats()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open camera!")
            self.is_video_running = False
            self.btn_detect_real_time.setText("Real-time Detection")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_count = 0
        start_time = time.time()

        while self.is_video_running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_count += 1

            result_img, class_counts, status, original, detections = self.detector.detect_image(frame)
            self.detection_results.extend(detections)

            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            self.display_image(result_img, target='result')
            self.update_detection_info(class_counts, status, frame_count, fps=current_fps)

            cv2.waitKey(1)
            QApplication.processEvents()

        self.cap.release()
        self.is_video_running = False
        self.btn_detect_real_time.setText("Real-time Detection")
        self.detector.print_stats()

    def save_image(self):
        if not hasattr(self, 'result_frame') or self.result_frame is None:
            QMessageBox.warning(self, "Warning", "No detection result to save!")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Images (*.jpg *.png)"
        )

        if save_path:
            cv2.imwrite(save_path, self.result_frame)
            QMessageBox.information(self, "Success", f"Image saved to:\n{save_path}")

            stats_path = save_path.rsplit('.', 1)[0] + "_stats.txt"
            self.detector.save_stats(stats_path)

    def show_heatmap_analysis(self):
        if not self.detection_results:
            QMessageBox.warning(self, "Warning", "No detection data available for analysis!")
            return

        dialog = HeatmapAnalysisDialog(
            self,
            heatmap_data=self.detection_results,
            image_size=(1280, 720)
        )
        dialog.exec_()

    def show_efficiency_analysis(self):
        if not self.detection_results:
            QMessageBox.warning(self, "Warning", "No detection data available for analysis!")
            return

        dialog = EfficiencyAnalysisDialog(
            self,
            detection_data=self.detection_results
        )
        dialog.exec_()

    def display_image(self, img, target='original'):
        if img is None:
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w

        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        if target == 'result':
            label = self.label_result if hasattr(self, 'label_result') else self.label_detection_result
        else:
            label = self.label_original if hasattr(self, 'label_original') else self.label_detection_result

        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def update_detection_info(self, class_counts, status, frame_num=None, total_frames=None, fps=None):
        info_lines = []

        if frame_num is not None:
            if total_frames is not None:
                progress = frame_num / total_frames * 100
                info_lines.append(f"Frame: {frame_num}/{total_frames} ({progress:.1f}%)")
            else:
                info_lines.append(f"Frame: {frame_num}")

        if fps is not None:
            info_lines.append(f"FPS: {fps:.1f}")

        info_lines.append(f"Status: {status}")

        if class_counts:
            papilla_count = class_counts.get('papilla', 0)
            instrument_count = class_counts.get('instrument', 0)
            info_lines.append(f"Papilla: {papilla_count} | Instruments: {instrument_count}")

        stats = self.detector.stats
        if stats['total_frames'] > 0:
            detection_rate = stats['has_papilla_detection'] / stats['total_frames'] * 100
            tracking_rate = stats['tracking'] / stats['total_frames'] * 100
            info_lines.append(f"Detection Rate: {detection_rate:.1f}% | Tracking: {tracking_rate:.1f}%")

        if hasattr(self, 'text_info'):
            self.text_info.setText('\n'.join(info_lines))
        else:
            self.statusBar().showMessage(' | '.join(info_lines))

    def closeEvent(self, event):
        self.is_video_running = False
        if self.cap:
            self.cap.release()

        self.detector.print_stats()

        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.is_video_running = False
        elif event.key() == Qt.Key_Space:
            if hasattr(self, 'is_paused'):
                self.is_paused = not self.is_paused
        elif event.key() == Qt.Key_R:
            self.detector.reset_all_memory()
            self.detector.reset_stats()
            self.statusBar().showMessage("Memory and stats reset!", 3000)
        elif event.key() == Qt.Key_S:
            self.save_image()
        elif event.key() == Qt.Key_P:
            self.detector.print_stats()

        super().keyPressEvent(event)


class MainApplication:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setStyle('Fusion')

        self.setup_stylesheet()

        self.login_form = None
        self.main_window = None

    def setup_stylesheet(self):
        stylesheet = """
            QMainWindow {
                background-color: #2b2b2b;
            }
            QPushButton {
                background-color: #3c3f41;
                color: white;
                border: 1px solid #555;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4a4d4f;
            }
            QPushButton:pressed {
                background-color: #2d5a88;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #555;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2d5a88;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QStatusBar {
                background-color: #3c3f41;
                color: white;
            }
            QMessageBox {
                background-color: #2b2b2b;
            }
            QMessageBox QLabel {
                color: white;
            }
        """
        self.app.setStyleSheet(stylesheet)

    def show_login(self):
        try:
            self.login_form = LoginForm()
            self.login_form.login_successful.connect(self.on_login_success)
            self.login_form.show()
        except Exception as e:
            print(f"Login form not available: {e}")
            self.on_login_success()

    def on_login_success(self):
        if self.login_form:
            self.login_form.close()

        self.main_window = ImageDetectionApp()
        self.main_window.showMaximized()

    def run(self):
        self.show_login()
        return self.app.exec_()


def main():
    print("=" * 60)
    print("🔬 内镜智能感知系统 - Smart Kalman Edition")
    print("=" * 60)
    print(f"📦 PyTorch Version: {torch.__version__}")
    print(f"🖥️  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📁 Model Path: {opt.weights}")
    print(f"🎯 Confidence Threshold: {opt.conf_thre}")
    print(f"📐 IoU Threshold: {opt.iou_thre}")
    print("=" * 60)

    try:
        application = MainApplication()
        sys.exit(application.run())

    except Exception as e:
        print(f"❌ Application Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

