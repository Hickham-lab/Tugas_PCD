import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.uic import loadUi
from roboflow import Roboflow
import threading


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super(ObjectDetectionApp, self).__init__()
        loadUi('Gui2.ui', self)

        try:
            self.rf = Roboflow(api_key="XV0H7bOYCoXCYJxxLWjh")
            self.project = self.rf.workspace("viktorikus").project("rock-paper-scissors-detection-qy2cs")
            self.model = self.project.version(9).model
            print("Roboflow model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Roboflow Error", f"Failed to load model: {str(e)}")
            self.model = None

        self.processing = False
        self.frame_count = 0
        self.cam = None
        self.camera_timer = None
        self.camera_label = None
        self.camera_window = None

        self.detectionButton.clicked.connect(self.object_detection)


    def object_detection(self):
        if self.model is None:
            QMessageBox.warning(self, "Error", "Roboflow model not loaded!")
            return

        self.setup_camera_window()

        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cam.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera!")
            return

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.camera_timer.start(30)

        self.last_frame = None
        self.last_prediction = None
        self.has_prediction = False
        self.camera_window.show()

    def setup_camera_window(self):
        self.camera_window = QMainWindow()
        self.camera_window.setWindowTitle("Object Detection - Roboflow")
        self.camera_window.resize(800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label)

        btn_close = QPushButton("Close Camera")
        btn_close.clicked.connect(self.stop_camera)
        layout.addWidget(btn_close)

        central_widget.setLayout(layout)
        self.camera_window.setCentralWidget(central_widget)

    def stop_camera(self):
        if self.camera_timer and self.camera_timer.isActive():
            self.camera_timer.stop()
        if self.cam and self.cam.isOpened():
            self.cam.release()
        if self.camera_window:
            self.camera_window.close()

    def update_camera_frame(self):
        if not self.cam or not self.cam.isOpened():
            return

        ret, frame = self.cam.read()
        if not ret:
            return

        self.last_frame = frame.copy()

        if not self.processing:
            self.processing = True
            threading.Thread(target=self.process_frame, args=(frame,), daemon=True).start()

        self.display_current_frame()

    def display_current_frame(self):
        if self.last_frame is None:
            return

        if self.has_prediction and self.last_prediction is not None:
            display_frame = self.last_prediction
        else:
            gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def process_frame(self, frame):
        try:
            resized = cv2.resize(frame, (224, 224))
            rgb_small = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            prediction = self.model.predict(rgb_small, confidence=20, overlap=30).json()

            h_ratio = frame.shape[0] / 224
            w_ratio = frame.shape[1] / 224

            display_frame = frame.copy()
            predictions = prediction['predictions']

            for pred in predictions:
                x = int(pred['x'] * w_ratio)
                y = int(pred['y'] * h_ratio)
                w = int(pred['width'] * w_ratio)
                h = int(pred['height'] * h_ratio)

                cv2.rectangle(display_frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{pred['class']} {pred['confidence']:.2f}",
                            (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if len(predictions) >= 2:
                p1 = predictions[0]
                p2 = predictions[1]

                class1 = p1['class']
                class2 = p2['class']

                winner = self.determine_winner(class1, class2)

                if winner == "Player 1":
                    loser_box = p2
                    winner_name = class1
                elif winner == "Player 2":
                    loser_box = p1
                    winner_name = class2
                else:
                    loser_box = None
                    winner_name = "Draw"

                if loser_box:
                    x = int(loser_box['x'] * w_ratio)
                    y = int(loser_box['y'] * h_ratio)
                    w = int(loser_box['width'] * w_ratio)
                    h = int(loser_box['height'] * h_ratio)

                    x1 = max(x - w // 2, 0)
                    y1 = max(y - h // 2, 0)
                    x2 = min(x + w // 2, frame.shape[1])
                    y2 = min(y + h // 2, frame.shape[0])

                    cropped = frame[y1:y2, x1:x2]
                    dominant_color = self.get_dominant_color(cropped)
                    cv2.rectangle(display_frame, (0, 0), (100, 100), dominant_color, -1)
                else:
                    dominant_color = (0, 0, 0)

                cv2.putText(display_frame, f"Winner: {winner_name}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)

            self.last_prediction = display_frame
            self.has_prediction = True

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            self.has_prediction = False
        finally:
            self.processing = False
            self.display_current_frame()

    def get_dominant_color(self, image):
        if image.size == 0:
            return (0, 0, 0)

        crop = cv2.resize(image, (50, 50))
        data = crop.reshape((-1, 3))
        data = np.float32(data)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        b = int(centers[0][0])
        g = int(centers[0][1])
        r = int(centers[0][2])

        return (b, g, r)

    @staticmethod
    def determine_winner(p1, p2):
        p1_lower = p1.lower()
        p2_lower = p2.lower()

        rules = {
            'rock': 'scissors',
            'scissors': 'paper',
            'paper': 'rock'
        }

        if p1_lower == p2_lower:
            return "Draw"

        if rules.get(p1_lower) == p2_lower:
            return "Player 1"
        else:
            return "Player 2"

    def closeEvent(self, event):
        if hasattr(self, 'cam') and self.cam.isOpened():
            self.cam.release()
        if hasattr(self, 'camera_timer') and self.camera_timer.isActive():
            self.camera_timer.stop()
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    window = ObjectDetectionApp()
    window.setWindowTitle('Object Detection App')
    window.show()
    sys.exit(app.exec_())
