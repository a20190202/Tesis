import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, Canvas, filedialog, Scale
import threading
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re


class VideoApp:
    def __init__(self, root, car_model, plate_model, ocr_model):
        self.root = root
        self.root.title("YOLO + OCR Plate Recognition")
        self.car_model = car_model
        self.plate_model = plate_model
        self.ocr = ocr_model
        self.cap = None
        self.paused = True
        self.frame_counter = 0
        self.fps = 0
        self.total_frames = 0

        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.label_time = Label(root, text="00:00 / 00:00")
        self.label_time.pack()

        self.btn_play_pause = Button(root, text="Play", command=self.toggle_play)
        self.btn_play_pause.pack()

        self.btn_select = Button(root, text="Select Video", command=self.load_video)
        self.btn_select.pack()

        self.running = False
        self.loop_running = False
        self.user_seeking = False
        self.progress_scale = None

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.MP4 *.avi")])
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_counter = 0
        self.running = True

        if self.progress_scale:
            self.progress_scale.destroy()

        self.progress_scale = Scale(self.root, from_=0, to=self.total_frames - 1,
                                    orient="horizontal", length=self.canvas_width,
                                    command=self.on_seek)
        self.progress_scale.bind("<Button-1>", lambda e: self.set_user_seeking(True))
        self.progress_scale.bind("<ButtonRelease-1>", lambda e: self.set_user_seeking(False))
        self.progress_scale.pack()

        success, frame = self.cap.read()
        if success:
            self.current_frame = frame
            self.frame_counter = 1
            self.display_frame(frame)

        if not self.loop_running:
            self.loop_running = True
            self.root.after(30, self.play_video)

    def set_user_seeking(self, seeking):
        self.user_seeking = seeking

    def on_seek(self, value):
        if self.cap and self.user_seeking:
            self.paused = True
            self.frame_counter = int(value)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_counter)
            success, frame = self.cap.read()
            if success:
                self.current_frame = frame
                self.display_frame(frame)

    def normalize_text(self, text):
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def display_frame(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.canvas_width / w, self.canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas.image = imgtk

        current_sec = self.frame_counter / self.fps
        total_sec = self.total_frames / self.fps
        self.label_time.config(
            text=f"{int(current_sec//60):02}:{int(current_sec%60):02} / {int(total_sec//60):02}:{int(total_sec%60):02}"
        )

    def toggle_play(self):
        if not self.cap:
            return
        self.paused = not self.paused
        self.btn_play_pause.config(text="Pause" if not self.paused else "Play")

    def preprocess_for_ocr(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        morph = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        return cv2.bitwise_not(morph)

    def play_video(self):
        if not self.cap:
            return

        frame = None
        threshold = 0.7

        if not self.paused and self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                self.running = False
                self.loop_running = False
                return

            self.frame_counter += 1

            if self.progress_scale and not self.user_seeking:
                self.progress_scale.set(self.frame_counter)

            font_scale = max(frame.shape[0] / 720, 0.3)
            font_thickness = max(int(frame.shape[0] / 720), 4)
            rect_thickness = max(int(frame.shape[0] / 720), 2)

            car_result = self.car_model.track(frame, persist=True, tracker="bytetrack.yaml")[0]

            if car_result.boxes is not None:
                car_boxes = car_result.boxes.xyxy.cpu().numpy()
                car_confs = car_result.boxes.conf.cpu().numpy()
                car_classes = car_result.boxes.cls.cpu().numpy().astype(int)
                class_names = self.car_model.names
                for box_car, conf, cls_id in zip(car_boxes, car_confs, car_classes):
                    x1, y1, x2, y2 = map(int, box_car)
                    label = f"{class_names[cls_id]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), rect_thickness)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)

                    car_roi = frame[y1:y2, x1:x2]
                    plate_result = self.plate_model.track(car_roi, persist=True, tracker="bytetrack.yaml")[0]

                    if plate_result.boxes is None:
                        continue

                    plate_boxes = plate_result.boxes.xyxy.cpu().numpy()
                    for box_plate in plate_boxes:
                        px1, py1, px2, py2 = map(int, box_plate)
                        abs_px1, abs_py1 = x1 + px1, y1 + py1
                        abs_px2, abs_py2 = x1 + px2, y1 + py2
                        cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), rect_thickness)
                        plate_roi = car_roi[py1:py2, px1:px2]
                        processed = self.preprocess_for_ocr(plate_roi)

                        if len(processed.shape) == 2:
                            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

                        ocr_result = self.ocr.ocr(processed, cls=False)

                        if ocr_result and len(ocr_result[0]) > 0:
                            for line in ocr_result[0]:
                                raw_text = line[1][0].strip()
                                score = line[1][1]
                                norm_text = self.normalize_text(raw_text)

                                if 5 <= len(norm_text) <= 6 and score > threshold and "PERU" not in norm_text:
                                    cv2.putText(frame, norm_text, (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                font_scale, (0, 255, 255), font_thickness)
                                    break

            self.current_frame = frame

        elif hasattr(self, 'current_frame'):
            frame = self.current_frame

        if frame is not None:
            self.display_frame(frame)

        if self.loop_running:
            self.root.after(30, self.play_video)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python gui_plate_v2.py <car_model.pt> <plate_model.pt>")
    else:
        model_car = YOLO(sys.argv[1])
        model_plate = YOLO(sys.argv[2])
        ocr = PaddleOCR(use_angle_cls=False, lang='en')

        root = Tk()
        app = VideoApp(root, model_car, model_plate, ocr)
        root.mainloop()
