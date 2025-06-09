import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, Canvas, filedialog
import threading
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR


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

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.MP4 *.avi")])
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_counter = 0
        self.running = True

        # 🔽 Read first frame immediately
        success, frame = self.cap.read()
        if success:
            self.current_frame = frame
            self.frame_counter = 1  # First frame already read
            self.display_frame(frame)  # ⬅️ New helper method to show it

        if not self.loop_running:
            self.loop_running = True
            self.root.after(30, self.play_video)

    def display_frame(self, frame):
        # Resize frame to fit canvas
        h, w = frame.shape[:2]
        scale = min(self.canvas_width / w, self.canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))

        # Convert to ImageTk and show
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas.image = imgtk

        # Update time label
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

        if not self.paused and self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                self.running = False
                self.loop_running = False
                return

            self.frame_counter += 1
            results = self.car_model.track(frame, persist=True, tracker="bytetrack.yaml")[0]

            if results.boxes:
                for box in results.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            self.current_frame = frame
        elif hasattr(self, 'current_frame'):
            frame = self.current_frame

        # If we have a frame (either new or from pause), show it
        if frame is not None:
            self.display_frame(frame)

        # Always continue looping
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
