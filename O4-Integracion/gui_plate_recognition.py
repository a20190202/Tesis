import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from tkinter import Tk, filedialog, Button, Label
import threading
import time

# Load models globally
model_car = None
model_plate = None
ocr = None
thread_lock = threading.Lock()
active_thread = None


def preprocess_for_ocr(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.dilate(thresh, kernel, iterations=1)
    inverted = cv2.bitwise_not(morph)
    return inverted


def run_plate_recognition(video_path):
    print(f"[DEBUG] Starting processing thread for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    frame_counter = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[DEBUG] End of video or read error.")
                break

            use_persistence = frame_counter > 5
            car_results = model_car.track(frame, persist=use_persistence, tracker="bytetrack.yaml")[0]
            frame_counter += 1

            if car_results.boxes is None:
                continue

            car_boxes = car_results.boxes.xyxy.cpu().numpy()
            car_confs = car_results.boxes.conf.cpu().numpy()
            car_classes = car_results.boxes.cls.cpu().numpy().astype(int)
            track_ids = car_results.boxes.id.cpu().numpy().astype(int) if car_results.boxes.id is not None else [-1] * len(car_boxes)
            class_names = model_car.names

            for box, conf, cls_id, track_id in zip(car_boxes, car_confs, car_classes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f'{class_names[cls_id]} {conf:.2f} ID:{track_id}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                car_crop = frame[y1:y2, x1:x2]
                plate_results = model_plate.track(car_crop, persist=False, tracker="bytetrack.yaml")[0]
                if plate_results.boxes is None:
                    continue

                plate_boxes = plate_results.boxes.xyxy.cpu().numpy()
                for pbox in plate_boxes:
                    px1, py1, px2, py2 = map(int, pbox)
                    roi = car_crop[py1:py2, px1:px2]
                    processed = preprocess_for_ocr(roi)
                    img_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                    ocr_result = ocr.ocr(img_rgb, cls=False)

                    abs_px1, abs_py1 = x1 + px1, y1 + py1
                    abs_px2, abs_py2 = x1 + px2, y1 + py2
                    cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)

                    if ocr_result and len(ocr_result[0]) > 0:
                        for line in ocr_result[0]:
                            text = line[1][0].strip()
                            score = line[1][1]
                            if len(text) >= 4 and "-" in text:
                                print(f'Detected: "{text}" with confidence: {score:.2f}')
                                cv2.putText(frame, text, (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                break


            cv2.imshow("Video with Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[DEBUG] User pressed 'q'.")
                break

    except Exception as e:
        print(f"[ERROR] Exception in video thread: {e}")

    finally:
        print("[DEBUG] Cleaning up after video.")
        cap.release()
        cv2.destroyAllWindows()
        print("[DEBUG] Thread for video finished.")


def launch_gui():
    def select_video():
        global active_thread
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if path:
            if active_thread and active_thread.is_alive():
                print("[WARNING] A video is already processing.")
                return

            def process():
                with thread_lock:
                    run_plate_recognition(path)

            root.withdraw()   # Hide GUI
            run_plate_recognition(path)
            root.deiconify()  # Show GUI again

    root = Tk()
    root.title("YOLO + OCR Plate Recognition")
    root.geometry("300x120")

    Label(root, text="Upload a video to analyze").pack(pady=10)
    Button(root, text="Select Video", command=select_video).pack(pady=5)
    Button(root, text="Exit", command=root.quit).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python gui_plate_recognition.py <car_model.pt> <plate_model.pt>")
    else:
        print("[INFO] Loading YOLO models and OCR...")
        model_car = YOLO(sys.argv[1])
        model_plate = YOLO(sys.argv[2])
        ocr = PaddleOCR(use_angle_cls=False, lang='en')
        print("[INFO] Models loaded. Launching GUI.")
        launch_gui()
