import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR


def preprocess_for_ocr(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.dilate(thresh, kernel, iterations=1)
    inverted = cv2.bitwise_not(morph)
    return inverted


def run_plate_recognition(video_path, car_model_weights, plate_model_weights):
    model_car = YOLO(car_model_weights)
    model_plate = YOLO(plate_model_weights)
    ocr = PaddleOCR(use_angle_cls=False, lang='en')

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        car_results = model_car.track(frame, persist=True, tracker="bytetrack.yaml")[0]
        if car_results.boxes is None:
            continue

        car_boxes = car_results.boxes.xyxy.cpu().numpy()
        car_confs = car_results.boxes.conf.cpu().numpy()
        car_classes = car_results.boxes.cls.cpu().numpy().astype(int)
        track_ids = car_results.boxes.id.cpu().numpy().astype(int) if car_results.boxes.id is not None else [-1] * len(car_boxes)
        class_names = model_car.names

        for i, (box, conf, cls_id, track_id) in enumerate(zip(car_boxes, car_confs, car_classes, track_ids)):
            x1, y1, x2, y2 = map(int, box)
            label = f'{class_names[cls_id]} {conf:.2f} ID:{track_id}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            car_crop = frame[y1:y2, x1:x2]
            plate_results = model_plate.track(car_crop, persist=True, tracker="bytetrack.yaml")[0]
            if plate_results.boxes is None:
                continue

            plate_boxes = plate_results.boxes.xyxy.cpu().numpy()
            for pbox in plate_boxes:
                px1, py1, px2, py2 = map(int, pbox)
                roi = car_crop[py1:py2, px1:px2]
                processed = preprocess_for_ocr(roi)
                img_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                ocr_result = ocr.ocr(img_rgb, cls=False)

                # Draw plate box on full frame
                abs_px1, abs_py1 = x1 + px1, y1 + py1
                abs_px2, abs_py2 = x1 + px2, y1 + py2
                cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)

                # Draw OCR text
                if ocr_result and len(ocr_result[0]) > 0:
                    for line in ocr_result[0]:
                        text = line[1][0].strip()
                        score = line[1][1]
                        if len(text) >= 4 and "-" in text:
                            print(f'Detected: "{text}" with confidence: {score:.2f}')
                            cv2.putText(frame, text, (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            break

                cv2.imshow("Processed Plate ROI", processed)

        cv2.imshow("Video with Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python plate_recognition.py <video_path> <car_model.pt> <plate_model.pt>")
    else:
        run_plate_recognition(sys.argv[1], sys.argv[2], sys.argv[3])
