from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime
import numpy as np


model = YOLO('yolov8n.pt')  
AIRPLANE_CLASS_ID = 4

def log_detection(filename: str, count: int):
    log_entry = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "file": filename,
        "airplane_count": count
    }

    history_file = "history.json"
    history = []

    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append(log_entry)

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def detect_airplanes(image_path: str, save_path: str = "output.jpg") -> int:
    results = model(image_path, classes=[AIRPLANE_CLASS_ID])[0]
    boxes = results.boxes
    airplane_boxes = [box for box in boxes if int(box.cls) == AIRPLANE_CLASS_ID]

    img = results.orig_img.copy()
    for box in airplane_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        label = f"airplane {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(save_path, img)
    log_detection(os.path.basename(image_path), len(airplane_boxes))
    return len(airplane_boxes)


def is_new_box(box, prev_boxes, threshold=50):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    for px, py in prev_boxes:
        if np.sqrt((cx - px)**2 + (cy - py)**2) < threshold:
            return False
    return True

def detect_airplanes_in_video(video_path: str, output_path: str = "output_video.avi") -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    unique_airplanes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, classes=[AIRPLANE_CLASS_ID], verbose=False)[0]
        airplane_boxes = [box for box in results.boxes if int(box.cls) == AIRPLANE_CLASS_ID]

        for box in airplane_boxes:
            if is_new_box(box, unique_airplanes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                unique_airplanes.append((cx, cy))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"airplane {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    log_detection(os.path.basename(video_path), len(unique_airplanes))
    return len(unique_airplanes)




IMAGE_PATH = "C:\\Users\\klyos\\OneDrive\\Рабочий стол\\Practice\\airplanes\\plane2.jpg"
OUTPUT_PATH = "C:\\Users\\klyos\\OneDrive\\Рабочий стол\\Practice\\airplanes\\out\\output.jpg"
count = detect_airplanes(image_path=IMAGE_PATH,
                 save_path=OUTPUT_PATH)
print(f'\nОбнаружено: {count} самолетов \n')






