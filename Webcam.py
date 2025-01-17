import cv2
import torch
from ultralytics import YOLO

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt").to(device)
    if device == "cuda":
        model.model.half()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "YOLOv8 Real-Time Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Make window resizable

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        with torch.no_grad():
            if device == "cuda":
                frame = frame.astype('float16')
            results = model.predict(frame, conf=0.5, iou=0.45, device=device)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = box.conf[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Get current window dimensions and resize frame to match
        x, y, w, h = cv2.getWindowImageRect(window_name)
        if w > 0 and h > 0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
