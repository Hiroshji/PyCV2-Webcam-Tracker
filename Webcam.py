import cv2
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
from playsound import playsound
import threading

def main():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt").to(device)
    if device == "cuda":
        model.model.half()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "YOLOv8 Real-Time Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def play_sound():
        playsound("SoundEffects/SpaceBoyfriendTheme.mp3")

    phone_detected = False  # track phone detection status

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        with torch.no_grad():
            if device == "cuda":
                frame = frame.astype('float16')
            results = model.predict(frame, conf=0.5, iou=0.45, device=device, imgsz=720)

        found_phone = False
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = box.conf[0]
            if label.lower() == "cell phone":
                found_phone = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        # Play sound once only when the phone is newly detected
        if found_phone and not phone_detected:
            threading.Thread(target=play_sound, daemon=True).start()
            phone_detected = True
        elif not found_phone:
            phone_detected = False

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
