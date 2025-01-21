# Imports
import cv2
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import pygame

# Initialization
def main():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    cudnn.benchmark = True

    # Initialize audio
    pygame.mixer.init()

    # Function Definitions
    def start_phone_sound():
        pygame.mixer.music.load("SoundEffects/HATSUNE.mp3")
        pygame.mixer.music.play(-1)  # loop indefinitely

    def stop_phone_sound():
        pygame.mixer.music.stop()

    def start_bottle_sound():
        pygame.mixer.music.load("SoundEffects/ANDROID.mp3")
        pygame.mixer.music.play(-1)  # loop indefinitely

    def stop_bottle_sound():
        pygame.mixer.music.stop()

    # Model Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt").to(device)
    if device == "cuda":
        model.model.half()

    # Video Capture Setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    window_name = "YOLOv8 Real-Time Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    phone_detected = False
    bottle_detected = False  # Initialize bottle detection flag

    # Define colors for different labels
    COLORS = {
        "cell phone": (0, 0, 255),  # Red
        "bottle": (0, 255, 0),      # Green
        # Add more labels and colors as needed
    }
    default_color = (255, 0, 0)  # Blue

    # Main Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        with torch.no_grad():
            if device == "cuda":
                frame = frame.astype('float16')
            results = model.predict(frame, conf=0.5, iou=0.45, device=device, imgsz=1088)

        found_phone = False
        found_bottle = False  # Initialize found_bottle flag
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = box.conf[0]
            color = COLORS.get(label.lower(), default_color)
            if label.lower() == "cell phone":
                found_phone = True
            elif label.lower() == "bottle":
                found_bottle = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Start/Stop phone sound
        if found_phone and not phone_detected:
            start_phone_sound()
            phone_detected = True
        elif not found_phone and phone_detected:
            stop_phone_sound()
            phone_detected = False

        # Start/Stop bottle sound
        if found_bottle and not bottle_detected:
            start_bottle_sound()
            bottle_detected = True
        elif not found_bottle and bottle_detected:
            stop_bottle_sound()
            bottle_detected = False

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Cleanup
    stop_phone_sound()
    stop_bottle_sound()
    cap.release()
    cv2.destroyAllWindows()

# Entry Point
if __name__ == "__main__":
    main()