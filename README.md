# Python Real-Time Object Detection

A Python-based real-time object detection application using the YOLOv8 model, OpenCV, and PyTorch. The program captures video from a webcam, processes each frame to detect objects, and displays bounding boxes and confidence scores for detected objects in fullscreen mode.

## Features
- Real-time object detection using **YOLOv8**.
- Supports **GPU acceleration** with PyTorch and CUDA.
- Displays detected objects with labels and confidence scores.
- Resizeable Window.

## Requirements
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- pygame

> [!NOTE]
> Anything above version 3.12.4 for python will not work given that not every library is updated for said verison)
Install the required libraries.
```bash
pip install opencv-python torch torchvision torchaudio ultralytics pygame
```

## Configuration
This section explains how to tweak specific settings in the script to modify its behavior and performance based on your preferences or hardware capabilities.

### Change Precision (16-bit or 32-bit)
By default, the model uses 32-bit precision. To improve performance on GPUs, you can switch to 16-bit precision. Modify the following lines in the code:
> [!NOTE]
> The file is currently on 16 bit it can be changed by removing "float16" and leaving it blank.

- **32-bit Precision (Default):**
  ```python
  if device == "cuda":
      model.model.half()```
- 16-bit Frame Processing:
  ```Python
  if device == "cuda":
    frame = frame.astype("float16")```
> Tip: Use 16-bit precision if your GPU has limited VRAM or if you want faster inference times. However, 32-bit precision provides slightly better accuracy.

### Webcam Resolution
Set the resolution of the webcam input using these lines:

```Python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
```







