import cv2
import os
from ultralytics import YOLO

# ----------------------- Configurations ----------------------- #
MODEL_PATH = r"C:\Users\dspvi\OneDrive\Desktop\Animal_Classification\best.pt"
VIDEO_PATH = r"C:\Users\dspvi\Downloads\855538-hd_1920_1080_25fps.mp4"
USE_WEBCAM = False  # Set True to use webcam

CONFIDENCE_THRESHOLD = 0.5
RESIZED_WIDTH = 640
RESIZED_HEIGHT = 360
WINDOW_NAME = "Real-Time Animal Detection"

# -------------------- Load YOLOv8 Model ----------------------- #
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[Error] Model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# ------------------- Initialize Video Source ------------------ #
cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)

if not cap.isOpened():
    raise IOError("[Error] Could not open video source")

# ----------------------- Main Loop ---------------------------- #
while True:
    ret, frame = cap.read()
    if not ret:
        print("[Info] End of video or cannot read frame.")
        break

    # Resize the frame
    frame_resized = cv2.resize(frame, (RESIZED_WIDTH, RESIZED_HEIGHT))

    # YOLOv8 prediction
    results = model.predict(source=frame_resized, conf=CONFIDENCE_THRESHOLD, save=False)[0]

    # Draw detections
    annotated_frame = results.plot()

    # Display results
    cv2.imshow(WINDOW_NAME, annotated_frame)

    # Exit on 'q' or ESC key
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

# ----------------------- Cleanup ------------------------------ #
cap.release()
cv2.destroyAllWindows()