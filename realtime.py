from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO(r'C:\Users\dspvi\OneDrive\Desktop\Animal_Classification\best.pt')

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Optional: Set the resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Perform inference using YOLOv8
    results = model.predict(source=frame, conf=0.25, save=False, verbose=False)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Show the annotated frame
    cv2.imshow("Real-Time Animal Detection", annotated_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()