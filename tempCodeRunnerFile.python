import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"D:\Downloads\Fire-Detection-using-YOLOv8-main\Fire-Detection-using-YOLOv8-main\best.pt")

# Use CAP_DSHOW backend for video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run prediction
    results = model.predict(source=frame, conf=0.4, save=True, show=True)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
