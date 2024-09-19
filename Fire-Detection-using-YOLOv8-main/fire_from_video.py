import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"D:\Downloads\Fire-Detection-using-YOLOv8-main\Fire-Detection-using-YOLOv8-main\best.pt")

# Video file path
path = r"D:\Downloads\Fire-Detection-using-YOLOv8-main\Fire-Detection-using-YOLOv8-main\Free VFX Asset_ Fire Elements.mp4"

# Open the video using OpenCV
video = cv2.VideoCapture(path)

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        print("End of video or cannot read the video.")
        break

    # Run prediction on the current frame
    results = model.predict(source=frame, conf=0.2, save=False, show=False)

    for result in results:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Draw a red rectangle around the detected fire
            cv2.rectangle(result.orig_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Define text and position (above the rectangle)
            label = "Fire"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = int(x1)  # Align the text horizontally with the box
            text_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 10  # Place text above the box or below if too high

            # Add the label "Fire" above the rectangle
            cv2.putText(result.orig_img, label, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

        # Display the frame with bounding boxes and labels
        cv2.imshow("Detections", result.orig_img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close windows
video.release()
cv2.destroyAllWindows()
