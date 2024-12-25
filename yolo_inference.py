from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO('models/yolov10.pt')

# Open input video
cap = cv2.VideoCapture('input_videos/video1.avi')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_videos/output_video_predicted.avi', fourcc, 30.0, (frame_width, frame_height))

# Get class name for goalkeeper (assuming 'goalkeeper' is the class label you want to ignore)
goalkeeper_class_name = 'goalkeeper'

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make prediction on the current frame
    results = model(frame)

    # The 'results' is a list, we need to access the first result
    result = results[0]  # Get the first result in the list

    # Draw the predictions on the frame
    for box in result.boxes:  # Access 'boxes' for predictions
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordinates of the bounding box
        conf = box.conf[0]  # Confidence score
        cls = box.cls[0]  # Class ID

        # Only draw bounding boxes with confidence greater than 0.5 and not for goalkeeper
        if conf > 0.6:
            label = model.names[int(cls)]  # Get the label name from the class ID

            # Skip drawing if the class is 'goalkeeper'
            if label != goalkeeper_class_name:
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with annotations to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
