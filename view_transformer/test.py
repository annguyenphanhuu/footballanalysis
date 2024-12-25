import numpy as np
import cv2


video_path = './input_videos/video.mp4'

pixel_vertices = np.array([[110, 1035], 
                           [265, 275], 
                           [910, 260], 
                           [1640, 915]])


cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Can't open video")
    exit()

ret, frame = cap.read()


if not ret:
    print("Can't read first frame")
    exit()

scale_percent = 70  
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


for i, point in enumerate(pixel_vertices):
    point_resized = (int(point[0] * scale_percent / 100), int(point[1] * scale_percent / 100))
    cv2.circle(frame_resized, point_resized, radius=10, color=(0, 255, 0), thickness=-1)
    cv2.putText(frame_resized, f'Point {i+1}: {point}', 
                (point_resized[0] + 15, point_resized[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Test Points", frame_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

cap.release()
