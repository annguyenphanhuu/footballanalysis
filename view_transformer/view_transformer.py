import numpy as np 
import cv2
from screeninfo import get_monitors

class ViewTransformer():
    def __init__(self, frame):
        self.frame = frame  # The frame to be used for interaction
        
        try:
            self.court_width = float(input("Enter the court width (in meters): "))
            self.court_length = float(input("Enter the court length (in meters): "))
        except ValueError:
            print("Invalid input! Using default dimensions.")
            self.court_width = 68
            self.court_length = 23.32

        self.pixel_vertices = []
        self.target_vertices = np.array([
            [0, self.court_width],
            [0, 0],
            [self.court_length, 0],
            [self.court_length, self.court_width]
        ])

        self.pixel_vertices = np.array(self.pixel_vertices, dtype=np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = None

        # Get the screen resolution
        monitor = get_monitors()[0]  # Assuming the first monitor is the primary
        screen_width = monitor.width
        screen_height = monitor.height

        # Resize the image to fit the screen while maintaining aspect ratio
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        if screen_width / screen_height > aspect_ratio:
            new_width = int(screen_height * aspect_ratio)
            new_height = screen_height
        else:
            new_width = screen_width
            new_height = int(screen_width / aspect_ratio)

        self.resized_frame = cv2.resize(self.frame, (new_width, new_height))

        # Scale window to 70% of screen dimensions
        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.7)

        # Mouse callback to capture 4 points
        self.selected_points = []
        cv2.namedWindow("Select Points", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Points", window_width, window_height)  # Set window size
        cv2.setMouseCallback("Select Points", self.mouse_callback)
        self.show_frame_for_selection()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selected_points) < 4:
                self.selected_points.append((x, y))
                cv2.circle(self.resized_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Select Points", self.resized_frame)

    def show_frame_for_selection(self):
        # Show the resized frame to select points
        while len(self.selected_points) < 4:
            cv2.imshow("Select Points", self.resized_frame)
            cv2.waitKey(1)
        cv2.destroyAllWindows()

        # After selecting 4 points, store the coordinates
        self.pixel_vertices = np.array(self.selected_points, dtype=np.float32)

        # Print the coordinates of the selected points
        print("Selected points:")
        for i, point in enumerate(self.selected_points):
            print(f"Point {i + 1}: {point}")

        # Compute perspective transform
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    # Other methods remain unchanged


    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
