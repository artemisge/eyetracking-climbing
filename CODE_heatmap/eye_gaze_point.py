import csv
import cv2
import numpy as np

def resize_image(image, max_width, max_height):
    """
    Resize an image while maintaining its aspect ratio.

    Args:
    - image: The image to be resized.
    - max_width: The maximum width for the resized image.
    - max_height: The maximum height for the resized image.

    Returns:
    - The resized image.
    """
    # Calculate aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]

    # Calculate new dimensions based on the aspect ratio
    new_width = max_width
    new_height = int(new_width / aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

# Load the video
video_path = 'data/Eyetracking_data/climbing data 2/000/world.mp4'
video_capture = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create video writer to save the annotated video
output_video_path = 'annotated_video_first_point_per_frame.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Load gaze data from CSV file
csv_file = 'data/Eyetracking_data/001_exported_data/gaze_positions.csv'

# Process each frame of the video
ret, frame = video_capture.read()
current_frame_index = -1  # Initialize the current frame index
with open(csv_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        world_index = int(row['world_index'])

        # Check if this is the first point for the current frame
        if world_index != current_frame_index:
            # Update the current frame index
            current_frame_index = world_index

            # Read gaze point from CSV
            gaze_x = int(float(row['norm_pos_x']) * frame_width)
            gaze_y = int(float(row['norm_pos_y']) * frame_height)

            # Draw circle for gaze point on the frame
            cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)

            # Write the frame with the circle to the output video
            output_video.write(frame)

            # Read the next frame
            ret, frame = video_capture.read()

        else:
            # Skip this point as it's not the first point for the current frame
            continue

# Release video capture and writer
video_capture.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
