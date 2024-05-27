import os
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def normalize_gaze(gaze_points, frame_shape):
    # Normalize gaze points to match frame dimensions
    H, W = frame_shape[:-1]
    X, Y = gaze_points[:, 0] * W, gaze_points[:, 1] * H
    Y = H - Y  # Invert Y-axis to match OpenCV's coordinate system
    return np.column_stack((X, Y)).astype(int)

# Load gaze data
csv_file = 'data/Eyetracking_data/001_exported_data/gaze_positions.csv'
gaze = pd.read_csv(csv_file)

# Folder containing frames
frames_folder = 'data/Eyetracking_data/extracted_frames_from_recording/'

# Output video path
output_video_path = 'pupil_labs_video_vis1.avi'
frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png')])

# Video properties
frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
frame_height, frame_width, _ = frame.shape
fps = 30  # Adjust as needed

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Iterate over frames
for frame_file in frame_files:
    # Read frame
    frame_path = os.path.join(frames_folder, frame_file)
    frame = cv2.imread(frame_path)

    # Extract frame index from filename
    frame_index = int(os.path.splitext(frame_file)[0].split('frame')[1])

    # Get gaze points for the current frame
    gaze_points = gaze[gaze["world_index"] == frame_index][["norm_pos_x", "norm_pos_y"]].values

    # Normalize gaze points
    gaze_points = normalize_gaze(gaze_points, frame.shape)

     # Draw gaze points on the frame
    for x, y in gaze_points:
        # Draw green dot
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

    # Draw gaze movement line
    if len(gaze_points) > 1:
        for i in range(1, len(gaze_points)):
            cv2.line(frame, tuple(gaze_points[i-1]), tuple(gaze_points[i]), (0, 0, 255), 3)

    # Write frame to video
    output_video.write(frame)

# Release resources
output_video.release()
cv2.destroyAllWindows()
