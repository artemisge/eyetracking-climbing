import os
import cv2
import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
import cv2
import apriltag
import numpy as np
import csv

# this code is supposed to detect april tags, take the gaze coordinates and make them into frame coordinates and then into global coordinates

# This part is for detecting april tags:

def detect_tags(image, detector):
    # Detect AprilTags in the image
    tags = detector.detect(image)

    return tags

def draw_bounding_boxes(image, tags, color):
    # Draw bounding boxes around detected tags
    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(image, tuple(tag.corners[idx-1].astype(int)), tuple(tag.corners[idx].astype(int)), color, 2)

    return image

def find_common_tags(tags1, tags2):
    # Find common tags detected in both images
    common_tags1 = []
    common_tags2 = []
    for tag1 in tags1:
        for tag2 in tags2:
            if tag1.tag_id == tag2.tag_id:
                common_tags1.append(tag1)
                common_tags2.append(tag2)
                break

    return common_tags1, common_tags2



def normalize_gaze(gaze_points, frame_shape):
    # Normalize gaze points to match frame dimensions
    H, W = frame_shape[:-1]
    X, Y = gaze_points[:, 0] * W, gaze_points[:, 1] * H
    Y = H - Y  # Invert Y-axis to match OpenCV's coordinate system
    return np.column_stack((X, Y)).astype(int)

def resize_to_same_height(image, reference_height):
    """
    Resize an image to match a reference height while maintaining its aspect ratio.

    Args:
    - image: The image to be resized.
    - reference_height: The target height to match.

    Returns:
    - The resized image.
    """
    # Calculate aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]

    # Calculate new width based on the aspect ratio and reference height
    new_width = int(reference_height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, reference_height))

    return resized_image

def map_coordinates(coord, transformation_matrix):
    coord = np.array([[coord[0], coord[1], 1]], dtype=float)
    mapped_coord = np.dot(transformation_matrix, coord.T)
    mapped_coord = mapped_coord / mapped_coord[2]
    return mapped_coord[0][0], mapped_coord[1][0]


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

def main():
    # Load the AprilTag detector
    options = apriltag.DetectorOptions(families="tag25h9")
    detector = apriltag.Detector(options)

    # Load the static image
    static_image = cv2.imread('data/images/april_tags_test/20240424_165705.jpg')
    static_image = resize_image(static_image, 400, 300)

    # Detect tags in the static image
    static_image_gray = cv2.cvtColor(static_image, cv2.COLOR_BGR2GRAY)
    tags_static = detect_tags(static_image_gray, detector)

    # Load gaze data
    csv_file = 'data/Eyetracking_data/001_exported_data/gaze_positions.csv'
    gaze = pd.read_csv(csv_file)

    # Folder containing frames
    frames_folder = 'data/Eyetracking_data/extracted_frames_from_recording/'

    # Output video path
    output_video_path = 'pupil_labs_w_april_tags_video_vis1.avi'
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png')])

    # Video properties
    frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    frame_height, frame_width, _ = frame.shape
    fps = 30  # Adjust as needed

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * 2, frame_height))

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

        static_image_tmp = static_image.copy()

        # Convert frame to grayscale for AprilTag detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect tags in the frame
        tags_frame = detect_tags(frame_gray, detector)

        # Find common tags between static image and frame
        common_tags_static, common_tags_frame = find_common_tags(tags_static, tags_frame)

        # Draw bounding boxes around common tags if found
        if common_tags_static and common_tags_frame:
            annotated_frame = draw_bounding_boxes(frame.copy(), common_tags_frame, (0, 255, 0))
            static_image_tmp = draw_bounding_boxes(static_image_tmp, common_tags_static, (0, 255, 0))

            # Calculate transformation matrix between video frame and static image
            pts1 = np.array(common_tags_frame[0].corners, dtype=float)
            pts2 = np.array(common_tags_static[0].corners, dtype=float)
            transformation_matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

            # draw green point in static image
            for x, y in gaze_points:
                # Map the gaze position relative to the center of the video frame
                mapped_x, mapped_y = map_coordinates((x, y), transformation_matrix)

                # Draw green dot
                cv2.circle(static_image_tmp, (int(mapped_x), int(mapped_y)), 10, (0, 255, 0), -1)

            # Draw gaze movement line
            if len(gaze_points) > 1:
                for i in range(1, len(gaze_points)):
                    start_point = tuple(int(coord) for coord in map_coordinates(tuple(gaze_points[i-1]), transformation_matrix))
                    end_point = tuple(int(coord) for coord in map_coordinates(tuple(gaze_points[i]), transformation_matrix))

                    cv2.line(static_image_tmp, start_point, end_point, (0, 0, 255), 3)

        else:
            # If no common tags found, draw a message indicating it
            cv2.putText(static_image_tmp, "No tags", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Choose the reference height
        reference_height = max(static_image_tmp.shape[0], frame_height)

        # Resize both the static image and frame to match the reference height
        resized_static_image = resize_to_same_height(static_image_tmp, reference_height)
        resized_frame = resize_to_same_height(frame, reference_height)

        # Concatenate the annotated frame and static image
        combined_image = np.hstack((resized_static_image, resized_frame))

        # Display the combined image
        cv2.imshow('Final Frame', combined_image)
        cv2.waitKey(1)  # Wait indefinitely for a key press

        # Write frame to video
        output_video.write(combined_image)

    # Release resources
    output_video.release()
    cv2.destroyAllWindows()

main()