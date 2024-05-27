import cv2
import apriltag
import numpy as np
import csv

def parse_gaze_csv(csv_file):
    """
    Parse the gaze data CSV file and extract norm_pos x and y for each frame.

    Args:
    - csv_file: The path to the CSV file.

    Returns:
    - A list containing tuples of norm_pos x and y for each frame.
    """
    gaze_data = []

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=',')
        current_world_index = None
        current_pos_x = None
        current_pos_y = None

        for row in csv_reader:
            world_index = int(row['world_index'])
            pos_x = float(row['norm_pos_x'])
            pos_y = float(row['norm_pos_y'])

            # Check if it's a new frame (world index changed)
            if world_index != current_world_index:
                # If it's not the first frame, add previous pos x and y to the list
                if current_world_index is not None:
                    gaze_data.append((current_pos_x, current_pos_y))

                # Update current frame data
                current_world_index = world_index
                current_pos_x = pos_x
                current_pos_y = pos_y

    # Add the last frame's pos x and y to the list
    gaze_data.append((current_pos_x, current_pos_y))

    return gaze_data

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

# Load the AprilTag detector
options = apriltag.DetectorOptions(families="tag25h9")
detector = apriltag.Detector(options)

# Load the static image
static_image = cv2.imread('data/images/april_tags_test/20240424_165705.jpg')
static_image = resize_image(static_image, 400, 300)

# Detect tags in the static image
static_image_gray = cv2.cvtColor(static_image, cv2.COLOR_BGR2GRAY)
tags_static = detect_tags(static_image_gray, detector)

# Load the video
video_path = 'data/Eyetracking_data/climbing data 2/000/world.mp4'
video_capture = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create video writer to save the annotated video
output_video_path = 'annotated_video1.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (1962 + static_image.shape[1], max(frame_height, static_image.shape[0])))

# Csv file load and parse
csv_file = "data/Eyetracking_data/001_exported_data/gaze_positions.csv"
gaze_positions = parse_gaze_csv(csv_file)

# Process each frame of the video
ret, frame = video_capture.read()
frame_index = 0  # Track the current frame index
while ret:
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

        # Use gaze position extracted from the CSV file
        gaze_pos_x, gaze_pos_y = gaze_positions[frame_index]

        # Calculate transformation matrix between video frame and static image
        pts1 = np.array(common_tags_frame[0].corners, dtype=float)
        pts2 = np.array(common_tags_static[0].corners, dtype=float)
        transformation_matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        # Map the gaze position relative to the center of the video frame
        gaze_x, gaze_y = map_coordinates((gaze_pos_x, gaze_pos_y), transformation_matrix)

        # Draw a point in the static image at the calculated position
        cv2.circle(static_image_tmp, (int(gaze_x), int(gaze_y)), 5, (255, 0, 0), -1)
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

    # Write the combined image to the output video
    output_video.write(combined_image)

    # Read the next frame
    ret, frame = video_capture.read()
    frame_index += 1


# Release video capture and writer
video_capture.release()
output_video.release()
