import cv2
import apriltag
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
video_path = 'data/videos/climbing2/world0.avi'
video_capture = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create video writer to save the annotated video
output_video_path = 'annotated_video1.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (1962 + static_image.shape[1], max(frame_height, static_image.shape[0])))

# Process each frame of the video
ret, frame = video_capture.read()
while ret:
    # if not ret:
    #     break

    static_image_tmp = static_image.copy()

    # Convert frame to grayscale for AprilTag detection
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tags in the frame
    tags_frame = detect_tags(frame_gray, detector)

    # Find common tags between static image and frame
    common_tags_static, common_tags_frame = find_common_tags(tags_static, tags_frame)

    # Draw bounding boxes around common tags
    annotated_frame = draw_bounding_boxes(frame.copy(), common_tags_frame, (0, 255, 0))
    static_image_tmp = draw_bounding_boxes(static_image_tmp, common_tags_static, (0, 255, 0))

    # Choose the reference height
    reference_height = max(static_image_tmp.shape[0], annotated_frame.shape[0])
    print(static_image_tmp.shape[0], annotated_frame.shape[0])
    print(reference_height)

    # Resize both images to match the reference height
    resized_static_image = resize_to_same_height(static_image_tmp, reference_height)
    resized_annotated_frame = resize_to_same_height(annotated_frame, reference_height)
    print(resized_annotated_frame.shape[1])
    # Concatenate the annotated frame and static image
    combined_image = np.hstack((resized_static_image, resized_annotated_frame))
    # cv2.imshow('Image', combined_image)

    # Wait for a key press for 100 milliseconds (0.1 seconds)
    # cv2.waitKey(10)
    # Write the combined image to the output video
    output_video.write(combined_image)
    # Read the next frame
    ret, frame = video_capture.read()

# Release video capture and writer
video_capture.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
