import cv2
import apriltag

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
                print(tag1.tag_id)
                common_tags1.append(tag1)
                common_tags2.append(tag2)
                break

    return common_tags1, common_tags2

# Load the AprilTag detector
options = apriltag.DetectorOptions(families="tag25h9")
detector = apriltag.Detector(options)

# Load the two images
image1 = cv2.imread('data/images/april_tags_test/20240424_165705.jpg')
image2 = cv2.imread('data/images/april_tags_test/world0_frame_312.jpg')

# Lower the quality of the first image and resize it
compression_params = [cv2.IMWRITE_JPEG_QUALITY, 50]  # Adjust quality (0-100), higher value means better quality
_, compressed_image_buffer = cv2.imencode('.jpg', image1, compression_params)
image1 = cv2.imdecode(compressed_image_buffer, cv2.IMREAD_COLOR)
image1 = resize_image(image1, 400, 300)

# Convert images to grayscale for AprilTag detection
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect tags in both images
tags_image1 = detect_tags(image1_gray, detector)
tags_image2 = detect_tags(image2_gray, detector)

# Find common tags in both images
common_tags1, common_tags2 = find_common_tags(tags_image1, tags_image2)

# Draw bounding boxes around common detected tags with the same color in both images
color = (0, 255, 0)  # Green color for bounding boxes
image1_with_boxes = draw_bounding_boxes(image1.copy(), common_tags1, color)
image2_with_boxes = draw_bounding_boxes(image2.copy(), common_tags2, color)

# Display the images with bounding boxes
cv2.imshow('Image 1 with Bounding Boxes', image1_with_boxes)
cv2.imshow('Image 2 with Bounding Boxes', image2_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
