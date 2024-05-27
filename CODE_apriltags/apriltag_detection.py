import cv2
import numpy as np
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

# Load the AprilTag detector
options = apriltag.DetectorOptions(families="tag25h9")
at_detector = apriltag.Detector(options)

# Load the two images
image1 = cv2.imread('data/images/april_tags_test/20240424_165705.jpg')
image2 = cv2.imread('data/images/april_tags_test/world0_frame_546.jpg')

# Lower the quality of the first image
compression_params = [cv2.IMWRITE_JPEG_QUALITY, 50]  # Adjust quality (0-100), higher value means better quality
_, compressed_image_buffer = cv2.imencode('.jpg', image1, compression_params)
image1 = cv2.imdecode(compressed_image_buffer, cv2.IMREAD_COLOR)
image1 = resize_image(image1, 800, 600)
# Convert grayscale images to color
# Convert images to grayscale for AprilTag detection
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect AprilTags in the first image
tags1 = at_detector.detect(image1_gray)

# Detect AprilTags in the second image
tags2 = at_detector.detect(image2_gray)

# Display detected tags in the first image
for tag in tags1:
    for idx in range(len(tag.corners)):
        cv2.line(image1, tuple(tag.corners[idx-1].astype(int)), tuple(tag.corners[idx].astype(int)), (255, 255, 0), 2)
    cv2.putText(image1, str(tag.tag_id), org=(tag.corners[0][0].astype(int), tag.corners[0][1].astype(int)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 0, 255), thickness=2)

# Display detected tags in the second image
for tag in tags2:
    for idx in range(len(tag.corners)):
        cv2.line(image2, tuple(tag.corners[idx-1].astype(int)), tuple(tag.corners[idx].astype(int)), (255, 0, 0), 2)
    cv2.putText(image2, str(tag.tag_id), org=(tag.corners[0][0].astype(int), tag.corners[0][1].astype(int)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)

# # Display the images with detected tags
cv2.imshow('Image 1 with AprilTags', image1)
cv2.imshow('Image 2 with AprilTags', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Now, you can proceed with matching the detected tags and mapping them between the two images.
# You may use feature matching algorithms or other techniques to establish correspondences and map the tags.
# Additionally, camera pose estimation and bundle adjustment can be performed to refine the mapping.
