import cv2
import numpy as np
import tkinter as tk

# Function to resize image while maintaining aspect ratio
def resize_image(img):
    img =  cv2.imread(img)

    # Get the screen dimensions
    root = tk.Tk()
    max_width = root.winfo_screenwidth() * 2/3
    max_height = root.winfo_screenheight() * 2/3
    root.destroy()

    # Get the dimensions of the input image
    height, width = img.shape[:2]

    # Check if resizing is necessary
    if width > max_width or height > max_height:
        # Calculate aspect ratio
        aspect_ratio = width / height

        # Resize based on the maximum dimension
        if width > height:
            new_width = int(max_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = int(max_height)
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        resized_img = cv2.resize(img, (new_width, new_height))
    else:
        resized_img = img

    return resized_img

def find_location(img1, img2):
    reference_img = resize_image(img1)
    distorted_img = resize_image(img2)

    # Convert images to grayscale
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    distorted_gray = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors for both images
    kp1, des1 = orb.detectAndCompute(reference_gray, None)
    kp2, des2 = orb.detectAndCompute(distorted_gray, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches
    matched_img = cv2.drawMatches(reference_img, kp1, distorted_img, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Apply homography to corners of the distorted image
    h, w = reference_gray.shape
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    # Draw the bounding box of the distorted image on the reference image
    reference_with_bbox = cv2.polylines(reference_img, [np.int32(transformed_corners)], True, (0, 255, 0), 2)

    # Display the matched keypoints and the bounding box
    cv2.imshow('Matches', matched_img)
    # cv2.imshow('Bounding Box', reference_with_bbox)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# img1 = 'data/images/external_climbing_empty.png'
# img2 = 'data/images/eyetracker_climbing_1.png'

# img1 = 'data/images/drive-download-20240313T101342Z-001/20240313_120757.jpg'
# img2 = 'data/images/drive-download-20240313T101342Z-001/20240313_120801.jpg'

# img1 = 'data/images/drive-download-20240313T101342Z-001/20240313_120808.jpg'
# img2 = 'data/images/drive-download-20240313T101342Z-001/20240313_120811.jpg'

# img1 = 'data/images/drive-download-20240313T101342Z-001/20240313_120815.jpg'
# img2 = 'data/images/drive-download-20240313T101342Z-001/20240313_120819.jpg'

img1 = 'data/images/homography_test_sequential_frames/world0_frame_104.jpg'
img2 = 'data/images/homography_test_sequential_frames/world0_frame_130.jpg'

find_location(img1, img2)
