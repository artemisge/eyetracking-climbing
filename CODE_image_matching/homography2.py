import cv2
import numpy as np

# Load the original and close-up images
img1 = 'data/images/homography_test_sequential_frames/world0_frame_78.jpg'
img2 = 'data/images/homography_test_sequential_frames/world0_frame_104.jpg'
original_img = cv2.imread(img1)
close_up_img = cv2.imread(img2)

# Convert images to grayscale
original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
close_up_gray = cv2.cvtColor(close_up_img, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors for both images
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(original_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(close_up_gray, None)

# Perform feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Extract matching keypoints
src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute homography matrix
H, _ = cv2.findHomography(dst_points, src_points, cv2.RANSAC)

# Warp the close-up image to align with the original image
warped_img = cv2.warpPerspective(close_up_img, H, (original_img.shape[1], original_img.shape[0]))

# Display the result
cv2.imshow('Warped Image', warped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()