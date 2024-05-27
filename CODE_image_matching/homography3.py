import cv2
import numpy as np

# Load the original, second, and third images
original_img = cv2.imread('data/images/homography_test_sequential_frames/world0_frame_52.jpg')
second_img = cv2.imread('data/images/homography_test_sequential_frames/world0_frame_78.jpg')
third_img = cv2.imread('data/images/homography_test_sequential_frames/world0_frame_104.jpg')

# Convert images to grayscale
original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
second_gray = cv2.cvtColor(second_img, cv2.COLOR_BGR2GRAY)
third_gray = cv2.cvtColor(third_img, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors for all three images
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(original_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(second_gray, None)
keypoints3, descriptors3 = orb.detectAndCompute(third_gray, None)

# Perform feature matching between the original and second images
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches1 = bf.match(descriptors1, descriptors2)

# Extract matching keypoints
src_points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches1]).reshape(-1, 1, 2)
dst_points1 = np.float32([keypoints2[m.trainIdx].pt for m in matches1]).reshape(-1, 1, 2)

# Compute homography matrix between original and second images
H1, _ = cv2.findHomography(dst_points1, src_points1, cv2.RANSAC)

# Perform feature matching between the second and third images
matches2 = bf.match(descriptors2, descriptors3)

# Extract matching keypoints
src_points2 = np.float32([keypoints2[m.queryIdx].pt for m in matches2]).reshape(-1, 1, 2)
dst_points2 = np.float32([keypoints3[m.trainIdx].pt for m in matches2]).reshape(-1, 1, 2)

# Compute homography matrix between second and third images
H2, _ = cv2.findHomography(dst_points2, src_points2, cv2.RANSAC)

# Combine homography matrices to obtain transformation from original to third image
H_combined = np.dot(H2, H1)

# Warp the third image according to the combined homography matrix
warped_third_img = cv2.warpPerspective(third_img, H_combined, (original_img.shape[1], original_img.shape[0]))

# Blend the first image with lowered opacity onto the warped third image
opacity = 0.5  # Adjust opacity as needed
blended_img = cv2.addWeighted(original_img, opacity, warped_third_img, 1 - opacity, 0)

# Display the result
cv2.imshow('Blended Images', blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()