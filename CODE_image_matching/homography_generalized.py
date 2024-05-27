import cv2
import numpy as np

def warp_images(images):
    # Convert the first image to grayscale
    original_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    
    # Initialize a list to store the combined homography matrices
    combined_H_matrices = [np.eye(3)]
    
    # Detect keypoints and compute descriptors for all images
    orb = cv2.ORB_create()
    keypoints = [orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None) for img in images]
    
    # Perform feature matching and compute homography matrices between consecutive images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for i in range(len(images) - 1):
        matches = bf.match(keypoints[i][1], keypoints[i+1][1])
        src_points = np.float32([keypoints[i][0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints[i+1][0][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(dst_points, src_points, cv2.RANSAC)
        combined_H_matrices.append(np.dot(H, combined_H_matrices[-1]))
    
    # Warp the last image according to the combined homography matrix
    warped_img = cv2.warpPerspective(images[-1], combined_H_matrices[-1], (images[0].shape[1], images[0].shape[0]))
    
    return warped_img

# Load the images
image_paths = ['data/images/homography_test_sequential_frames/world0_frame_26.jpg', 'data/images/homography_test_sequential_frames/world0_frame_52.jpg', 'data/images/homography_test_sequential_frames/world0_frame_78.jpg', 'data/images/homography_test_sequential_frames/world0_frame_104.jpg']
images = [cv2.imread(path) for path in image_paths]

# Warp the last image
warped_last_img = warp_images(images)

# Overlay the last warped image onto the first image with lowered opacity
opacity = 0.5  # Adjust opacity as needed
blended_img = cv2.addWeighted(images[0], opacity, warped_last_img, 1 - opacity, 0)

# Display the result
cv2.imshow('Last Warped Image on Pale Background', blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

