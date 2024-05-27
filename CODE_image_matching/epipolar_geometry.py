import numpy as np
import cv2
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

def main1(img1, img2):
    # Load images
    img1_color = resize_image(img1)
    img2_color = resize_image(img2)
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors between images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Get corresponding points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # Select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Compute epilines corresponding to points in the second image and draw them on the first image
    lines1 = cv2.computeCorrespondEpilines(pts2, 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_lines = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0]*img1.shape[1])/r[1]])
        img1_lines = cv2.line(img1_lines, (x0, y0), (x1, y1), color, 1)
        img1_lines = cv2.circle(img1_lines, tuple(map(int, pt1.ravel())), 5, color, -1)

    # Compute epilines corresponding to points in the first image and draw them on the second image
    lines2 = cv2.computeCorrespondEpilines(pts1, 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_lines = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)
    for r, pt1, pt2 in zip(lines2, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0]*img2.shape[1])/r[1]])
        img2_lines = cv2.line(img2_lines, (x0, y0), (x1, y1), color, 1)
        img2_lines = cv2.circle(img2_lines, tuple(map(int, pt2.ravel())), 5, color, -1)

    # Draw rectangle around the region of the smaller image in the larger image
    h, w = img2.shape[:2]
    rect_top_left = (int(w/2 - w/2), int(h/2 - h/2))
    rect_bottom_right = (int(w/2 + w/2), int(h/2 + h/2))
    img1_lines = cv2.rectangle(img1_lines, rect_top_left, rect_bottom_right, (0, 255, 0), 2)

    # Display images with epipolar lines
    cv2.imshow('Image 1 with Epipolar Lines', img1_lines)
    cv2.imshow('Image 2 with Epipolar Lines', img2_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # img1 = 'data/images/external_climbing_empty.png'
    # img2 = 'data/images/eyetracker_climbing_1.png'

    # img1 = 'data/images/drive-download-20240313T101342Z-001/20240313_120757.jpg'
    # img2 = 'data/images/drive-download-20240313T101342Z-001/20240313_120801.jpg'

    # img1 = 'data/images/drive-download-20240313T101342Z-001/20240313_120808.jpg'
    # img2 = 'data/images/drive-download-20240313T101342Z-001/20240313_120811.jpg'

    img1 = 'data/images/drive-download-20240313T101342Z-001/20240313_120815.jpg'
    img2 = 'data/images/drive-download-20240313T101342Z-001/20240313_120819.jpg'

    main1(img1, img2)
