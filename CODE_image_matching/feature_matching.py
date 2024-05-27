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

def feature_matching(img1, img2):
    # Read images
    img1 = resize_image(img1)
    img2 = resize_image(img2)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:5], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched keypoints and images
    cv2.imshow("Feature Matches", img_matches)
    # cv2.imshow("Image 1", img1)
    # cv2.imshow("Image 2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return matched keypoints and descriptors
    return kp1, des1, kp2, des2

# img1 = 'data/images/external_climbing_empty.png'
# img2 = 'data/google_colab_masks/stickers/eyetracker_sticker.png'

# img1 = 'data/images/drive-download-20240313T101342Z-001/20240313_120757.jpg'
# img2 = 'data/images/drive-download-20240313T101342Z-001/20240313_120801.jpg'

# img1 = 'data/images/drive-download-20240313T101342Z-001/20240313_120808.jpg'
# img2 = 'data/images/drive-download-20240313T101342Z-001/20240313_120811.jpg'

# img1 = 'data/images/drive-download-20240313T101342Z-001/20240313_120815.jpg'
# img2 = 'data/images/drive-download-20240313T101342Z-001/20240313_120819.jpg'

# Testing feature matching between eyetracker frames. Not working
img1 = 'data/images/feature_matching_data/climbing2_screenshots_eyetracker/background.png'
img2 = 'data/images/feature_matching_data/climbing2_screenshots_eyetracker/closeup1.png'

# testing april tags feature matching. Not working
# img1 = 'data/images/april_tags_test/20240424_165705.jpg'
# img2 = 'data/images/april_tags_test/world0_frame_546.jpg'

kp1, des1, kp2, des2 = feature_matching(img1, img2)