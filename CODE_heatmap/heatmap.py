import os
import cv2
import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
import cv2
import apriltag
import numpy as np
import csv
from scipy.ndimage import gaussian_filter


# copy from the pl with april tags no2. I need to find a way to code the mapped gaze points to the static image


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



def normalize_gaze(gaze_points, frame_shape):
    # Normalize gaze points to match frame dimensions
    H, W = frame_shape[:-1]
    X, Y = gaze_points[:, 0] * W, gaze_points[:, 1] * H
    Y = H - Y  # Invert Y-axis to match OpenCV's coordinate system
    return np.column_stack((X, Y)).astype(int)

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

def generate_heatmap(size_x, size_y, points, sd_gauss):
    heatmap = np.zeros((size_y, size_x), dtype=np.float32)
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < size_x and 0 <= y < size_y:
            heatmap[y, x] += 1
    heatmap = gaussian_filter(heatmap, sd_gauss)
    heatmap = np.clip(heatmap / heatmap.max(), 0, 1)  # Normalize heatmap
    heatmap = (heatmap * 255).astype(np.uint8)  # Convert to 8-bit for OpenCV
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.6):
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def main():
    options = apriltag.DetectorOptions(families="tag25h9")
    detector = apriltag.Detector(options)

    static_image = cv2.imread('data/images/april_tags_test/20240424_165705.jpg')
    static_image = resize_image(static_image, 400, 300)

    static_image_gray = cv2.cvtColor(static_image, cv2.COLOR_BGR2GRAY)
    tags_static = detect_tags(static_image_gray, detector)

    csv_file = 'data/Eyetracking_data/001_exported_data/gaze_positions.csv'
    gaze = pd.read_csv(csv_file)

    frames_folder = 'data/Eyetracking_data/extracted_frames_from_recording/'
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png')])

    frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    frame_height, frame_width, _ = frame.shape

    all_mapped_points = []

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        frame_index = int(os.path.splitext(frame_file)[0].split('frame')[1])
        gaze_points = gaze[gaze["world_index"] == frame_index][["norm_pos_x", "norm_pos_y"]].values
        gaze_points = normalize_gaze(gaze_points, frame.shape)
        static_image_tmp = static_image.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags_frame = detect_tags(frame_gray, detector)
        common_tags_static, common_tags_frame = find_common_tags(tags_static, tags_frame)
        if common_tags_static and common_tags_frame:
            pts1 = np.array(common_tags_frame[0].corners, dtype=float)
            pts2 = np.array(common_tags_static[0].corners, dtype=float)
            transformation_matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            for x, y in gaze_points:
                mapped_x, mapped_y = map_coordinates((x, y), transformation_matrix)
                all_mapped_points.append((mapped_x, mapped_y))

    static_height, static_width, _ = static_image.shape
    sd_gauss = 15
    heatmap = generate_heatmap(static_width, static_height, all_mapped_points, sd_gauss)
    combined_image = overlay_heatmap(static_image, heatmap)
    
    cv2.imwrite('data/heatmap_data/heatmap.png', heatmap)
    cv2.imwrite('data/heatmap_data/combined_image.png', combined_image)

    cv2.imshow('Heatmap', heatmap)
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()