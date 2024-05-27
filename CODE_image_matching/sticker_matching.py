import glob
import os
import cv2

path_stickers_folder = 'data/google_colab_masks/stickers/external'
path_main_hold = 'data/google_colab_masks/stickers/eyetracker_sticker4.png'

# Load the main template image in color
main_hold = cv2.imread(path_main_hold)

# Initialize a list to store matching scores
matching_scores = []

# Iterate over each test image (sticker) in the folder
for i, sticker_path in enumerate(sorted(glob.glob(os.path.join(path_stickers_folder, '*.png')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))):
    # Load the test image (sticker) in color
    sticker = cv2.imread(sticker_path)
    
    # Calculate scaling factors for resizing while preserving aspect ratio
    template_aspect_ratio = main_hold.shape[1] / main_hold.shape[0]
    sticker_aspect_ratio = sticker.shape[1] / sticker.shape[0]

    if template_aspect_ratio > sticker_aspect_ratio:
        # Resize based on width
        new_width = sticker.shape[1]
        new_height = int(new_width / template_aspect_ratio)
    else:
        # Resize based on height
        new_height = sticker.shape[0]
        new_width = int(new_height * template_aspect_ratio)

    # Resize the template image while preserving aspect ratio
    template_resized = cv2.resize(main_hold, (new_width, new_height))
    
    # Perform template matching using color-based similarity measures
    result = cv2.matchTemplate(sticker, template_resized, method=cv2.TM_CCOEFF_NORMED)
    
    # Extract the maximum matching value
    max_val = cv2.minMaxLoc(result)[1]
    
    # Append image number and matching score to the list
    matching_scores.append((i + 1, max_val))

for i, matching_score in enumerate(matching_scores):
    print(str(i+1) + " " + str(matching_score))
    
# Find the image with the highest matching score
best_match = max(matching_scores, key=lambda x: x[1])

print(f"The best match is image {best_match[0]} with a matching score of {best_match[1]}")
