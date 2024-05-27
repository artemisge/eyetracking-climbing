# use original image and masks to extract "stickers"
# both from external camera view and eye tracker

import glob
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

def extract_save_sticker(img, mask, output_path):
    res = cv2.bitwise_and(img, img, mask=mask)
    grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, grey = cv2.threshold(grey, 0, 55, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(grey)
    res = res[y:y+h, x:x+w]
    cv2.imwrite(output_path, res)

# path_external = 'data/images/external_climbing_empty.png'
path_eyetracker = 'data/feature_matching_data/trial1_1_eyetracker_4.png'
path_mask_eyetracker = 'data/google_colab_masks/masks/v2/4.png'
# path_folder_mask_external = 'data/google_colab_masks/masks/external'

path_stickers = 'data/google_colab_masks/stickers/external'

# eye tracker
img = cv2.imread(path_eyetracker)
mask = cv2.imread(path_mask_eyetracker,0)
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

res = cv2.bitwise_and(img,img,mask = mask)

grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
_, grey = cv2.threshold(grey, 0, 55, cv2.THRESH_BINARY)

x,y,w,h = cv2.boundingRect(grey)
res = res[y:y+h, x:x+w]

cv2.imwrite("data/google_colab_masks/stickers/eyetracker_sticker4.png", res)

# external
# Load external image (outside loop)
# img_external = cv2.imread(path_external)

# # Process each mask from the external folder with the same external image
# for mask_path in glob.glob(os.path.join(path_folder_mask_external, '*.png')):
#     # Load mask
#     mask = cv2.imread(mask_path, 0)
    
#     # Generate output sticker path
#     sticker_num = os.path.basename(mask_path).split('_')[-1].split('.')[0]
#     output_path = os.path.join(path_stickers, f'sticker_image_{sticker_num}.png')
    
#     # Extract and save sticker
#     extract_save_sticker(img_external, mask, output_path)

cv2.destroyAllWindows()