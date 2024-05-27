# from ultralytics import SAM
import os
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# Load SAM
#model = SAM('sam_b.pt')

# Load YOLO costum trained model
model_path = os.path.join("runs", "detect", "train5", "weights", "last.pt")
model = YOLO(model_path)
threshold = 0.8

# Load Image
image_path = 'data/images/eyetracker_climbing_1.png'
image = cv2.imread(image_path)

# YOLO prediction
yolo_results = model.predict(image, verbose=False)

for result in yolo_results:
    for box in result.boxes:
        left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=np.int64).squeeze()
        width = right - left
        height = bottom - top
        center = (left + int((right-left)/2), top + int((bottom-top)/2))
        label = yolo_results[0].names[int(box.cls)]
        confidence = float(box.conf.cpu())

        cv2.rectangle(image, (left, top),(right, bottom), (255, 0, 0), 2)
        
# cv2.imshow('YOLO bounding boxes', image)
# cv2.waitKey(0)

# load the SAM model
sam_checkpoint = "data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.96,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

	
masks = mask_generator_.generate(image)
print(len(masks))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
# ax[0].imshow(orgional_image)
# ax[0].set_title('Original Image')
# ax[0].axis('off')
ax[0].imshow(image)
show_anns(masks)
ax[0].set_title('Generated Masks')
ax[0].axis('off')
plt.show()

cv2.destroyAllWindows()