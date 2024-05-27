import os
from ultralytics import YOLO
import cv2

image = cv2.imread("data/images/external_climbing_empty.png")

model_path = os.path.join("runs", "detect", "train5", "weights", "last.pt")
# runs/detect/train6/weights/last.pt runs/detect/train3/weights/last.pt
model = YOLO(model_path)

threshold = 0.5
results = model(image, save_txt=True)[0]
print("RESULTS")
print(results)
print("BOXES")
print(results.boxes)
print("DATA")
print(results.boxes.data.tolist())
# for result in results.boxes.data.tolist():
#     x1, y1, x2, y2, score, class_id = result

#     if score > threshold:
#         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#         cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
# cv2.imshow('',image)
# cv2.waitKey(1)