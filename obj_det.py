# from ultralytics import YOLO
# import cv2

# # group of neural network models trained with PyTorch-pt
# model = YOLO("yolov8m.pt")
# results = model.predict("Screenshot_20240206_151245.png")
# result = results[0]
# len(result.boxes)
# box = result.boxes[0]

# for box in result.boxes:
#   class_id = result.names[box.cls[0].item()]
#   cords = box.xyxy[0].tolist()
#   cords = [round(x) for x in cords]
#   conf = round(box.conf[0].item(), 2)
#   print("Object type:", class_id)
#   print("Coordinates:", cords)
#   print("Probability:", conf)
#   print("---")


from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import os

model = YOLO('yolov8n.pt')

# Replace 'phone_v1_1.avi' with the path to your video file
video_path = 'phone_v1_1.avi'

# Open the video file
cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, img = cap.read()

    if not ret:
        # If no frame is read, break out of the loop
        break
    
    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results = model.predict(img)

    for r in results:
        annotator = Annotator(img)
        boxes = r.boxes

        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    img = annotator.result()
    cv2.imshow('YOLO V8 Detection', img)
    
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()