import os
from ultralytics import YOLO
import cv2

video_path = os.path.join('.', 'data/videos/climbing/trial_1_1/world.mp4')
# trial_number = video_path.split('/')[-2].split('_')[1]  # Extracting trial number from the file path
# video_path_out = os.path.join('data/test_videos/model_testing/', f'trial_{trial_number}_2_eyetracker.mp4')
video_path_out = os.path.join('data/videos/model_testing/new_model_test_on_old_data_video0.avi')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join("runs", "detect", "yolov8n_artemis_climbing2/weights/best.pt")
# runs/detect/yolov8n_artemis_climbing2/weights/best.pt
model = YOLO(model_path)

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()