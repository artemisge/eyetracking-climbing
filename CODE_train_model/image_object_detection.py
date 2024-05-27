from ultralytics import YOLO
model = YOLO("runs/detect/train3/weights/last.pt")
results = model.predict("data/images/person_climbing.png", conf=0.3)
result = results[0]
print("results: " + str(len(result.boxes)))
