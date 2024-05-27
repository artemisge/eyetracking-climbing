from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train6/weights/last.pt")  # build a new model from scratch

# Use the model
# model.resume = True
# model.train(epochs=100)
results = model.val()# data="CODE_train_model/config.yaml", epochs=1)  # train the model