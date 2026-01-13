

from ultralytics import YOLO

# model = YOLO("model.pt")
# model.val()

# model = YOLO("runs/detect/train21/weights/best.pt")
# model = YOLO("runs/best_train/train/weights/best.pt")
model=YOLO("runs/best_train/train/weights/best.pt")
# model.train(data="VisDrone.yaml",seed=1,epochs=150,batch=8,iou=0.7,lr0=0.01, lrf=0.0001,momentum=0.937,weight_decay=0.0005,patience=2000,resume=True)
# model.val(data='VisDrone.yaml')"datasets/sign",batch=8
# from ultralytics import YOLO

# # Load a model
# model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Validate the model
metrics = model.val(data='bccd.yaml',device='0',batch=8,imgsz=320,iou=0.5,project="runs/best_val")  # no arguments needed, dataset and settings remembered
print(metrics)

metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
print("------------------------------")
print(metrics.box.map75)
print("------------------------------")
print(metrics.box.maps)