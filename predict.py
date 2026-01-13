
from ultralytics import YOLO

model = YOLO("runs/best_train/train16/weights/best.pt")
model.predict(source='datasets/bccd/images/test',iou=0.5, **{'save': True})


# # model.train(data="VisDrone.yaml",seed=1,epochs=100,batch=8,iou=0.7,device='0,1,2,3',lr0=0.01, lrf=0.0001,momentum=0.937,weight_decay=0.0005,patience=2000,resume=True)resume=True,resume=True,resume=True),,resume="runs/detect/train36/weights/last.pt",resume='runs/detect/mx/weights/last.pt'
# nms=True,,resume='runs/detect/train8/weights/last.pt'
# model.train(data="xView.yaml",seed=1,epochs=10000,batch=2,iou=0.5,device='0',lr0=0.01, lrf=0.0001,momentum=0.937,weight_decay=0.0005,optimizer='AdamW'patience=5000,resume=True)
# model.train(data="xView.yaml",seed=1,epochs=1000,batch=2,iou=0.5,device='0',lr0=0.01, lrf=0.0001,momentum=0.937,weight_decay=0.0005,patience=5000,resume=True)