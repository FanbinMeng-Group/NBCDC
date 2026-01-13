# from ultralytics import YOLO

# # 安装命令
# # python setup.py develop

# # 数据集示例百度云链接
# # 链接：https://pan.baidu.com/s/19FM7XnKEFC83vpiRdtNA8A?pwd=n93i
# # 提取码：n93i

# if __name__ == '__main__':
#     # 直接使用预训练模型创建模型.
#     # model = YOLO('ultralytics-main/yolov8n.pt')
#     # model.train(**{'cfg': 'ultralytics-main/ultralytics/models/v8/yolov8.yaml', 'data': 'ultralytics-main/ultralytics/datasets/bccd.yaml'})

#     # 使用yaml配置文件来创建模型,并导入预训练权重.
#     model = YOLO('ultralytics/models/v8/yolov8.yaml')
#     model.load('yolov8n.pt')
#     model.train(**{'cfg': 'ultralytics/models/v8/yolov8.yaml', 'data': 'ultralytics/datasets/bccd.yaml'})

#     # 模型验证
#     # model = YOLO('runs/detect/train5/weights/best.pt')
#     # model.val(**{'data': 'dataset/data.yaml','split':'test','iou':0.9,'save_json':True})
#     #
#     # 模型推理
#     # model = YOLO('runs/detect/train5/weights/best.pt')
#     # model.predict(source='dataset/images/test', **{'save': True})

    
    
    
    
    
    
    
    

from ultralytics import YOLO
model = YOLO("YOLOv8_BiFPN_SPD_MultiSEAM_DetectHead_4.yaml")

#model = YOLO("yolov8-attention.yaml")
# model = YOLO('runs/detect/mx/weights/last.pt')



model.train(data="bccd.yaml",seed=1,epochs=100,pretrained=True,batch=48,device='0',iou=0.5,imgsz=320,weight_decay=0.0005,patience=200,workspace=32,optimizer='SGD',project="runs/best_train")



# # model.train(data="VisDrone.yaml",seed=1,epochs=100,batch=8,iou=0.7,device='0,1,2,3',lr0=0.01, lrf=0.0001,momentum=0.937,weight_decay=0.0005,patience=2000,resume=True)resume=True,resume=True,resume=True),,resume="runs/detect/train36/weights/last.pt",resume='runs/detect/mx/weights/last.pt'
# nms=True,,resume='runs/detect/train8/weights/last.pt'
# model.train(data="xView.yaml",seed=1,epochs=10000,batch=2,iou=0.5,device='0',lr0=0.01, lrf=0.0001,momentum=0.937,weight_decay=0.0005,optimizer='AdamW'patience=5000,resume=True)
# model.train(data="xView.yaml",seed=1,epochs=1000,batch=2,iou=0.5,device='0',lr0=0.01, lrf=0.0001,momentum=0.937,weight_decay=0.0005,patience=5000,resume=True)