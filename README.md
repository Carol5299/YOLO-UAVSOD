Training: yolo task=detect mode=train model=YOLO-UAVSOD.yaml data=ultralytics/datasets/VisDrone.yaml epochs=300 batch=8 imgsz=640 device=1

Val:  yolo task=detect mode=val model=yolov8s.pt data=ultralytics/datasets/VisDrone.yaml task=test batch=1 imgsz=640 device=0
