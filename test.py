from ultralytics import YOLOv10

if __name__ == '__main__':
    # Load a model
    model = YOLOv10(r'./ultralytics/cfg/models/v10/yolov10n.yaml')  # build a new model from YAML
    model.info()