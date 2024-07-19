from ultralytics import YOLOv10
import os
from datetime import datetime
import time
import cv2
import threading
import time

model = YOLOv10("runs/detect/train_v10s.fireHazard/weights/best.pt")
#model = YOLOv10("runs/detect/train_v10s.exhibition-v4/weights/best.pt")#来自exhibiiton
#model = YOLOv10("weights/yolov10n.pt")

# source_path="E:/data-23robotConference/23/F7/"
#source_path="rtsp://admin:hk123456@192.168.168.90:554/h265/ch14/subtype/av_stream"
#source_path="D:/python_project/yolov10/ultralytics/assets/bus.jpg"
source_path="D:/python_project/mydata/lux/huizhan.mp4"

start_time = time.time()
results = model.predict(
    source=source_path,
    conf=0.15,
    iou=0.70,
    imgsz=640,
    half=True,     #Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.
    #device=1,       #cuda:0
    vid_stride=1,   #视频流跳帧
    stream=False,   #视频以流式输入，减少缓存占用;;;;;用从cv2读取时不需要配置
    stream_buffer=False,#视频流实时缓冲所有帧,用于实时视频
    visualize=False,     #检测过程各层可视化
    agnostic_nms=False,       #类无关nms,合成不同类别的框，在检测一个目标分属多类时有用
    retina_masks = False,       #使用高分辨率分割掩码（如果模型中可用）。这可以提高分割任务的掩码质量，提供更精细的细节。
    show=True,
    save=False,
    show_conf=False,
    show_labels=False,
    save_frames=False,
)
end_time = time.time()
execution_time = end_time - start_time
print("执行时间：", execution_time, "秒")

