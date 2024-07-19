from ultralytics import YOLOv10
import cv2
import subprocess as sp
import numpy as np
import ffmpeg



model = YOLOv10("runs/detect/train_v10s.fireHazard/weights/best.pt")
source_path="D:/python_project/mydata/lux/huizhan.mp4"



rtsp_url = 'rtsp://admin:hk123456@192.168.168.90:554/h265/ch14/subtype/av_stream'
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Could not open RTSP stream.")
    exit(-1)

#=========================推========================

# RTSP推流地址
rtsp_output_url = 'rtsp://172.16.80.36:8554/video'

# 创建一个空的Input对象作为输入源，使用管道接收数据
input_pipe = Input(pipe='pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(frame_width, frame_height))

# 创建RTSP输出对象
output = Output(rtsp_output_url, vcodec='libx264', pix_fmt='yuv420p', r=fps)

# 启动ffmpeg进程，连接输入管道和输出RTSP地址
ffmpeg_process = (
    input_pipe
    .global_args('-hide_banner')
    .global_args('-loglevel', 'warning')
    .output(output)
    .run_async(pipe_stdin=True)
)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    faces = model.predict(
        source=frame,
        conf=0.15,
        iou=0.70,
        imgsz=640,
        half=True,
        # Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.
        # device=1,       #cuda:0
        vid_stride=1,  # 视频流跳帧
        stream=False,  # 视频以流式输入，减少缓存占用;;;;;用从cv2读取时不需要配置
        stream_buffer=False,  # 视频流实时缓冲所有帧,用于实时视频
        visualize=False,  # 检测过程各层可视化
        agnostic_nms=False,  # 类无关nms,合成不同类别的框，在检测一个目标分属多类时有用
        retina_masks=False,  # 使用高分辨率分割掩码（如果模型中可用）。这可以提高分割任务的掩码质量，提供更精细的细节。
        show=True,
        save=False,
        show_conf=False,
        show_labels=False,
        save_frames=False,
    )

    person_boxes = [(x, y, x + w, y + h) for (x, y, w, h) in faces]
    for box in person_boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # 将处理后的帧送入ffmpeg进程
    ffmpeg_process.stdin.write(frame.tobytes())

    # 显示带有框的帧（可选，仅用于本地调试）
    cv2.imshow('RTSP Video Stream with Person Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

