from ultralytics import YOLOv10
import cv2
import multiprocessing
import time
import queue
import psutil  # 用于系统资源监控
import os
from datetime import datetime

# 日志文件
log_file = "system_log.txt"


# 全局变量
connection_timeout = 2  # 连接超时时间（秒）
read_time = 0.01  # 读取一次的时间（秒）
read_interval = 5  # 每个视频流的读取间隔（秒）
rtsp_list = [
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch1/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch3/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch4/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch5/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch6/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch7/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch8/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch9/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch10/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch11/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch12/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch13/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch14/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch15/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch16/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch17/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch18/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch19/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch20/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch21/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch22/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch23/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch24/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch25/subtype/av_stream',
    'rtsp://admin:hk123456@192.168.168.90:554/h265/ch26/subtype/av_stream',
]

# #一些参数
# #model,conf,iou,vid_stride,upload_stride,classes,saveBbox
#函数
def predict_frame(frame,rtsp_url):
    model = YOLOv10("weights/yolov10n.pt")
    results = model.predict(
        source=frame,
        conf=0.15,
        iou=0.70,
        imgsz=640,
        half=True,# Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.
        device=1,
        classes=[],
        vid_stride=150,  # 视频流跳帧
        agnostic_nms=False,  # 类无关nms,合成不同类别的框，在检测一个目标分属多类时有用
        show=False,
        save=False,
        save_frames=False,
        show_labels=False,
    )
    output_folder = f'D:/python_project/yolov10/runs/outputs/{rtsp_url[46:-18]}/'
    print(f'---------------------------------------------已检测--{rtsp_url[46:-18]}')
    if len(results[0].boxes)>0:
        #print('----------------------------------------------------------------牛的，有框子')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        formatted_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
        print(datetime.now())
        print(formatted_datetime)
        cv2.imwrite(f"{output_folder}{formatted_datetime}.jpg", frame)
        #print(f'---------------------------------------------------------------写了{rtsp_url[46:-18]}')
# ######-------------------------------------------------定义进/xian程--------------------------------
# #电子围栏+消防通道进程
# def process_video_stream_guard(rtsp_url, result_queue, error_queue,freq):
#     cap = None
#     while True:
#         try:
#             #触发器
#             if cap is None or not cap.isOpened():
#                 print(f"Attempting to reconnect to {rtsp_url}...")
#                 cap = cv2.VideoCapture(rtsp_url)
#                 if not cap.isOpened():
#                     error_queue.put((rtsp_url, "Failed to connect to RTSP stream连不上了，立刻整改"))
#                     time.sleep(5)  # 等待5秒后再重试
#                     continue
#             # 主函数
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"Lost connection to {rtsp_url}. Retrying...破流读不了一点儿...")
#                 cap.release()  # 释放旧的VideoCapture对象
#                 cap = None  # 设置为None以触发重连
#                 continue
#             prediction = predict_frame(frame,rtsp_url)
#             # result_queue.put(prediction)#结果入队
#             # queueHead=result_queue.get()
#
#
#         except Exception as e:
#             print(f"Error processing {rtsp_url}: {e}")
#             error_queue.put((rtsp_url, str(e)))
#             cap.release() if cap is not None else None
#             cap = None
#             time.sleep(5)  # 等待5秒后再重试#
# #入口总人数进程
# def process_video_stream_gate(rtsp_url, result_queue, error_queue,freq):
#     cap = None
#     while True:
#         try:
#             #触发器
#             if cap is None or not cap.isOpened():
#                 print(f"Attempting to reconnect to {rtsp_url}...")
#                 cap = cv2.VideoCapture(rtsp_url)
#                 if not cap.isOpened():
#                     error_queue.put((rtsp_url, "Failed to connect to RTSP stream连不上了，立刻整改"))
#                     time.sleep(5)  # 等待5秒后再重试
#                     continue
#             # 主函数
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"Lost connection to {rtsp_url}. Retrying...破流读不了一点儿...")
#                 cap.release()  # 释放旧的VideoCapture对象
#                 cap = None  # 设置为None以触发重连
#                 continue
#             prediction = predict_frame(frame,rtsp_url)
#             # result_queue.put(prediction)#结果入队
#             # queueHead=result_queue.get()
#
#
#         except Exception as e:
#             print(f"Error processing {rtsp_url}: {e}")
#             error_queue.put((rtsp_url, str(e)))
#             cap.release() if cap is not None else None
#             cap = None
#             time.sleep(5)  # 等待5秒后再重试
# #场馆内进程
# def process_video_stream_inVenue(rtsp_url, result_queue, error_queue,freq):
#     count = 0
#     cap = None
#     while True:
#         try:
#             #触发器
#             if cap is None or not cap.isOpened():
#                 print(f"Attempting to reconnect to {rtsp_url}...")
#                 cap = cv2.VideoCapture(rtsp_url)
#                 if not cap.isOpened():
#                     error_queue.put((rtsp_url, "Failed to connect to RTSP stream连不上了，立刻整改..."))
#                     time.sleep(5)  # 等待5秒后再重试
#                     continue
#             # 主函数
#             ret, frame = cap.read()
#             if ret:
#                 count += 1
#                 if (count % freq == 0):
#                     predict_frame(frame, rtsp_url)
#
#             if not ret:
#                 print(f"Lost connection to {rtsp_url}. Retrying...破流读不了一点儿...")
#                 cap.release()  # 释放旧的VideoCapture对象
#                 cap = None  # 设置为None以触发重连
#                 continue
#             prediction = predict_frame(frame,rtsp_url)
#             # result_queue.put(prediction)#结果入队
#             # queueHead=result_queue.get()
#
#
#         except Exception as e:
#             print(f"Error processing {rtsp_url}: {e}")
#             error_queue.put((rtsp_url, str(e)))
#             cap.release() if cap is not None else None
#             cap = None
#             time.sleep(5)  # 等待5秒后再重试



# 日志文件
log_file = "system_log.txt"


# 使用 cv2 捕获 RTSP 流数据的函数
def read_rtsp_list(queue_):
    while True:
        for rtsp in rtsp_list:
            try:
                cap = cv2.VideoCapture(rtsp)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        queue_.put((frame, rtsp))
                else:
                    print(f"Failed to connect to {rtsp}, retrying...")
                cap.release()
            except Exception as e:
                print(f"Error connecting to {rtsp}: {e}, retrying...")
            time.sleep(0)  # 不等待，立即进行下一轮循环

# 检测任务的函数
def detection_task(queue_):
    """
    从队列中获取数据并进行检测
    :param queue_: 数据队列
    """
    while True:
        if not queue_.empty():
            data,rtsp = queue_.get()
            print(f"------------------------------开始处理{rtsp[46:-18]}")  # 假设数据是图像帧，打印其形状
            predict_frame(data, rtsp)
        else:
            # 当队列为空时，进行空循环以保持进程占用处理器
            while queue_.empty():
                pass
            time.sleep(0.05)  # 避免过度占用 CPU

# 资源监控函数
def monitor_resources():
    """
    监控系统资源使用情况并记录日志
    """
    while True:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        with open(log_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%\n")
        time.sleep(10)  # 每 10 秒监控一次

def main():
    ##图片队列是全局队列

    queue_ = multiprocessing.Queue()
    read_processes = multiprocessing.Process(target=read_rtsp_list, args=(queue_,))
    read_processes.start()

    # 创建检测进程
    detection_processes = []
    for _ in range(10):
        process = multiprocessing.Process(target=detection_task, args=(queue_,))
        process.start()
        detection_processes.append(process)

    # # 创建资源监控进程
    # monitor_process = multiprocessing.Process(target=monitor_resources)
    # monitor_process.start()

if __name__ == "__main__":
    main()