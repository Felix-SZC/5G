import cv2
import os
import time
import datetime

# --- 配置 ---
RESOLUTION = (1280, 720)  # 录制分辨率 (宽度, 高度)
FPS = 10.0               # 录制帧率
OUTPUT_DIR = 'recordings'  # 视频存放的文件夹名
# --- 结束配置 ---

def record_video():
    """
    录制视频，实时显示录制时间，并保存到指定文件夹。
    按 Ctrl+C 停止录制。
    """
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        print(f"创建文件夹: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    # 设置视频捕获设备 (0 通常是默认摄像头)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        return

    # 设置摄像头的分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # 确认最终生效的分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头已打开, 实际分辨率: {width}x{height}, 帧率: {actual_fps:.2f} FPS")

    # 定义视频编码器和创建VideoWriter对象
    # 使用 'MJPG' 编码器, 生成 .avi 文件
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    # 生成带时间戳的文件名
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"rec_{timestamp}.avi"
    filepath = os.path.join(OUTPUT_DIR, filename)

    out = cv2.VideoWriter(filepath, fourcc, FPS, (width, height))
    if not out.isOpened():
        print("错误: VideoWriter 初始化失败。")
        cap.release()
        return

    print(f"开始录制... 将保存至: {filepath}")
    print("按 [Ctrl+C] 停止录制。")

    start_time = time.time()
    last_printed_time = -1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取视频帧。")
                break
            
            # 写入帧
            out.write(frame)

            # 更新并显示录制时间
            elapsed_seconds = int(time.time() - start_time)
            if elapsed_seconds > last_printed_time:
                # 使用 \r 实现单行刷新
                print(f"  已录制: {elapsed_seconds} 秒", end='\r')
                last_printed_time = elapsed_seconds

    except KeyboardInterrupt:
        # 用户按下 Ctrl+C
        print("\n检测到 [Ctrl+C]，正在停止录制...")

    finally:
        # 释放所有资源
        cap.release()
        out.release()
        print(f"\n录制结束。视频已保存: {filepath}")

if __name__ == '__main__':
    record_video()
