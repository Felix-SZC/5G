import cv2
import time

def check_camera_fps():
    """
    测试摄像头在不同分辨率下的最大实际帧率。
    """
    # 常见的摄像头分辨率列表
    resolutions = {
        "320x240": (320, 240),
        "640x480": (640, 480),
        "800x600": (800, 600),
        "1280x720": (1280, 720), # 720p
        "1920x1080": (1920, 1080) # 1080p
    }

    print("正在检测摄像头支持的最大帧率...")
    print("------------------------------------")

    # 尝试打开默认摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头。请检查摄像头是否连接正常。")
        return

    for name, (width, height) in resolutions.items():
        # 尝试设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 确认实际生效的分辨率
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 如果摄像头不支持此分辨率，实际分辨率可能不会改变
        # 为避免重复测试，我们只在分辨率成功改变时测试
        if actual_width != width or actual_height != height:
            # print(f"跳过不支持的分辨率: {name}")
            continue

        # --- 开始帧率测试 ---
        num_frames_to_capture = 100
        print(f"测试分辨率: {actual_width}x{actual_height}...")
        
        # "热身" - 丢弃前几帧，让摄像头稳定
        for _ in range(10):
            cap.read()

        start_time = time.time()

        for i in range(num_frames_to_capture):
            ret, frame = cap.read()
            if not ret:
                print("  -> 错误: 无法抓取帧。")
                break
        
        end_time = time.time()
        
        if end_time > start_time:
            # 计算实际帧率
            elapsed_time = end_time - start_time
            actual_fps = num_frames_to_capture / elapsed_time
            print(f"  -> 实测最大帧率: {actual_fps:.2f} FPS\n")
        else:
            print("  -> 测试时间过短，无法计算帧率。\n")

    # 释放摄像头
    cap.release()
    print("------------------------------------")
    print("检测完成。")


if __name__ == '__main__':
    check_camera_fps()
