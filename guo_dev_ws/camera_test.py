#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头显示脚本 - 1280x720分辨率
打开摄像头并以1280x720分辨率显示画面
按 'q' 键退出
"""

import cv2
import sys
import time


# 配置参数
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30  # 目标帧率


def show_camera():
    """
    以指定分辨率打开摄像头并显示画面
    """
    print("=" * 50)
    print("摄像头显示程序")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误: 无法打开摄像头!")
        print("请检查:")
        print("  1. 摄像头是否已连接")
        print("  2. 摄像头是否被其他程序占用")
        print("  3. 摄像头驱动是否正常")
        return 1
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    # 设置缓冲区大小为1，减少延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 读取实际生效的参数
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("-" * 50)
    print(f"摄像头信息:")
    print(f"  目标分辨率: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"  实际分辨率: {actual_width}x{actual_height}")
    print(f"  目标帧率: {TARGET_FPS} FPS")
    print(f"  摄像头报告帧率: {actual_fps:.2f} FPS")
    print("-" * 50)
    print("按 'q' 键退出程序")
    print("=" * 50)
    
    # 创建窗口
    window_name = f'Camera - {actual_width}x{actual_height}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 用于计算实际帧率
    frame_count = 0
    start_time = time.time()
    fps_display = 0.0
    
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            
            if not ret:
                print("错误: 无法读取摄像头画面!")
                break
            
            frame_count += 1
            
            # 每30帧计算一次实际帧率
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps_display = 30 / elapsed_time
                start_time = time.time()
            
            # 在画面上显示信息
            info_text = f"Resolution: {actual_width}x{actual_height} | FPS: {fps_display:.1f}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示画面
            cv2.imshow(window_name, frame)
            
            # 检测按键，按 'q' 退出
            # waitKey(1) 表示等待1毫秒，这样可以达到更高的帧率
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print(f"\n用户按下 'q' 键，退出程序...")
                print(f"总共处理了 {frame_count} 帧")
                break
                
    except KeyboardInterrupt:
        print(f"\n接收到中断信号 (Ctrl+C)，退出程序...")
        print(f"总共处理了 {frame_count} 帧")
    
    finally:
        # 释放资源
        print("正在释放摄像头资源...")
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出。")
    
    return 0


if __name__ == "__main__":
    sys.exit(show_camera())
