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
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
TARGET_FPS = 10  # 目标帧率


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
    
    # 清空初始缓冲区 - 丢弃前几帧，避免旧帧堆积
    print("正在清空摄像头缓冲区...")
    for _ in range(5):
        cap.read()
    
    # 用于计算实际帧率
    frame_count = 0
    last_fps_time = time.time()
    fps_display = 0.0
    fps_frame_count = 0
    
    # 用于帧跳过机制 - 如果处理跟不上，跳过旧帧
    last_frame_time = time.time()
    target_frame_interval = 1.0 / TARGET_FPS
    
    try:
        while True:
            # 检查是否需要跳过帧（如果处理速度跟不上）
            current_time = time.time()
            time_since_last_frame = current_time - last_frame_time
            
            # 如果距离上一帧时间太短，说明处理速度跟不上，跳过这一帧
            if time_since_last_frame < target_frame_interval * 0.5:
                # 读取并丢弃这一帧，避免缓冲区堆积
                cap.read()
                continue
            
            # 读取一帧
            ret, frame = cap.read()
            
            if not ret:
                print("错误: 无法读取摄像头画面!")
                break
            
            frame_count += 1
            fps_frame_count += 1
            last_frame_time = current_time
            
            # 每秒计算一次实际帧率（更准确）
            elapsed_since_fps = current_time - last_fps_time
            if elapsed_since_fps >= 1.0:
                fps_display = fps_frame_count / elapsed_since_fps
                fps_frame_count = 0
                last_fps_time = current_time
            
            # 在画面上显示信息
            info_text = f"Resolution: {actual_width}x{actual_height} | FPS: {fps_display:.1f}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示画面
            cv2.imshow(window_name, frame)
            
            # 检测按键，按 'q' 退出
            # waitKey(10) 在树莓派上更稳定，避免CPU占用过高
            key = cv2.waitKey(10) & 0xFF
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
