# ImageSobel 测试Demo使用说明

## 功能
这个测试程序用于可视化 `ImageSobel` 函数的每个处理步骤，帮助调试和理解稀疏跑道线条的检测效果。

## 编译
在 guo_dev_ws 目录下执行：
```bash
mkdir build
cd build
cmake ..
make test_sobel
```

## 使用方法

### 方法1：将图片放到test目录
1. 将测试图片（jpg、jpeg或png格式）放到 `test/` 目录下
2. 运行程序：
```bash
./test_sobel
```
程序会自动读取test目录中的第一张图片

### 方法2：指定图片路径
```bash
./test_sobel /path/to/your/image.jpg
```

## 显示内容
程序会在一个新窗口中显示以下10个处理步骤：

**第一行（预处理）：**
1. **Original** - 原始输入图像
2. **Resized** - 缩放到320x240
3. **Gray** - 灰度图
4. **Blurred** - 高斯模糊后

**第二行（边缘检测）：**
5. **Sobel Gradient** - Sobel边缘检测结果
6. **Binary** - OTSU自适应阈值二值化
7. **Cropped ROI** - ROI区域裁剪
8. **Morphed** - 形态学处理（闭运算+膨胀）

**第三行（直线检测）：**
9. **Hough Lines** - 霍夫直线检测结果（在原图上用红线标记）
10. **Final Result** - 最终结果（供Tracking函数使用）

## 键盘控制
- **空格键**：重新显示结果
- **ESC**：退出程序

## 注意事项
1. 确保树莓派上已正确安装OpenCV
2. 如果图片路径错误或无法读取，程序会提示错误信息
3. 程序会按照摄像头分辨率（320x240）缩放图像进行测试

