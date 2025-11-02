// ImageSobel 测试Demo - 简化版
// 按步骤显示每张图片

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <dirent.h>

using namespace std;
using namespace cv;

const int FAST_MODE = 0;

int main(int argc, char** argv) {
    string imgPath;
    
    // 获取图片路径
    if (argc > 1) {
        imgPath = argv[1];
    } else {
        cout << "用法: ./test_sobel <图片路径>" << endl;
        return -1;
    }
    
    // 读取图片
    Mat frame = imread(imgPath, IMREAD_COLOR);
    if (frame.empty()) {
        cout << "错误: 无法读取图片 " << imgPath << endl;
        return -1;
    }
    
    cout << "原始尺寸: " << frame.cols << "x" << frame.rows << endl;
    
    // 缩放到摄像头分辨率
    Mat resizedFrame;
    resize(frame, resizedFrame, Size(320, 240));
    imshow("1. Resized 320x240", resizedFrame);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 灰度图
    Mat grayImage;
    cvtColor(resizedFrame, grayImage, COLOR_BGR2GRAY);
    imshow("2. Gray", grayImage);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 高斯模糊
    Mat blurredImage;
    GaussianBlur(grayImage, blurredImage, Size(5, 5), 1.5);
    imshow("3. Blurred", blurredImage);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // Sobel边缘检测
    Mat sobelY;
    Sobel(blurredImage, sobelY, CV_64F, 0, 1, 3);
    Mat gradientMagnitude = abs(sobelY);
    Mat gradientMagnitude_8u;
    convertScaleAbs(gradientMagnitude, gradientMagnitude_8u);
    imshow("4. Sobel Gradient", gradientMagnitude_8u);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 二值化 - 先用固定阈值替代OTSU（尝试更低的阈值保留更多边缘）
    Mat binaryImage;
    // 方案1：尝试较低的固定阈值（如30-50）
    threshold(gradientMagnitude_8u, binaryImage, 40, 255, THRESH_BINARY);
    
    // 方案2：如果需要测试OTSU，先打印阈值看看
    // double otsu_threshold = threshold(gradientMagnitude_8u, binaryImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
    // cout << "OTSU计算出的阈值: " << otsu_threshold << endl;
    
    imshow("5. Binary", binaryImage);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // ROI裁剪
    Mat croppedImage = binaryImage(Rect(1, 109, 318, 46));
    imshow("6. Cropped ROI", croppedImage);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 形态学操作
    Mat morphImage = croppedImage.clone();
    Mat kernel_close = getStructuringElement(MORPH_RECT, Size(9, 5));
    morphologyEx(morphImage, morphImage, MORPH_CLOSE, kernel_close);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(morphImage, morphImage, kernel, Point(-1, -1), 1);
    imshow("7. Morphed", morphImage);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // Hough直线检测
    vector<Vec4i> lines;
    HoughLinesP(morphImage, lines, 1, CV_PI / 180, 20, 15, 8);
    cout << "检测到 " << lines.size() << " 条直线" << endl;
    
    // 在原图上绘制结果
    Mat houghResult = resizedFrame.clone();
    rectangle(houghResult, Rect(1, 109, 318, 46), Scalar(0, 255, 0), 1);
    
    Mat finalImage = Mat::zeros(240, 320, CV_8U);
    
    for (const auto &l : lines) {
        double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
        double length = hypot(l[3] - l[1], l[2] - l[0]);
        
        if (abs(angle) > 15 && length > 8) {
            Vec4i adjustedLine = l;
            adjustedLine[0] += 1; adjustedLine[1] += 109;
            adjustedLine[2] += 1; adjustedLine[3] += 109;
            
            line(finalImage, Point(adjustedLine[0], adjustedLine[1]),
                 Point(adjustedLine[2], adjustedLine[3]), Scalar(255), 3, LINE_AA);
            
            line(houghResult, Point(adjustedLine[0], adjustedLine[1]),
                 Point(adjustedLine[2], adjustedLine[3]), Scalar(0, 0, 255), 2, LINE_AA);
        }
    }
    
    imshow("8. Hough Lines", houghResult);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    imshow("9. Final Result", finalImage);
    cout << "按ESC退出" << endl;
    waitKey(0);
    
    destroyAllWindows();
    cout << "测试完成!" << endl;
    return 0;
}
