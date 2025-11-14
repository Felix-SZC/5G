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
const int MIN_COMPONENT_AREA = 400;

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
    
    Rect roiRect(1, 109, 318, 46);
    Mat roi = resizedFrame(roiRect);
    imshow("2. ROI", roi);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // ROI 灰度图
    Mat grayRoi;
    cvtColor(roi, grayRoi, COLOR_BGR2GRAY);
    imshow("3. Gray ROI", grayRoi);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 均值滤波
    Mat blurredRoi;
    blur(grayRoi, blurredRoi, Size(5, 5));
    imshow("4. Blurred ROI", blurredRoi);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // Sobel边缘检测
    Mat sobelX, sobelY;
    // 使用CV_16S以提高性能，避免使用昂贵的CV_64F浮点运算
    Sobel(blurredRoi, sobelX, CV_16S, 1, 0, 3);
    Sobel(blurredRoi, sobelY, CV_16S, 0, 1, 3);

    // 转换回CV_8U并计算梯度
    Mat absSobelX, absSobelY;
    convertScaleAbs(sobelX, absSobelX);
    convertScaleAbs(sobelY, absSobelY);

    // 组合梯度，权重偏向Y方向
    Mat gradientMagnitude8u;
    addWeighted(absSobelY, 1.0, absSobelX, 0.5, 0, gradientMagnitude8u);

    imshow("5. Sobel Gradient ROI", gradientMagnitude8u);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 顶帽操作减弱阴影
    Mat topHat;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(20, 3));
    morphologyEx(blurredRoi, topHat, MORPH_TOPHAT, kernel);
    imshow("6. Top-hat", topHat);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    Mat adaptiveMask;
    threshold(topHat, adaptiveMask, 5, 255, THRESH_BINARY);
    imshow("7. Top-hat Threshold", adaptiveMask);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    Mat gradientMask;
    threshold(gradientMagnitude8u, gradientMask, 15, 255, THRESH_BINARY);
    Mat gradientKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(gradientMask, gradientMask, gradientKernel);
    imshow("8. Gradient Mask ROI", gradientMask);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    Mat binaryMask;
    bitwise_and(adaptiveMask, gradientMask, binaryMask);
    medianBlur(binaryMask, binaryMask, 3);

    // Mat noiseKernel = getStructuringElement(MORPH_RECT, Size(1, 1));
    // morphologyEx(binaryMask, binaryMask, MORPH_OPEN, noiseKernel);

    imshow("9. Binary Mask ROI", binaryMask);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 形态学操作 (原地，无克隆)
    static cv::Mat kernel_close = getStructuringElement(MORPH_RECT, Size(9, 5));
    morphologyEx(binaryMask, binaryMask, MORPH_CLOSE, kernel_close);
    static cv::Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(binaryMask, binaryMask, kernel_dilate, Point(-1, -1), 1);

    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(binaryMask, labels, stats, centroids, 8, CV_32S);
    Mat filteredMorph = Mat::zeros(binaryMask.size(), CV_8U);
    for (int i = 1; i < numLabels; ++i) {
        if (stats.at<int>(i, CC_STAT_AREA) >= MIN_COMPONENT_AREA) {
            filteredMorph.setTo(255, labels == i);
        }
    }

    imshow("10. Morphed ROI", filteredMorph);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // Hough直线检测
    vector<Vec4i> lines;
    HoughLinesP(filteredMorph, lines, 1, CV_PI / 180, 20, 15, 8);
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
            adjustedLine[0] += roiRect.x; adjustedLine[1] += roiRect.y;
            adjustedLine[2] += roiRect.x; adjustedLine[3] += roiRect.y;

            line(finalImage, Point(adjustedLine[0], adjustedLine[1]),
                 Point(adjustedLine[2], adjustedLine[3]), Scalar(255), 3, LINE_AA);

            line(houghResult, Point(adjustedLine[0], adjustedLine[1]),
                 Point(adjustedLine[2], adjustedLine[3]), Scalar(0, 0, 255), 2, LINE_AA);
        }
    }

    imshow("11. Hough Lines", houghResult);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    imshow("12. Final Result", finalImage);
    cout << "按ESC退出" << endl;
    waitKey(0);
    
    destroyAllWindows();
    cout << "测试完成!" << endl;
    return 0;
}
