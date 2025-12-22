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

std::vector<int> lane_widths; // 存储不同高度对应的赛道宽度

// 功能: 初始化赛道宽度查找表
void initialize_lane_widths() {
    lane_widths.assign(240, 0); // 为0-239像素高度初始化

    // 给定数据点
    const int y1 = 170, width1 = 320;
    const int y2 = 130, width2 = 180;

    // 线性插值: width = m*y + c
    double m = static_cast<double>(width1 - width2) / (y1 - y2);
    double c = width1 - m * y1;

    for (int y = 0; y < 240; ++y) {
        int width = static_cast<int>(m * y + c);
        // 边界保护
        if (width < 1) width = 1;
        if (width > 320) width = 320;
        lane_widths[y] = width;
    }
    std::cout << "[初始化] 赛道宽度查找表生成完毕。" << std::endl;
}

const int FAST_MODE = 0;
const int MIN_COMPONENT_AREA = 100;

int main(int argc, char** argv) {
    initialize_lane_widths();
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
    addWeighted(absSobelY, 1.0, absSobelX, 1.0, 0, gradientMagnitude8u);

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
    threshold(gradientMagnitude8u, gradientMask, 30, 255, THRESH_BINARY);
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
    
    // 先进行连通域分析，过滤掉不符合面积要求的连通域
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(binaryMask, labels, stats, centroids, 8, CV_32S);
    Mat filteredByArea = Mat::zeros(binaryMask.size(), CV_8U);
    for (int i = 1; i < numLabels; ++i) {
        if (stats.at<int>(i, CC_STAT_AREA) >= MIN_COMPONENT_AREA) {
            filteredByArea.setTo(255, labels == i);
        }
    }
    
    imshow("10. Area Filtered ROI", filteredByArea);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 形态学操作 (在面积过滤后的结果上进行)
    static cv::Mat kernel_close = getStructuringElement(MORPH_RECT, Size(1, 9));
    morphologyEx(filteredByArea, filteredByArea, MORPH_CLOSE, kernel_close);

    Mat filteredMorph = filteredByArea;

    imshow("11. Morphed ROI", filteredMorph);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // Hough直线检测
    vector<Vec4i> lines;
    HoughLinesP(filteredMorph, lines, 1, CV_PI / 180, 8, 30, 25);
    cout << "检测到 " << lines.size() << " 条直线" << endl;

    // 在原图上绘制结果
    Mat houghResult = resizedFrame.clone();
    rectangle(houghResult, Rect(1, 109, 318, 46), Scalar(0, 255, 0), 1);

    Mat finalImage = Mat::zeros(240, 320, CV_8U);

    // 筛选最佳左右线
    vector<Vec4i> leftLines, rightLines;
    for (const auto &l : lines) {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        if (abs(dx) < 1e-6) continue; // 忽略垂直线

        double slope = dy / dx; // 斜率
        double angle = atan2(dy, dx) * 180.0 / CV_PI;

        // 简单的角度过滤，忽略过于水平的线
        if (abs(angle) < 5.0) continue;

        // 左边线大概长"/" (slope < 0), 右边线长"\" (slope > 0)
        // 注意OpenCV图像坐标系Y轴向下
        // "/" : x增加y减少 => dy < 0, dx > 0 => slope < 0
        // "\" : x增加y增加 => dy > 0, dx > 0 => slope > 0
        if (slope < 0) {
            leftLines.push_back(l);
        } else {
            rightLines.push_back(l);
        }
    }

    cout << "左候选线: " << leftLines.size() << ", 右候选线: " << rightLines.size() << endl;

    Vec4i bestLeft = {0, 0, 0, 0}, bestRight = {0, 0, 0, 0};
    bool foundPair = false;
    double minError = 1e9;
    
    // ROI在全局图像中的Y偏移
    int roi_y_offset = 109;
    // 我们在ROI的中间位置计算宽度进行校验
    int check_y_local = 23; 
    int check_y_global = roi_y_offset + check_y_local;
    int ideal_width = 0;
    
    if (check_y_global >= 0 && static_cast<size_t>(check_y_global) < lane_widths.size()) {
        ideal_width = lane_widths[check_y_global];
    } else {
        cout << "[警告] 校验行超出范围" << endl;
    }
    
    // --- Pass 1: Find the minimum possible width error ---
    if (ideal_width > 0) {
        for (const auto& l_left : leftLines) {
            for (const auto& l_right : rightLines) {
                double k_left = (double)(l_left[3] - l_left[1]) / (l_left[2] - l_left[0]);
                double x_left = l_left[0] + (check_y_local - l_left[1]) / k_left;
                double k_right = (double)(l_right[3] - l_right[1]) / (l_right[2] - l_right[0]);
                double x_right = l_right[0] + (check_y_local - l_right[1]) / k_right;
                double width_calc = x_right - x_left;
                if (width_calc > 0) {
                    minError = std::min(minError, abs(width_calc - ideal_width));
                }
            }
        }
    }

    // --- Pass 2: Find the longest pair within the dynamic tolerance ---
    if (ideal_width > 0 && minError < 1e9) {
        const double dynamic_tolerance = minError + 10.0; // Dynamic tolerance
        double maxTotalLength = 0;

        for (const auto& l_left : leftLines) {
            for (const auto& l_right : rightLines) {
                double k_left = (double)(l_left[3] - l_left[1]) / (l_left[2] - l_left[0]);
                double x_left = l_left[0] + (check_y_local - l_left[1]) / k_left;
                double k_right = (double)(l_right[3] - l_right[1]) / (l_right[2] - l_right[0]);
                double x_right = l_right[0] + (check_y_local - l_right[1]) / k_right;
                double width_calc = x_right - x_left;

                if (width_calc > 0) {
                    double error = abs(width_calc - ideal_width);
                    if (error <= dynamic_tolerance) {
                        double len_left = hypot(l_left[2] - l_left[0], l_left[3] - l_left[1]);
                        double len_right = hypot(l_right[2] - l_right[0], l_right[3] - l_right[1]);
                        double totalLength = len_left + len_right;

                        if (totalLength > maxTotalLength) {
                            maxTotalLength = totalLength;
                            bestLeft = l_left;
                            bestRight = l_right;
                            foundPair = true;
                        }
                    }
                }
            }
        }
    }

    if (foundPair) {
        cout << "找到最佳匹配线对，宽度误差: " << minError << " (理想: " << ideal_width << ")" << endl;
        
        // 绘制最佳左线
        Vec4i adjustedLeft = bestLeft;
        adjustedLeft[0] += roiRect.x; adjustedLeft[1] += roiRect.y;
        adjustedLeft[2] += roiRect.x; adjustedLeft[3] += roiRect.y;
        
        line(finalImage, Point(adjustedLeft[0], adjustedLeft[1]),
             Point(adjustedLeft[2], adjustedLeft[3]), Scalar(255), 3, LINE_AA);
        line(houghResult, Point(adjustedLeft[0], adjustedLeft[1]),
             Point(adjustedLeft[2], adjustedLeft[3]), Scalar(0, 255, 255), 2, LINE_AA); // 黄色

        // 绘制最佳右线
        Vec4i adjustedRight = bestRight;
        adjustedRight[0] += roiRect.x; adjustedRight[1] += roiRect.y;
        adjustedRight[2] += roiRect.x; adjustedRight[3] += roiRect.y;

        line(finalImage, Point(adjustedRight[0], adjustedRight[1]),
             Point(adjustedRight[2], adjustedRight[3]), Scalar(255), 3, LINE_AA);
        line(houghResult, Point(adjustedRight[0], adjustedRight[1]),
             Point(adjustedRight[2], adjustedRight[3]), Scalar(0, 255, 255), 2, LINE_AA); // 黄色
             
    } else {
        cout << "未找到合适的左右线对" << endl;
        // 如果没有找到匹配的对，可以选择画出所有或者什么都不画
        // 这里为了演示，我们画出所有候选线（用不同颜色，例如蓝色表示左，红色表示右）
        for (const auto& l : leftLines) {
             line(houghResult, Point(l[0]+roiRect.x, l[1]+roiRect.y), Point(l[2]+roiRect.x, l[3]+roiRect.y), Scalar(255, 0, 0), 1, LINE_AA);
        }
        for (const auto& l : rightLines) {
             line(houghResult, Point(l[0]+roiRect.x, l[1]+roiRect.y), Point(l[2]+roiRect.x, l[3]+roiRect.y), Scalar(0, 0, 255), 1, LINE_AA);
        }
    }

    imshow("12. Hough Lines (Filtered)", houghResult);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    imshow("13. Final Result", finalImage);
    cout << "按ESC退出" << endl;
    waitKey(0);
    
    destroyAllWindows();
    cout << "测试完成!" << endl;
    return 0;
}
