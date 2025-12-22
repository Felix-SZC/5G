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
    
    // --- START: New lane filtering logic ---

    // A. 初始化赛道宽度查找表 (逻辑从 main.cpp 移植)
    std::vector<int> lane_widths;
    lane_widths.assign(240, 0); // 对应 320x240 完整图像高度
    const int y1 = 170, width1 = 320;
    const int y2 = 130, width2 = 180;
    double m = static_cast<double>(width1 - width2) / (y1 - y2);
    double c = width1 - m * y1;
    for (int y = 0; y < 240; ++y) {
        int width = static_cast<int>(m * y + c);
        if (width < 1) width = 1;
        if (width > 320) width = 320;
        lane_widths[y] = width;
    }

    // B. 连通域分析，找出所有候选线条
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(binaryMask, labels, stats, centroids, 8, CV_32S);

    std::vector<int> candidate_labels;
    Mat allCandidatesMask = Mat::zeros(binaryMask.size(), CV_8U);
    for (int i = 1; i < numLabels; ++i) {
        if (stats.at<int>(i, CC_STAT_AREA) >= MIN_COMPONENT_AREA) {
            candidate_labels.push_back(i);
            allCandidatesMask.setTo(255, labels == i);
        }
    }

    imshow("10a. All Candidates by Area", allCandidatesMask);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    // C. 将候选线条分到左/右组
    std::vector<int> left_candidates;
    std::vector<int> right_candidates;
    int image_center_x = binaryMask.cols / 2;
    for (int label : candidate_labels) {
        double centroid_x = centroids.at<double>(label, 0);
        if (centroid_x < image_center_x) {
            left_candidates.push_back(label);
        } else {
            right_candidates.push_back(label);
        }
    }

    // D. 遍历所有左右组合，根据宽度模型评分，找出最佳组合
    long best_error = -1;
    int best_left_label = -1;
    int best_right_label = -1;

    for (int left_label : left_candidates) {
        for (int right_label : right_candidates) {
            long current_pair_error = 0;
            int valid_rows = 0;

            for (int y_roi = 0; y_roi < roiRect.height; ++y_roi) {
                int y_frame = y_roi + roiRect.y;

                int lx = -1; // 左线条在本行的最右侧 x 坐标
                int rx = -1; // 右线条在本行的最左侧 x 坐标

                for (int x = roiRect.width - 1; x >= 0; --x) {
                    if (labels.at<int>(y_roi, x) == left_label) {
                        lx = x;
                        break;
                    }
                }

                for (int x = 0; x < roiRect.width; ++x) {
                    if (labels.at<int>(y_roi, x) == right_label) {
                        rx = x;
                        break;
                    }
                }

                if (lx != -1 && rx != -1 && rx > lx) {
                    int actual_width = rx - lx;
                    if (y_frame >= 0 && y_frame < (int)lane_widths.size()) {
                         int expected_width = lane_widths[y_frame];
                         current_pair_error += std::abs(actual_width - expected_width);
                         valid_rows++;
                    }
                }
            }

            if (valid_rows > 10) { // 要求至少有10行匹配才认为是有效组合
                long average_error = current_pair_error / valid_rows;
                if (best_error == -1 || average_error < best_error) {
                    best_error = average_error;
                    best_left_label = left_label;
                    best_right_label = right_label;
                }
            }
        }
    }

    // E. 生成只包含最佳组合的二值图
    Mat filteredByLanes = Mat::zeros(binaryMask.size(), CV_8U);
    if (best_left_label != -1 && best_right_label != -1) {
        cout << "最佳组合: 左 " << best_left_label << ", 右 " << best_right_label << " (平均误差: " << best_error << ")" << endl;
        filteredByLanes.setTo(255, labels == best_left_label);
        filteredByLanes.setTo(255, labels == best_right_label);
    } else {
        cout << "警告: 未找到有效的车道线组合!" << endl;
    }

    imshow("10b. Best Pair Filtered", filteredByLanes);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    // --- END: New lane filtering logic ---

    // 形态学操作 (在筛选出的最佳车道线上进行)
    // 使用细而高的结构元素，只在竖直方向（y 方向）做闭运算，用来“接上”竖直方向断裂
    static cv::Mat kernel_close = getStructuringElement(MORPH_RECT, Size(1, 9));
    morphologyEx(filteredByLanes, filteredByLanes, MORPH_CLOSE, kernel_close);
    // 不再做额外的膨胀，避免车道线明显变粗

    Mat filteredMorph = filteredByLanes;

    imshow("11. Morphed ROI", filteredMorph);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 将ROI内的结果图像复制回一个320x240的黑色背景图像，以匹配main.cpp中的输出格式
    Mat finalImage = Mat::zeros(Size(320, 240), CV_8U);
    filteredMorph.copyTo(finalImage(roiRect));

    // 在一个克隆的原图上绘制ROI矩形，用于对比
    Mat resultOnOriginal = resizedFrame.clone();
    rectangle(resultOnOriginal, roiRect, Scalar(0, 255, 0), 1);
    
    // 将二值图像转换为彩色，以便叠加显示
    Mat binaryColor;
    cvtColor(finalImage, binaryColor, COLOR_GRAY2BGR);
    // 将白色区域标记为红色
    binaryColor.setTo(Scalar(0, 0, 255), finalImage == 255);
    
    // 将处理结果（红色车道线）半透明叠加到原图上
    addWeighted(resultOnOriginal, 1.0, binaryColor, 0.7, 0.0, resultOnOriginal);


    imshow("12. Final Result Overlay", resultOnOriginal);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    imshow("13. Final Binary Image", finalImage);
    cout << "按ESC退出" << endl;
    waitKey(0);
    
    destroyAllWindows();
    cout << "测试完成!" << endl;
    return 0;
}
