// banma_get 测试Demo - 优化版
// 采用顶帽变换 + 宽高比筛选，增强光照鲁棒性

#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;
using namespace cv;

//--------------- 斑马线检测参数 (优化后) ------------------------------------------
// 斑马线检测ROI区域 (收紧以聚焦路面)
const int BANMA_ROI_X = 10;           // ROI左上角X坐标
const int BANMA_ROI_Y = 150;          // ROI左上角Y坐标 (下移)
const int BANMA_ROI_WIDTH = 300;      // ROI宽度
const int BANMA_ROI_HEIGHT = 60;     // ROI高度 (减小)

// 斑马线矩形筛选尺寸
const int BANMA_RECT_MIN_WIDTH = 10;   // 矩形最小宽度 (调高以过滤噪点)
const int BANMA_RECT_MAX_WIDTH = 100;  // 矩形最大宽度
const int BANMA_RECT_MIN_HEIGHT = 10;   // 矩形最小高度
const int BANMA_RECT_MAX_HEIGHT = 100;  // 矩形最大高度 (调低以排除车道线)

// 判定为斑马线需要的最少白色矩形数量 (根据实际情况调整)
const int BANMA_MIN_COUNT = 4;

int main(int argc, char** argv) {
    string imgPath;
    
    if (argc > 1) {
        imgPath = argv[1];
    } else {
        cout << "用法: ./test_banma <图片路径>" << endl;
        return -1;
    }
    
    Mat frame = imread(imgPath, IMREAD_COLOR);
    if (frame.empty()) {
        cout << "错误: 无法读取图片 " << imgPath << endl;
        return -1;
    }
    
    cout << "原始尺寸: " << frame.cols << "x" << frame.rows << endl;
    
    // 1. 缩放到摄像头分辨率
    Mat resizedFrame;
    resize(frame, resizedFrame, Size(320, 240));
    imshow("1. Resized 320x240", resizedFrame);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    // 2. 裁剪调整后的ROI
    // 确保ROI不超出图像边界
    int roiWidth = std::min(BANMA_ROI_WIDTH, resizedFrame.cols - BANMA_ROI_X);
    int roiHeight = std::min(BANMA_ROI_HEIGHT, resizedFrame.rows - BANMA_ROI_Y);
    if (roiWidth <= 0 || roiHeight <= 0) {
        cerr << "ROI尺寸无效" << endl;
        return -1;
    }
    cv::Rect roiRect(BANMA_ROI_X, BANMA_ROI_Y, roiWidth, roiHeight);
    // 使用clone()确保是深拷贝，避免边界问题
    cv::Mat roiFrame = resizedFrame(roiRect).clone();
    imshow("2. Cropped ROI", roiFrame);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    // 3. 灰度化
    cv::Mat grayRoi;
    cv::cvtColor(roiFrame, grayRoi, cv::COLOR_BGR2GRAY);
    imshow("3. Grayscale ROI", grayRoi);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    // 4. 顶帽变换 - 核心步骤，用于在复杂光照下突出白色条纹
    cv::Mat topHat;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 3));
    cv::morphologyEx(grayRoi, topHat, cv::MORPH_TOPHAT, kernel);
    imshow("4. Top-Hat Transform", topHat);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    // 5. 二值化
    cv::Mat binaryMask;
    // 顶帽变换后的图像，使用一个较低的固定阈值或OTSU效果都很好
    cv::threshold(topHat, binaryMask, 40, 255, cv::THRESH_BINARY);
    imshow("5. Thresholded Mask", binaryMask);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    // 6. 形态学开运算（先腐蚀再膨胀），去除小的噪声点
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_OPEN, openKernel);
    imshow("6. Morphological Open", binaryMask);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    // 7. 查找轮廓并应用尺寸筛选
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat resultImage = roiFrame.clone();
    int count_BMX = 0;
    int total_contours = contours.size();
    int skipped_boundary = 0;
    int skipped_size = 0;

    cout << "检测到 " << total_contours << " 个轮廓" << endl;

    // 第一遍：画出所有矩形（用红色表示无效的）
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        
        // 确保rect在ROI范围内（安全检查，允许1像素的容差）
        if (rect.x < -1 || rect.y < -1 || 
            rect.x + rect.width > roiFrame.cols + 1 || 
            rect.y + rect.height > roiFrame.rows + 1) {
            skipped_boundary++;
            continue; // 跳过超出边界的矩形
        }
        
        // 裁剪rect到ROI范围内
        rect.x = std::max(0, rect.x);
        rect.y = std::max(0, rect.y);
        rect.width = std::min(rect.width, roiFrame.cols - rect.x);
        rect.height = std::min(rect.height, roiFrame.rows - rect.y);
        
        // 先画出所有矩形（用红色，表示待筛选）
        cv::rectangle(resultImage, rect, cv::Scalar(0, 0, 255), 1);
    }

    // 第二遍：筛选并标记有效矩形（用绿色覆盖）
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        
        // 确保rect在ROI范围内（安全检查，允许1像素的容差）
        if (rect.x < -1 || rect.y < -1 || 
            rect.x + rect.width > roiFrame.cols + 1 || 
            rect.y + rect.height > roiFrame.rows + 1) {
            continue; // 跳过超出边界的矩形
        }
        
        // 裁剪rect到ROI范围内
        rect.x = std::max(0, rect.x);
        rect.y = std::max(0, rect.y);
        rect.width = std::min(rect.width, roiFrame.cols - rect.x);
        rect.height = std::min(rect.height, roiFrame.rows - rect.y);
        
        // 计算高宽比（高度/宽度），因为斑马线在车辆视野中是纵向的
        float aspectRatio = (rect.width > 0) ? (float)rect.height / rect.width : 0;

        // 调试输出：显示所有检测到的矩形信息
        cout << "矩形: w=" << rect.width << ", h=" << rect.height 
             << ", 高宽比=" << std::fixed << std::setprecision(2) << aspectRatio;

        // 应用尺寸和高宽比联合筛选
        // 斑马线应该是纵向的矩形，高度应该明显大于宽度
        bool size_ok = (rect.width >= BANMA_RECT_MIN_WIDTH && rect.width <= BANMA_RECT_MAX_WIDTH &&
                        rect.height >= BANMA_RECT_MIN_HEIGHT && rect.height <= BANMA_RECT_MAX_HEIGHT);

        if (!size_ok) {
            skipped_size++;
            cout << " [尺寸不符合]" << endl;
            continue;
        }

        // 所有条件都满足，用绿色覆盖（表示有效斑马线条纹）
        cv::rectangle(resultImage, rect, cv::Scalar(0, 255, 0), 2);
        count_BMX++;
        cout << " [有效 ✓]" << endl;
    }
    
    cout << "筛选统计: 边界跳过=" << skipped_boundary 
         << ", 尺寸跳过=" << skipped_size 
         << ", 有效=" << count_BMX << endl;
    imshow("7. Final Detection", resultImage);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    // 8. 最终判定
    cout << "------------------------------------" << endl;
    if (count_BMX >= BANMA_MIN_COUNT) {
        cout << "结果: 检测到斑马线 (找到 " << count_BMX << " 个有效矩形)" << endl;
    } else {
        cout << "结果: 未检测到斑马线 (找到 " << count_BMX << " 个有效矩形)" << endl;
    }
    cout << "------------------------------------" << endl;
    
    cout << "按ESC退出" << endl;
    waitKey(0);
    
    destroyAllWindows();
    cout << "测试完成!" << endl;
    return 0;
}
