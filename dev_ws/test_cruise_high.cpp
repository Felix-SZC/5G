#include <iostream>
#include <cstdlib>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <string>
#include <pigpio.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;

// 全局标志，用于信号处理
bool program_finished = false;
std::chrono::steady_clock::time_point cruise_high_start_time;
bool cruise_high_started = false;

//------------速度参数配置------------------------------------------------------------------------------------------
const int MOTOR_SPEED_DELTA_CRUISE_FAST = 2500;      // 常规巡航速度增量（高速）

//------------时间参数配置（单位：秒）------------------------------------------------------------------------------------------
const float START_DELAY_SECONDS = 2.0f;              // 发车延时时间（秒）

//------------状态机定义------------------------------------------------------------------------------------------
enum class CarState {
    Idle,           // 等待发车
    StartDelay,     // 发车延时
    Cruise_High     // 高速巡航
};

CarState current_state = CarState::Idle;

// 功能: 将CarState枚举转换为可读字符串
std::string carStateToString(CarState state) {
    switch (state) {
        case CarState::Idle:           return "Idle (等待发车)";
        case CarState::StartDelay:     return "StartDelay (发车延时)";
        case CarState::Cruise_High:    return "Cruise_High (高速巡航)";
        default:                       return "Unknown";
    }
}

// 功能: 设置车辆状态并打印状态变更日志
void setCarState(CarState newState) {
    if (current_state != newState) {
        std::cout << "[状态变更] " << carStateToString(current_state) 
                  << " -> " << carStateToString(newState) << std::endl;

        current_state = newState;
        
        // 记录进入高速巡航的时间
        if (newState == CarState::Cruise_High) {
            cruise_high_start_time = std::chrono::steady_clock::now();
            cruise_high_started = true;
            std::cout << "[时间统计] 开始计时，按 Ctrl+C 结束测试" << std::endl;
        }
    }
}

// 状态上下文变量
std::chrono::steady_clock::time_point start_delay_time;

//-----------------图像处理相关----------------------------------------------
Mat frame; // 存储视频帧
Mat bin_image; // 存储二值化图像--Sobel检测后图像

//---------------蓝色挡板发车相关----------------------------------------------
int find_first = 0; // 标记是否第一次找到蓝色挡板
int blue_detect_count = 0; // 蓝色挡板连续检测计数
const int BLUE_DETECT_THRESHOLD = 3; // 需要连续检测到的帧数才能确认找到蓝色挡板

// HSV颜色范围
const int BLUE_H_MIN = 100;  // 色调H最小值
const int BLUE_H_MAX = 130;  // 色调H最大值
const int BLUE_S_MIN = 50;   // 饱和度S最小值
const int BLUE_S_MAX = 255;  // 饱和度S最大值
const int BLUE_V_MIN = 50;   // 亮度V最小值
const int BLUE_V_MAX = 255;  // 亮度V最大值

// 蓝色检测ROI区域（限制检测范围）
const int BLUE_ROI_X = 90;      // ROI左上角X坐标
const int BLUE_ROI_Y = 80;      // ROI左上角Y坐标
const int BLUE_ROI_WIDTH = 220;  // ROI宽度
const int BLUE_ROI_HEIGHT = 100; // ROI高度

const double BLUE_AREA_VALID = 2000.0; // 有效面积阈值
const double BLUE_REMOVE_AREA_MIN = 500.0; // 移开检测的最小面积阈值（过滤小噪点）

//-----------------巡线相关-----------------------------------------------
std::vector<cv::Point> mid; // 存储中线
std::vector<cv::Point> left_line; // 存储左线条
std::vector<cv::Point> right_line; // 存储右线条
std::vector<cv::Point> last_mid; // 存储上一次的中线，用于平滑滤波

int error_first; // 存储第一次误差
int last_error;
float servo_pwm_diff; // 存储舵机PWM差值
float servo_pwm; // 存储舵机PWM值

// 赛道宽度查找表
std::vector<int> lane_widths; // 存储不同高度对应的赛道宽度

//---------------舵机相关---------------------------------------------
const int servo_pin = 12; // 存储舵机引脚号
const float servo_pwm_range = 10000.0; // 存储舵机PWM范围
const float servo_pwm_frequency = 50.0; // 存储舵机PWM频率
const float servo_pwm_duty_cycle_unlock = 730.0; // 存储舵机PWM占空比解锁值

float servo_pwm_mid = servo_pwm_duty_cycle_unlock; // 存储舵机中值

//---------------电机相关---------------------------------------------
const int motor_pin = 13; // 存储电机引脚号
const float motor_pwm_range = 40000.0; // 存储电机PWM范围
const float motor_pwm_frequency = 200.0; // 存储电机PWM频率
const float motor_pwm_duty_cycle_unlock = 11400.0; // 存储电机PWM占空比解锁值

float motor_pwm_mid = motor_pwm_duty_cycle_unlock; // 存储电机PWM初始化值

//---------------云台相关---------------------------------------------
const int yuntai_LR_pin = 22; // 存储云台引脚号
const float yuntai_LR_pwm_range = 1000.0; // 存储云台PWM范围
const float yuntai_LR_pwm_frequency = 50.0; // 存储云台PWM频率
const float yuntai_LR_pwm_duty_cycle_unlock = 63.0; //大左小右 

const int yuntai_UD_pin = 23; // 存储云台引脚号
const float yuntai_UD_pwm_range = 1000.0; // 存储云台PWM范围
const float yuntai_UD_pwm_frequency = 50.0; // 存储云台PWM频率
const float yuntai_UD_pwm_duty_cycle_unlock = 55.0; //大上下小

// 信号处理函数
void signalHandler(int signum) {
    std::cout << "\n[信号] 接收到中断信号 (Ctrl+C)" << std::endl;
    
    if (cruise_high_started) {
        auto end_time = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - cruise_high_start_time).count() / 1000000.0;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "[时间统计] 高速巡线总时间: " << std::fixed << std::setprecision(2) 
                  << total_duration << " 秒" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    
    // 停止电机和舵机
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock);
    gpioPWM(servo_pin, servo_pwm_mid);
    
    program_finished = true;
}

// 功能: 初始化舵机、电机与云台PWM，完成GPIO库初始化
void servo_motor_pwmInit(void) 
{
    if (gpioInitialise() < 0) {
        std::cout << "GPIO初始化失败！请使用sudo权限运行！" << std::endl;
        return;
    }
    else
        std::cout << "GPIO初始化成功，系统正常！" << std::endl;

    gpioSetMode(servo_pin, PI_OUTPUT);
    gpioSetPWMfrequency(servo_pin, servo_pwm_frequency);
    gpioSetPWMrange(servo_pin, servo_pwm_range);
    gpioPWM(servo_pin, servo_pwm_duty_cycle_unlock);

    gpioSetMode(motor_pin, PI_OUTPUT);
    gpioSetPWMfrequency(motor_pin, motor_pwm_frequency);
    gpioSetPWMrange(motor_pin, motor_pwm_range);
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock);

    gpioSetMode(yuntai_LR_pin, PI_OUTPUT); // 设置云台引脚为输出模式
    gpioSetPWMfrequency(yuntai_LR_pin, yuntai_LR_pwm_frequency); // 设置云台PWM频率
    gpioSetPWMrange(yuntai_LR_pin, yuntai_LR_pwm_range); // 设置云台PWM范围
    gpioPWM(yuntai_LR_pin, yuntai_LR_pwm_duty_cycle_unlock); // 设置云台PWM占空比解锁值

    gpioSetMode(yuntai_UD_pin, PI_OUTPUT); // 设置云台引脚为输出模式
    gpioSetPWMfrequency(yuntai_UD_pin, yuntai_UD_pwm_frequency); // 设置云台PWM频率
    gpioSetPWMrange(yuntai_UD_pin, yuntai_UD_pwm_range); // 设置云台PWM范围
    gpioPWM(yuntai_UD_pin, yuntai_UD_pwm_duty_cycle_unlock); // 设置云台PWM占空比解锁值
}

// 功能: 初始化赛道宽度查找表
void initialize_lane_widths() {
    lane_widths.assign(240, 0);

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
    std::cout << "[初始化] 赛道宽度查找表生成完毕。" << std::endl;
}

// 功能: 对输入图像进行畸变校正，返回去畸变后的图像
cv::Mat undistort(const cv::Mat &frame) 
{
    static cv::Mat mapx, mapy;
    static cv::Size cachedSize;
    static bool initialized = false;

    if (!initialized || cachedSize != frame.size())
    {
        const double k1 = 0.0439656098483248;
        const double k2 = -0.0420991522460257;
        const double p1 = 0.0;
        const double p2 = 0.0;
        const double k3 = 0.0;

        cv::Mat K = (cv::Mat_<double>(3, 3) << 176.842468665091, 0.0, 159.705914860981,
                     0.0, 176.990910857055, 120.557953465790,
                     0.0, 0.0, 1.0);

        cv::Mat D = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
        cv::initUndistortRectifyMap(K, D, cv::Mat(), K, frame.size(), CV_32FC1, mapx, mapy);
        cachedSize = frame.size();
        initialized = true;
    }

    cv::Mat undistortedFrame;
    cv::remap(frame, undistortedFrame, mapx, mapy, cv::INTER_LINEAR);
    return undistortedFrame;
}

// 功能: 提取巡线二值图（Sobel+亮度自适应+形态学）
cv::Mat ImageSobel(cv::Mat &frame) 
{
    const int min_area_threshold = 100; // 找到斑马线前

    const cv::Size targetSize(320, 240);
    cv::Mat resizedFrame;
    if (frame.size() != targetSize)
    {
        cv::resize(frame, resizedFrame, targetSize);
    }
    else
    {
        resizedFrame = frame;
    }

    const cv::Rect roiRect(1, 109, 318, 46);
    cv::Mat roi = resizedFrame(roiRect);

    cv::Mat grayRoi;
    cv::cvtColor(roi, grayRoi, cv::COLOR_BGR2GRAY);

    int kernelSize = 5;
    cv::Mat blurredRoi;
    cv::blur(grayRoi, blurredRoi, cv::Size(kernelSize, kernelSize));

    cv::Mat sobelX, sobelY;
    cv::Sobel(blurredRoi, sobelX, CV_16S, 1, 0, 3);
    cv::Sobel(blurredRoi, sobelY, CV_16S, 0, 1, 3);

    cv::Mat absSobelX, absSobelY;
    cv::convertScaleAbs(sobelX, absSobelX);
    cv::convertScaleAbs(sobelY, absSobelY);
    
    cv::Mat gradientMagnitude8U;
    cv::addWeighted(absSobelY, 1.0, absSobelX, 1.0, 0, gradientMagnitude8U);

    cv::Mat topHat;
    static cv::Mat kernel_tophat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 3));
    cv::morphologyEx(blurredRoi, topHat, cv::MORPH_TOPHAT, kernel_tophat);

    cv::Mat adaptiveMask;
    cv::threshold(topHat, adaptiveMask, 5, 255, cv::THRESH_BINARY);

    cv::Mat gradientMask;
    cv::threshold(gradientMagnitude8U, gradientMask, 30, 255, cv::THRESH_BINARY);
    static cv::Mat kernel_gradient_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(gradientMask, gradientMask, kernel_gradient_dilate);

    cv::Mat binaryMask;
    cv::bitwise_and(adaptiveMask, gradientMask, binaryMask);

    cv::medianBlur(binaryMask, binaryMask, 3);

    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binaryMask, labels, stats, centroids, 8, CV_32S);
    cv::Mat filteredByArea = cv::Mat::zeros(binaryMask.size(), CV_8U);
    for (int i = 1; i < numLabels; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_area_threshold)
        {
            filteredByArea.setTo(255, labels == i);
        }
    }

    static cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 9));
    cv::morphologyEx(filteredByArea, filteredByArea, cv::MORPH_CLOSE, kernel_close);
    cv::Mat filteredMorph = filteredByArea;

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(filteredMorph, lines, 1, CV_PI / 180, 15, 40, 25);

    cv::Mat finalImage = cv::Mat::zeros(targetSize, CV_8U);

    std::vector<cv::Vec4i> leftLines, rightLines;
    const float angle_threshold = 15.0f;
    
    for (const auto &l : lines)
    {
        double angle = std::atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
        double length = std::hypot(l[3] - l[1], l[2] - l[0]);

        if (std::abs(angle) > angle_threshold && length > 8)
        {
            double dx = l[2] - l[0];
            double dy = l[3] - l[1];
            if (std::abs(dx) < 1e-6) continue;

            double slope = dy / dx;
            if (slope < 0) {
                leftLines.push_back(l);
            } else {
                rightLines.push_back(l);
            }
        }
    }

    cv::Vec4i bestLeft = {0, 0, 0, 0}, bestRight = {0, 0, 0, 0};
    bool foundPair = false;
    double minError = 1e9;
    
    int roi_y_offset = 109;
    int check_y_local = 23; 
    int check_y_global = roi_y_offset + check_y_local;
    int ideal_width = 0;
    
    if (check_y_global >= 0 && static_cast<size_t>(check_y_global) < lane_widths.size()) {
        ideal_width = lane_widths[check_y_global];
    }

    if (ideal_width > 0) {
        for (const auto& l_left : leftLines) {
            for (const auto& l_right : rightLines) {
                double k_left = (double)(l_left[3] - l_left[1]) / (l_left[2] - l_left[0]);
                double x_left = l_left[0] + (check_y_local - l_left[1]) / k_left;
                double k_right = (double)(l_right[3] - l_right[1]) / (l_right[2] - l_right[0]);
                double x_right = l_right[0] + (check_y_local - l_right[1]) / k_right;
                double width_calc = x_right - x_left;
                
                if (width_calc > 0) {
                    minError = std::min(minError, std::abs(width_calc - ideal_width));
                }
            }
        }
    }

    if (ideal_width > 0 && minError < 1e9) {
        const double dynamic_tolerance = minError + 10.0;
        double maxTotalLength = 0;

        for (const auto& l_left : leftLines) {
            for (const auto& l_right : rightLines) {
                double k_left = (double)(l_left[3] - l_left[1]) / (l_left[2] - l_left[0]);
                double x_left = l_left[0] + (check_y_local - l_left[1]) / k_left;
                double k_right = (double)(l_right[3] - l_right[1]) / (l_right[2] - l_right[0]);
                double x_right = l_right[0] + (check_y_local - l_right[1]) / k_right;
                double width_calc = x_right - x_left;

                if (width_calc > 0) {
                    double error = std::abs(width_calc - ideal_width);
                    if (error <= dynamic_tolerance) {
                        double len_left = std::hypot(l_left[2] - l_left[0], l_left[3] - l_left[1]);
                        double len_right = std::hypot(l_right[2] - l_right[0], l_right[3] - l_right[1]);
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

    if (foundPair)
    {
        cv::Vec4i adjustedLeft = bestLeft;
        adjustedLeft[0] += roiRect.x; adjustedLeft[1] += roiRect.y;
        adjustedLeft[2] += roiRect.x; adjustedLeft[3] += roiRect.y;
        cv::line(finalImage,
                 cv::Point(adjustedLeft[0], adjustedLeft[1]),
                 cv::Point(adjustedLeft[2], adjustedLeft[3]),
                 cv::Scalar(255), 3, cv::LINE_AA);

        cv::Vec4i adjustedRight = bestRight;
        adjustedRight[0] += roiRect.x; adjustedRight[1] += roiRect.y;
        adjustedRight[2] += roiRect.x; adjustedRight[3] += roiRect.y;
        cv::line(finalImage,
                 cv::Point(adjustedRight[0], adjustedRight[1]),
                 cv::Point(adjustedRight[2], adjustedRight[3]),
                 cv::Scalar(255), 3, cv::LINE_AA);
    }
    else
    {
        for (const auto& l : leftLines) {
            cv::Vec4i adjustedLine = l;
            adjustedLine[0] += roiRect.x; adjustedLine[1] += roiRect.y;
            adjustedLine[2] += roiRect.x; adjustedLine[3] += roiRect.y;
            cv::line(finalImage, cv::Point(adjustedLine[0], adjustedLine[1]), cv::Point(adjustedLine[2], adjustedLine[3]), cv::Scalar(255), 3, cv::LINE_AA);
        }
        for (const auto& l : rightLines) {
            cv::Vec4i adjustedLine = l;
            adjustedLine[0] += roiRect.x; adjustedLine[1] += roiRect.y;
            adjustedLine[2] += roiRect.x; adjustedLine[3] += roiRect.y;
            cv::line(finalImage, cv::Point(adjustedLine[0], adjustedLine[1]), cv::Point(adjustedLine[2], adjustedLine[3]), cv::Scalar(255), 3, cv::LINE_AA);
        }
    }

    return finalImage;
}

// 功能: 基于巡线二值图逐行搜索车道左右边界并计算中线
void Tracking(cv::Mat &dilated_image) 
{
    if (dilated_image.empty() || dilated_image.type() != CV_8U) 
    {
        std::cerr << "[警告] Tracking输入图像无效，跳过本帧！" << std::endl;
        return;
    }

    if (dilated_image.rows < 154 || dilated_image.cols < 319) {
        std::cerr << "[错误] Tracking: 图像尺寸不足" << std::endl;
        return;
    }

    int begin = 160;
    if (!last_mid.empty() && last_mid.size() >= 20) 
    {
        int sum_x = 0;
        for (size_t i = 0; i < std::min((size_t)20, last_mid.size()); ++i) 
        {
            sum_x += last_mid[i].x;
        }
        begin = sum_x / std::min((size_t)20, last_mid.size());
    }

    left_line.clear();
    right_line.clear();
    mid.clear();

    for (int i = 153; i >= 110; --i) 
    {
        if (i >= dilated_image.rows) continue;
        
        int left = begin;
        int right = begin;
        bool left_found = false;
        bool right_found = false;

        while (left > 1) 
        {
            if (left + 1 >= dilated_image.cols) {
                --left;
                continue;
            }
            
            if (dilated_image.at<uchar>(i, left) == 255 &&
                dilated_image.at<uchar>(i, left + 1) == 255) 
            {
                left_found = true;
                left_line.emplace_back(left, i);
                break;
            }
            --left;
        }
        if (!left_found) 
        {
            left_line.emplace_back(1, i);
        }

        while (right < std::min(318, dilated_image.cols - 1)) 
        {
            if (right < 2) {
                ++right;
                continue;
            }
            
            if (dilated_image.at<uchar>(i, right) == 255 &&
                dilated_image.at<uchar>(i, right - 2) == 255) 
            {
                right_found = true;
                right_line.emplace_back(right, i);
                break;
            }
            ++right;
        }
        if (!right_found) 
        {
            right_line.emplace_back(std::min(318, dilated_image.cols - 1), i);
        }

        const cv::Point &left_point = left_line.back();
        const cv::Point &right_point = right_line.back();
        int mid_x = (left_point.x + right_point.x) / 2;
        mid.emplace_back(mid_x, i);

        begin = mid_x;
    }
    
    last_mid = mid;
}

// 比较两个轮廓的面积
bool Contour_Area(const vector<Point>& contour1, const vector<Point>& contour2)
{
    return contourArea(contour1) > contourArea(contour2);
}

// 定义蓝色挡板 寻找函数
void blue_card_find(void)
{   
    if (frame.empty()) {
        cerr << "[错误] blue_card_find: frame为空" << endl;
        return;
    }

    Mat change_frame;
    cvtColor(frame, change_frame, COLOR_BGR2HSV);

    Mat mask;
    Scalar scalarl = Scalar(BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN);
    Scalar scalarH = Scalar(BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX);
    inRange(change_frame, scalarl, scalarH, mask);

    cv::Rect roi_blue(BLUE_ROI_X, BLUE_ROI_Y, BLUE_ROI_WIDTH, BLUE_ROI_HEIGHT);
    
    if (roi_blue.x + roi_blue.width > mask.cols || roi_blue.y + roi_blue.height > mask.rows) {
        cerr << "[错误] blue_card_find: ROI超出边界" << endl;
        return;
    }
    
    Mat mask_roi = mask(roi_blue).clone();

    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    
    try {
        findContours(mask_roi, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    } catch (const cv::Exception& e) {
        cerr << "[错误] blue_card_find: findContours失败: " << e.what() << endl;
        return;
    }
    
    if (contours.size() > 0)
    {
        sort(contours.begin(), contours.end(), Contour_Area);
        double max_area = contourArea(contours[0]);
        cout << "蓝色检测: 最大面积=" << (int)max_area;
        
        vector<vector<Point>> newContours;
        
        for (const vector<Point> &contour : contours)
        {
            double area = contourArea(contour);
            if (area >= BLUE_AREA_VALID) 
            {
                newContours.push_back(contour);
            }
        }

        if (newContours.size() > 0)
        {
            blue_detect_count++;
            cout << " -> 有效目标，计数=" << blue_detect_count << "/" << BLUE_DETECT_THRESHOLD << endl;
            
            if (blue_detect_count >= BLUE_DETECT_THRESHOLD)
            {
                cout << ">>> 找到蓝色挡板！连续检测通过！ <<<" << endl;
                find_first = 1;
                blue_detect_count = 0;
            }
        }
        else
        {
            cout << " (无效或面积不足)" << endl;
            if (blue_detect_count > 0) 
            {
                blue_detect_count = 0;
            }
        }
    }
    else
    {
        if (blue_detect_count > 0) 
        {
            blue_detect_count = 0;
        }
    }
}

// 检测蓝色挡板是否移开
void blue_card_remove(void)
{
    if (frame.empty()) {
        cerr << "[错误] blue_card_remove: frame为空" << endl;
        return;
    }

    Mat change_frame;
    cvtColor(frame, change_frame, COLOR_BGR2HSV);

    Mat mask;
    Scalar scalarl = Scalar(BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN);
    Scalar scalarH = Scalar(BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX);
    inRange(change_frame, scalarl, scalarH, mask);

    cv::Rect roi_blue(BLUE_ROI_X, BLUE_ROI_Y, BLUE_ROI_WIDTH, BLUE_ROI_HEIGHT);
    
    if (roi_blue.x + roi_blue.width > mask.cols || roi_blue.y + roi_blue.height > mask.rows) {
        cerr << "[错误] blue_card_remove: ROI超出边界" << endl;
        return;
    }
    
    Mat mask_roi = mask(roi_blue).clone();

    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    
    try {
        findContours(mask_roi, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    } catch (const cv::Exception& e) {
        cerr << "[错误] blue_card_remove: findContours失败: " << e.what() << endl;
        return;
    }

    vector<vector<Point>> validContours;
    for (const auto &contour : contours) 
    {
        double area = contourArea(contour);
        if (area >= BLUE_REMOVE_AREA_MIN) 
        {
            validContours.push_back(contour);
        }
    }

    if (validContours.empty()) 
    {
        setCarState(CarState::StartDelay);
        start_delay_time = std::chrono::steady_clock::now();
    } 
    else 
    {
        cout << "仍检测到蓝色物体（面积：" << contourArea(validContours[0]) << "），等待移开..." << endl;
    }
}

// 功能: 常规巡线PD控制器
float servo_pd(int target) {
    if (mid.size() < 26) {
        cerr << "[警告] servo_pd: mid向量元素不足，返回中值" << endl;
        return servo_pwm_mid;
    }

    int pidx = int((mid[23].x + mid[25].x) / 2);

    float kp = 0.8;
    float kd = 3.5;

    error_first = target - pidx;

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error);

    last_error = error_first;

    servo_pwm = servo_pwm_mid + servo_pwm_diff;

    if (servo_pwm > 1000)
    {
        servo_pwm = 1000;
    }
    else if (servo_pwm < 580)
    {
        servo_pwm = 580;
    }
    
    return servo_pwm;
}

// 控制舵机电机
void motor_servo_contral()
{
    float servo_pwm_now = servo_pwm_mid;

    switch (current_state)
    {
        case CarState::Idle:
        case CarState::StartDelay:
            gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock);
            gpioPWM(servo_pin, servo_pwm_mid);
            return;

        case CarState::Cruise_High:
            servo_pwm_now = servo_pd(160);
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_CRUISE_FAST);
            break;
            
        default:
            gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock);
            gpioPWM(servo_pin, servo_pwm_mid);
            return;
    }

    gpioPWM(servo_pin, servo_pwm_now);
}

//-----------------------------------------------------------------------------------主函数-----------------------------------------------
int main(int argc, char* argv[])
{
    // 注册信号处理函数
    signal(SIGINT, signalHandler);
    
    std::cout << "========================================" << std::endl;
    std::cout << "高速巡线测试程序" << std::endl;
    std::cout << "按 Ctrl+C 结束测试并显示总时间" << std::endl;
    std::cout << "========================================\n" << std::endl;

    gpioTerminate();
    servo_motor_pwmInit();
    initialize_lane_widths();

    // 打开摄像头
    VideoCapture capture;
    capture.open(0);

    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(cv::CAP_PROP_FPS, 30);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

    if (!capture.isOpened())
    {
        cout << "无法打开摄像头，请检查设备连接！" << endl;
        return -1;
    }

    cout << "摄像头帧率: " << capture.get(cv::CAP_PROP_FPS) << endl;
    cout << "摄像头宽度: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << endl;
    cout << "摄像头高度: " << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;

    std::chrono::steady_clock::time_point last_time_display = std::chrono::steady_clock::now();

    while (capture.read(frame) && !program_finished)
    {
        if (frame.empty()) {
            cerr << "[错误] 读取到空帧，跳过处理" << endl;
            continue;
        }

        try {
            frame = undistort(frame);

            // 状态更新与预处理
            if (current_state == CarState::StartDelay) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed_sec = std::chrono::duration_cast<std::chrono::microseconds>(now - start_delay_time).count() / 1000000.0;
                if (elapsed_sec >= START_DELAY_SECONDS) {
                    cout << "[流程] 发车延时结束，开始高速巡航" << endl;
                    setCarState(CarState::Cruise_High);
                }
            }

            // 图像处理
            if (current_state == CarState::Idle) {
                if (find_first == 0) {
                    blue_card_find();
                } else {
                    blue_card_remove();
                }
            }
            else if (current_state == CarState::StartDelay || current_state == CarState::Cruise_High)
            {
                bin_image = ImageSobel(frame);
                Tracking(bin_image);
            }

            // 各状态具体逻辑处理
            switch (current_state)
            {
            case CarState::Idle:
            case CarState::StartDelay:
                break;

            case CarState::Cruise_High:
                // 每秒显示一次当前运行时间
                {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed_display = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - last_time_display).count();
                    
                    if (elapsed_display >= 1000) { // 每秒更新一次
                        auto elapsed_sec = std::chrono::duration_cast<std::chrono::microseconds>(
                            now - cruise_high_start_time).count() / 1000000.0;
                        std::cout << "\r[时间] 高速巡线已运行: " << std::fixed << std::setprecision(2) 
                                  << elapsed_sec << " 秒" << std::flush;
                        last_time_display = now;
                    }
                }
                break;
            }

            // 电机与舵机控制
            motor_servo_contral();

        } catch (const cv::Exception& e) {
            cerr << "[错误] OpenCV异常: " << e.what() << endl;
            continue;
        } catch (const std::exception& e) {
            cerr << "[错误] 标准异常: " << e.what() << endl;
            continue;
        } catch (...) {
            cerr << "[错误] 未知异常，跳过当前帧" << endl;
            continue;
        }
    }

    // 清理资源
    cout << "\n[清理] 释放资源..." << endl;
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock);
    gpioPWM(servo_pin, servo_pwm_mid);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    gpioTerminate();
    cout << "[清理] 系统退出完成" << endl;
    
    return 0;
}

