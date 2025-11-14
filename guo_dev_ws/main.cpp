#include <iostream> // 标准输入输出流库
#include <cstdlib> // 标准库
#include <unistd.h> // Unix标准库

#include <opencv2/opencv.hpp> // OpenCV主头文件
#include <opencv2/core/core.hpp> // OpenCV核心功能
#include <opencv2/highgui.hpp> // OpenCV高层GUI功能
#include <opencv2/imgproc/imgproc_c.h> // OpenCV图像处理功能

#include <string> // 字符串库
#include <pigpio.h> // GPIO控制库
#include <thread> // 线程库
#include <vector> // 向量容器库
#include <chrono> // 时间库
#include <iomanip> // 格式化输出

#include "fastestdet.hpp" // FastestDet库

using namespace std; // 使用标准命名空间
using namespace cv; // 使用OpenCV命名空间

//------------速度参数配置------------------------------------------------------------------------------------------
const int MOTOR_SPEED_DELTA_CRUISE = 1300; // 常规巡航速度增量
const int MOTOR_SPEED_DELTA_AVOID = 1300;  // 避障阶段速度增量
const int MOTOR_SPEED_DELTA_PARK = 1300;   // 车库阶段速度增量

//---------------调试选项-------------------------------------------------
const bool SHOW_SOBEL_DEBUG = false; // 是否显示Sobel调试窗口
const int SOBEL_DEBUG_REFRESH_INTERVAL_MS = 120; // 调试窗口刷新间隔，减轻imshow开销

//---------------性能统计---------------------------------------------------
int number = 0; // 已处理帧计数
bool SHOW_FPS = false; // 是否显示FPS信息，可通过命令行参数控制

//------------有关的全局变量定义------------------------------------------------------------------------------------------

std::vector<DetectObject> result; // 存储FastestDet检测结果
std::vector<DetectObject> result_ab; // 存储FastestDet检测结果

// 模型路径配置
std::string model_param_obs = "models/obs.param";
std::string model_bin_obs = "models/obs.bin";
int num_classes_obs = 1;
std::vector<std::string> labels_obs{"zhuitong"};

std::string model_param_lr = "models/lr.param";
std::string model_bin_lr = "models/lr.bin";
int num_classes_lr = 2;
std::vector<std::string> labels_lr{"left", "right"};

std::string model_param_ab = "models/ab.param";
std::string model_bin_ab = "models/ab.bin";
int num_classes_ab = 2;
std::vector<std::string> labels_ab{"A", "B"};

// 模型指针（延迟初始化，避免全局构造函数问题）
FastestDet* fastestdet_obs = nullptr;
FastestDet* fastestdet_lr = nullptr;
FastestDet* fastestdet_ab = nullptr;

//-----------------图像相关----------------------------------------------
Mat frame; // 存储视频帧
Mat bin_image; // 存储二值化图像--Sobel检测后图像

// 图像处理参数
const int MIN_COMPONENT_AREA = 400; // 连通区域最小面积阈值（用于过滤噪声）

//-----------------巡线相关-----------------------------------------------
std::vector<cv::Point> mid; // 存储中线
std::vector<cv::Point> left_line; // 存储左线条
std::vector<cv::Point> right_line; // 存储右线条

//---------------舵机和电机相关---------------------------------------------
int error_first; // 存储第一次误差
int last_error; // 存储上一次误差
float servo_pwm_diff; // 存储舵机PWM差值
float servo_pwm; // 存储舵机PWM值

//---------------发车信号定义-----------------------------------------------
int find_first = 0; // 标记是否第一次找到蓝色挡板
int fache_sign = 0; // 标记发车信号

//---------------斑马线相关-------------------------------------------------
int banma = 0; // 斑马线检测结果

//----------------变道相关---------------------------------------------------
int changeroad = 0; // 变道检测结果 (0=未识别, 1=左转, 2=右转)
bool has_detected_turn_sign = false; // 是否已成功识别到转向标识

//----------------避障相关---------------------------------------------------
int bz_heighest = 0; // 避障高度
int bz_xcenter = 0; // 存储避障中心点
int bz_get = 0;
std::vector<cv::Point> mid_bz; // 存储中线
std::vector<cv::Point> left_line_bz; // 存储左线条
std::vector<cv::Point> right_line_bz; // 存储右线条
std::vector<cv::Point> last_mid_bz; // 存储上一帧避障中线
bool is_in_avoidance = false; // 是否处于避障状态锁
int last_known_bz_xcenter = 0; // 最后一次检测到的障碍物位置
int last_known_bz_bottom = 0;
int last_known_bz_heighest = 0;
int count_bz = 0; // 避障计数器
int bz_disappear_count = 0; // 障碍物连续消失计数器
const int BZ_DISAPPEAR_THRESHOLD = 5; // 确认障碍物消失的帧数阈值
int bz_y2 = 170; // 可见障碍物底部阈值

//----------------停车相关---------------------------------------------------
int park_mid = 160; // 停车车库中线检测结果
int flag_gohead = 0; // 前进标志
int flag_turn_done = 0; // 转向完成标志
std::chrono::steady_clock::time_point zebra_stop_start_time;
bool is_stopping_at_zebra = false;
bool is_parking_phase = false; // 是否进入寻找车库阶段
int latest_park_id = 0; // 最近检测到的车库ID (1=A, 2=B)
const int PARKING_Y_THRESHOLD = 200; // 触发入库的Y轴阈值

// 定义舵机和电机引脚号、PWM范围、PWM频率、PWM占空比解锁值
const int servo_pin = 12; // 存储舵机引脚号
const float servo_pwm_range = 10000.0; // 存储舵机PWM范围
const float servo_pwm_frequency = 50.0; // 存储舵机PWM频率
const float servo_pwm_duty_cycle_unlock = 730.0; // 存储舵机PWM占空比解锁值

//---------------------------------------------------------------------------------------------------
float servo_pwm_mid = servo_pwm_duty_cycle_unlock; // 存储舵机中值
//---------------------------------------------------------------------------------------------------

const int motor_pin = 13; // 存储电机引脚号
const float motor_pwm_range = 40000.0; // 存储电机PWM范围
const float motor_pwm_frequency = 200.0; // 存储电机PWM频率
const float motor_pwm_duty_cycle_unlock = 11400.0; // 存储电机PWM占空比解锁值

//--------------------------------------------------------------------------------------------------
float motor_pwm_mid = motor_pwm_duty_cycle_unlock; // 存储电机PWM初始化值
//--------------------------------------------------------------------------------------------------

const int yuntai_LR_pin = 22; // 存储云台引脚号
const float yuntai_LR_pwm_range = 1000.0; // 存储云台PWM范围
const float yuntai_LR_pwm_frequency = 50.0; // 存储云台PWM频率
const float yuntai_LR_pwm_duty_cycle_unlock = 63.0; //大左小右 

const int yuntai_UD_pin = 23; // 存储云台引脚号
const float yuntai_UD_pwm_range = 1000.0; // 存储云台PWM范围
const float yuntai_UD_pwm_frequency = 50.0; // 存储云台PWM频率
const float yuntai_UD_pwm_duty_cycle_unlock = 58.0; //大上下小

//---------------平滑滤波相关-------------------------------------------------
std::vector<cv::Point> last_mid; // 存储上一次的中线，用于平滑滤波
int blue_detect_count = 0; // 蓝色挡板连续检测计数
const int BLUE_DETECT_THRESHOLD = 10; // 需要连续检测到的帧数才能确认找到蓝色挡板

//---------------蓝色检测参数------------------------------------------
// HSV颜色范围
const int BLUE_H_MIN = 100;  // 色调H最小值
const int BLUE_H_MAX = 130;  // 色调H最大值
const int BLUE_S_MIN = 50;   // 饱和度S最小值
const int BLUE_S_MAX = 255;  // 饱和度S最大值
const int BLUE_V_MIN = 50;   // 亮度V最小值
const int BLUE_V_MAX = 255;  // 亮度V最大值

// 蓝色检测ROI区域（限制检测范围）
const int BLUE_ROI_X = 50;      // ROI左上角X坐标
const int BLUE_ROI_Y = 80;      // ROI左上角Y坐标
const int BLUE_ROI_WIDTH = 220;  // ROI宽度
const int BLUE_ROI_HEIGHT = 100; // ROI高度

// 蓝色面积阈值
const double BLUE_AREA_VALID = 2000.0; // 有效面积阈值

// 蓝色挡板移开检测参数
const double BLUE_REMOVE_AREA_MIN = 500.0; // 移开检测的最小面积阈值（过滤小噪点）

//---------------斑马线检测参数（可调节）------------------------------------------
// HSV白色范围
const int BANMA_WHITE_H_MIN = 0;    // 色调H最小值
const int BANMA_WHITE_H_MAX = 180;  // 色调H最大值
const int BANMA_WHITE_S_MIN = 0;    // 饱和度S最小值
const int BANMA_WHITE_S_MAX = 30;   // 饱和度S最大值
const int BANMA_WHITE_V_MIN = 200;  // 亮度V最小值（高亮度白色）
const int BANMA_WHITE_V_MAX = 255;  // 亮度V最大值

// 斑马线检测ROI区域
const int BANMA_ROI_X = 2;           // ROI左上角X坐标
const int BANMA_ROI_Y = 50;         // ROI左上角Y坐标
const int BANMA_ROI_WIDTH = 318;     // ROI宽度
const int BANMA_ROI_HEIGHT = 250;    // ROI高度

// 斑马线矩形筛选尺寸（斑马线由多个白色矩形组成）
const int BANMA_RECT_MIN_WIDTH = 5;   // 矩形最小宽度
const int BANMA_RECT_MAX_WIDTH = 100;   // 矩形最大宽度
const int BANMA_RECT_MIN_HEIGHT = 5;  // 矩形最小高度
const int BANMA_RECT_MAX_HEIGHT = 100;  // 矩形最大高度

// 斑马线判定阈值
const int BANMA_MIN_COUNT = 6;  // 判定为斑马线需要的最少白色矩形数量

// 形态学处理参数
const int BANMA_MORPH_KERNEL_SIZE = 3;  // 形态学处理kernel大小（3x3）


//--------------------------------------------------------------------------

// 功能: 初始化舵机、电机与云台PWM，完成GPIO库初始化
void servo_motor_pwmInit(void) 
{
    if (gpioInitialise() < 0) // 初始化GPIO，如果失败则返回
    {
        std::cout << "GPIO初始化失败！请使用sudo权限运行！" << std::endl; // 输出失败信息
        return; // 返回
    }
    else
        std::cout << "GPIO初始化成功，系统正常！" << std::endl; // 输出成功信息

    gpioSetMode(servo_pin, PI_OUTPUT); // 设置舵机引脚为输出模式
    gpioSetPWMfrequency(servo_pin, servo_pwm_frequency); // 设置舵机PWM频率
    gpioSetPWMrange(servo_pin, servo_pwm_range); // 设置舵机PWM范围
    gpioPWM(servo_pin, servo_pwm_duty_cycle_unlock); // 设置舵机PWM占空比解锁值

    gpioSetMode(motor_pin, PI_OUTPUT); // 设置电机引脚为输出模式
    gpioSetPWMfrequency(motor_pin, motor_pwm_frequency); // 设置电机PWM频率
    gpioSetPWMrange(motor_pin, motor_pwm_range); // 设置电机PWM范围
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock); // 设置电机PWM占空比解锁值

    gpioSetMode(yuntai_LR_pin, PI_OUTPUT); // 设置云台引脚为输出模式
    gpioSetPWMfrequency(yuntai_LR_pin, yuntai_LR_pwm_frequency); // 设置云台PWM频率
    gpioSetPWMrange(yuntai_LR_pin, yuntai_LR_pwm_range); // 设置云台PWM范围
    gpioPWM(yuntai_LR_pin, yuntai_LR_pwm_duty_cycle_unlock); // 设置云台PWM占空比解锁值

    gpioSetMode(yuntai_UD_pin, PI_OUTPUT); // 设置云台引脚为输出模式
    gpioSetPWMfrequency(yuntai_UD_pin, yuntai_UD_pwm_frequency); // 设置云台PWM频率
    gpioSetPWMrange(yuntai_UD_pin, yuntai_UD_pwm_range); // 设置云台PWM范围
    gpioPWM(yuntai_UD_pin, yuntai_UD_pwm_duty_cycle_unlock); // 设置云台PWM占空比解锁值

}

//------------------------------------------------------------------------------------------------------------
// 功能: 对输入图像进行畸变校正，返回去畸变后的图像
cv::Mat undistort(const cv::Mat &frame) 
{
    static cv::Mat mapx, mapy; // 映射矩阵
    static cv::Size cachedSize;
    static bool initialized = false;

    if (!initialized || cachedSize != frame.size())
    {
        const double k1 = 0.0439656098483248; // 畸变系数k1
        const double k2 = -0.0420991522460257; // 畸变系数k2
        const double p1 = 0.0; // 畸变系数p1
        const double p2 = 0.0; // 畸变系数p2
        const double k3 = 0.0; // 畸变系数k3

        // 相机内参矩阵
        cv::Mat K = (cv::Mat_<double>(3, 3) << 176.842468665091, 0.0, 159.705914860981,
                     0.0, 176.990910857055, 120.557953465790,
                     0.0, 0.0, 1.0);

        // 畸变系数矩阵
        cv::Mat D = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
        cv::initUndistortRectifyMap(K, D, cv::Mat(), K, frame.size(), CV_32FC1, mapx, mapy);
        cachedSize = frame.size();
        initialized = true;
    }

    cv::Mat undistortedFrame; // 去畸变后的图像帧

    // 应用映射，得到去畸变后的图像
    cv::remap(frame, undistortedFrame, mapx, mapy, cv::INTER_LINEAR);

    return undistortedFrame; // 返回去畸变后的图像
}

 

// 功能: 在二值图像上从起点到终点绘制指定宽度的白线（用于补线）
cv::Mat drawWhiteLine(cv::Mat binaryImage, cv::Point start, cv::Point end, int lineWidth)
{
    cv::Mat resultImage = binaryImage.clone(); // 克隆输入的二值图像

    int x1 = start.x, y1 = start.y; // 获取起点的x和y坐标
    int x2 = end.x, y2 = end.y; // 获取终点的x和y坐标

    if (x1 == x2) // 如果起点和终点的x坐标相同，说明是垂直线
    {
        for (int y = std::min(y1, y2); y <= std::max(y1, y2); ++y) // 遍历y坐标
        {
            for (int i = -lineWidth / 2; i <= lineWidth / 2; ++i) // 遍历线宽
            {
                resultImage.at<uchar>(cv::Point(x1 + i, y)) = 255; // 将线宽范围内的像素值设为255（白色）
            }
        }
    }
    else // 如果起点和终点的x坐标不同，说明是斜线
    {
        double slope = static_cast<double>(y2 - y1) / (x2 - x1); // 计算斜率
        double intercept = y1 - slope * x1; // 计算截距

        for (int x = std::min(x1, x2); x <= std::max(x1, x2); ++x) // 遍历x坐标
        {
            int y = static_cast<int>(slope * x + intercept); // 计算对应的y坐标
            for (int i = -lineWidth / 2; i <= lineWidth / 2; ++i) // 遍历线宽
            {
                int newY = std::max(0, std::min(y + i, resultImage.rows - 1)); // 确保y坐标在图像范围内
                resultImage.at<uchar>(cv::Point(x, newY)) = 255; // 将线宽范围内的像素值设为255（白色）
            }
        }
    }

    return resultImage; // 返回绘制了白线的图像
}

// 功能: 提取巡线二值图（Sobel+亮度自适应+形态学），可选输出调试覆盖图
cv::Mat ImageSobel(cv::Mat &frame, cv::Mat *debugOverlay = nullptr) 
{
    const cv::Size targetSize(320, 240);
    cv::Mat resizedFrame;
    if (frame.size() != targetSize)
    {
        cv::resize(frame, resizedFrame, targetSize);
    }
    else
    {
        resizedFrame = frame.clone();
    }

    const cv::Rect roiRect(1, 109, 318, 46); // 巡线ROI区域
    cv::Mat roi = resizedFrame(roiRect); // 直接使用ROI视图

    cv::Mat grayRoi;
    cv::cvtColor(roi, grayRoi, cv::COLOR_BGR2GRAY); // ROI灰度化

    int kernelSize = 5;
    cv::Mat blurredRoi;
    cv::blur(grayRoi, blurredRoi, cv::Size(kernelSize, kernelSize)); // ROI均值滤波降噪

    cv::Mat sobelX, sobelY;
    cv::Sobel(blurredRoi, sobelX, CV_64F, 1, 0, 3); // X方向梯度
    cv::Sobel(blurredRoi, sobelY, CV_64F, 0, 1, 3); // Y方向梯度
    cv::Mat gradientMagnitude = cv::abs(sobelY) + 0.5 * cv::abs(sobelX); // 组合梯度更偏向纵向
    cv::Mat gradientMagnitude8U;
    cv::convertScaleAbs(gradientMagnitude, gradientMagnitude8U); // 转为8位方便阈值

    cv::Mat hsvRoi;
    cv::cvtColor(roi, hsvRoi, cv::COLOR_BGR2HSV); // ROI HSV分离亮度信息
    cv::Mat vChannel;
    cv::extractChannel(hsvRoi, vChannel, 2); // 仅提取V通道

    cv::Mat claheOutput;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(4, 4)); // 自适应直方图均衡
    clahe->apply(vChannel, claheOutput);                       // 仅对V通道增强
    cv::GaussianBlur(claheOutput, claheOutput, cv::Size(5, 5), 0);   // 平滑提升稳定性

    cv::Mat adaptiveMask;
    cv::adaptiveThreshold(claheOutput, adaptiveMask, 255,
                          cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                          31, -10); // 自适应阈值提取亮线

    cv::Mat gradientMask;
    cv::threshold(gradientMagnitude8U, gradientMask, 30, 255, cv::THRESH_BINARY); // 梯度二值掩码
    cv::Mat gradientKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(gradientMask, gradientMask, gradientKernel);

    cv::Mat binaryMask;
    cv::bitwise_and(adaptiveMask, gradientMask, binaryMask); // 亮度+梯度联合约束

    cv::medianBlur(binaryMask, binaryMask, 3); // 中值去椒盐噪声
    cv::Mat noiseKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_OPEN, noiseKernel); // 小结构开运算

    cv::Mat morphImage = binaryMask.clone();
    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 5)); // 闭运算连接断裂
    cv::morphologyEx(morphImage, morphImage, cv::MORPH_CLOSE, kernelClose);
    cv::Mat kernelDilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); // 膨胀加粗车道线
    cv::dilate(morphImage, morphImage, kernelDilate, cv::Point(-1, -1), 1);

    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(morphImage, labels, stats, centroids, 8, CV_32S); // 连通域分析
    cv::Mat filteredMorph = cv::Mat::zeros(morphImage.size(), CV_8U);
    for (int i = 1; i < numLabels; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= MIN_COMPONENT_AREA)
        {
            filteredMorph.setTo(255, labels == i);
        }
    }
    morphImage = filteredMorph;

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(morphImage, lines, 1, CV_PI / 180, 20, 15, 8);

    cv::Mat finalImage = cv::Mat::zeros(targetSize, CV_8U);
    cv::Mat overlayImage;
    if (debugOverlay)
    {
        overlayImage = resizedFrame.clone();
        cv::rectangle(overlayImage, roiRect, cv::Scalar(0, 255, 0), 1);
    }

    for (const auto &l : lines)
    {
        double angle = std::atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
        double length = std::hypot(l[3] - l[1], l[2] - l[0]);

        if (std::abs(angle) > 15 && length > 8)
        {
            cv::Vec4i adjustedLine = l;
            adjustedLine[0] += roiRect.x;
            adjustedLine[1] += roiRect.y;
            adjustedLine[2] += roiRect.x;
            adjustedLine[3] += roiRect.y;

            cv::line(finalImage,
                     cv::Point(adjustedLine[0], adjustedLine[1]),
                     cv::Point(adjustedLine[2], adjustedLine[3]),
                     cv::Scalar(255), 3, cv::LINE_AA);
            if (debugOverlay)
            {
                cv::line(overlayImage,
                         cv::Point(adjustedLine[0], adjustedLine[1]),
                         cv::Point(adjustedLine[2], adjustedLine[3]),
                         cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
        }
    }

    if (debugOverlay)
    {
        *debugOverlay = overlayImage;
    }

    return finalImage;
}

// 功能: 基于巡线二值图逐行搜索车道左右边界并计算中线
void Tracking(cv::Mat &dilated_image) 
{
    // 参数检查
    if (dilated_image.empty() || dilated_image.type() != CV_8U) 
    {
        std::cerr << "[警告] Tracking输入图像无效，跳过本帧！" << std::endl;
        return;
    }

    // 检查图像尺寸
    if (dilated_image.rows < 154 || dilated_image.cols < 319) {
        std::cerr << "[错误] Tracking: 图像尺寸不足 (" << dilated_image.cols << "x" << dilated_image.rows << ")，需要至少 319x154" << std::endl;
        return;
    }

    // 如果上一次有有效数据，使用上一次的中点作为起始点，否则使用默认值
    int begin = 160; // 初始化起始位置
    if (!last_mid.empty() && last_mid.size() >= 20) 
    {
        // 使用上一次中线的平均值作为起始搜索点，提高稳定性
        int sum_x = 0;
        for (size_t i = 0; i < std::min((size_t)20, last_mid.size()); ++i) 
        {
            sum_x += last_mid[i].x;
        }
        begin = sum_x / std::min((size_t)20, last_mid.size());
    }

    left_line.clear(); // 清空左线条
    right_line.clear(); // 清空右线条
    mid.clear(); // 清空中线

    // 逐行搜索，从第153行到第110行
    for (int i = 153; i >= 110; --i) 
    {
        // 确保行索引在有效范围内
        if (i >= dilated_image.rows) continue;
        
        int left = begin;  // 左侧搜索起点
        int right = begin; // 右侧搜索起点
        bool left_found = false; // 标记是否找到左线
        bool right_found = false; // 标记是否找到右线

        // 搜索左线
        while (left > 1) 
        {
            // 边界检查
            if (left + 1 >= dilated_image.cols) {
                --left;
                continue;
            }
            
            if (dilated_image.at<uchar>(i, left) == 255 &&
                dilated_image.at<uchar>(i, left + 1) == 255) 
            {
                left_found = true;
                left_line.emplace_back(left, i); // 记录左线点
                break;
            }
            --left;
        }
        if (!left_found) 
        {
            left_line.emplace_back(1, i); // 左线未找到，默认记录最左侧点
        }

        // 搜索右线
        while (right < std::min(318, dilated_image.cols - 1)) 
        {
            // 边界检查
            if (right < 2) {
                ++right;
                continue;
            }
            
            if (dilated_image.at<uchar>(i, right) == 255 &&
                dilated_image.at<uchar>(i, right - 2) == 255) 
            {
                right_found = true;
                right_line.emplace_back(right, i); // 记录右线点
                break;
            }
            ++right;
        }
        if (!right_found) 
        {
            right_line.emplace_back(std::min(318, dilated_image.cols - 1), i); // 右线未找到，默认记录最右侧点
        }

        // 计算中点
        const cv::Point &left_point = left_line.back();
        const cv::Point &right_point = right_line.back();
        int mid_x = (left_point.x + right_point.x) / 2;
        mid.emplace_back(mid_x, i); // 记录中点

        // 更新下一行的搜索起点
        begin = mid_x;
    }
    
    // 保存当前中线数据供下一帧使用
    last_mid = mid;
}

// 功能: 避障模式下的巡线追踪，使用历史障碍物高度作为搜索上界
void Tracking_bz(cv::Mat &dilated_image) 
{
    if (dilated_image.empty() || dilated_image.type() != CV_8U)
    {
        std::cerr << "[警告] Tracking_bz输入图像无效，跳过本帧！" << std::endl;
        return;
    }

    int begin = 160; // 初始化起始位置
    if (!last_mid_bz.empty())
    {
        int sum_x = 0;
        const size_t sample_count = std::min(static_cast<size_t>(20), last_mid_bz.size());
        for (size_t i = 0; i < sample_count; ++i)
        {
            sum_x += last_mid_bz[i].x;
        }
        begin = sum_x / sample_count;
    }

    left_line_bz.clear(); // 清空避障左线条
    right_line_bz.clear(); // 清空避障右线条
    mid_bz.clear(); // 清空避障中线

    int lower_bound = bz_heighest;
    if (lower_bound < 0 || lower_bound > 153)
    {
        lower_bound = 110; // 回落到安全的巡线搜索下限
    }

    for (int i = 153; i >= lower_bound; --i)
    {
        int left = begin;
        int right = begin;
        bool left_found = false;
        bool right_found = false;

        while (left > 1)
        {
            if (dilated_image.at<uchar>(i, left) == 255 &&
                dilated_image.at<uchar>(i, left + 1) == 255)
            {
                left_found = true;
                left_line_bz.emplace_back(left, i);
                break;
            }
            --left;
        }
        if (!left_found)
        {
            left_line_bz.emplace_back(1, i);
        }

        while (right < 318)
        {
            if (dilated_image.at<uchar>(i, right) == 255 &&
                dilated_image.at<uchar>(i, right - 2) == 255)
            {
                right_found = true;
                right_line_bz.emplace_back(right, i);
                break;
            }
            ++right;
        }
        if (!right_found)
        {
            right_line_bz.emplace_back(318, i);
        }

        const cv::Point &left_point = left_line_bz.back();
        const cv::Point &right_point = right_line_bz.back();
        int mid_x = (left_point.x + right_point.x) / 2;
        mid_bz.emplace_back(mid_x, i);

        begin = mid_x;
    }

    last_mid_bz = mid_bz;
}

// 比较两个轮廓的面积
// 功能: 轮廓面积比较（用于排序，返回面积更大的在前）
bool Contour_Area(const vector<Point>& contour1, const vector<Point>& contour2)
{
    return contourArea(contour1) > contourArea(contour2); // 返回轮廓1是否大于轮廓2
}

// 定义蓝色挡板 寻找函数
// 功能: 在限定ROI内通过HSV阈值查找蓝色挡板，带连续帧计数确认
void blue_card_find(void)  // 输入为mask图像
{   
    if (frame.empty()) {
        cerr << "[错误] blue_card_find: frame为空" << endl;
        return;
    }

    Mat change_frame; // 存储颜色空间转换后的图像
    cvtColor(frame, change_frame, COLOR_BGR2HSV); // 转换颜色空间

    Mat mask; // 存储掩码图像

    // 定义HSV范围 hsv颜色空间特点：色调H、饱和度S、亮度V
    Scalar scalarl = Scalar(BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN); // HSV的低值
    Scalar scalarH = Scalar(BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX);  // HSV的高值
    inRange(change_frame, scalarl, scalarH, mask); // 创建掩码

    // 限制检测区域到画面中央区域，减少边缘干扰
    cv::Rect roi_blue(BLUE_ROI_X, BLUE_ROI_Y, BLUE_ROI_WIDTH, BLUE_ROI_HEIGHT);
    
    // 边界检查
    if (roi_blue.x + roi_blue.width > mask.cols || roi_blue.y + roi_blue.height > mask.rows) {
        cerr << "[错误] blue_card_find: ROI超出边界" << endl;
        return;
    }
    
    Mat mask_roi = mask(roi_blue).clone(); // 克隆以避免findContours修改原图

    vector<vector<Point>> contours; // 存储轮廓的向量
    vector<Vec4i> hierarcy; // 存储层次结构的向量
    
    try {
        findContours(mask_roi, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE); // 查找轮廓
    } catch (const cv::Exception& e) {
        cerr << "[错误] blue_card_find: findContours失败: " << e.what() << endl;
        return;
    }
    
    if (contours.size() > 0) // 如果找到轮廓
    {
        sort(contours.begin(), contours.end(), Contour_Area); // 按轮廓面积排序
        double max_area = contourArea(contours[0]);
        cout << "蓝色检测: 最大面积=" << (int)max_area;
        
        vector<vector<Point>> newContours; // 存储新的轮廓向量（满足所有条件的）
        
        for (const vector<Point> &contour : contours) // 遍历每个轮廓
        {
            double area = contourArea(contour);
            // 只保留面积 >= BLUE_AREA_VALID 的轮廓
            if (area >= BLUE_AREA_VALID) 
            {
                newContours.push_back(contour);
            }
        }

        // 连续检测计数机制：只有连续多帧都检测到才确认
        if (newContours.size() > 0)
        {
            blue_detect_count++; // 增加计数
            cout << " -> 有效目标，计数=" << blue_detect_count << "/" << BLUE_DETECT_THRESHOLD << endl;
            
            if (blue_detect_count >= BLUE_DETECT_THRESHOLD) // 连续检测到足够帧数
            {
                cout << ">>> 找到蓝色挡板！连续检测通过！ <<<" << endl;
                find_first = 1; // 更新标志位
                blue_detect_count = 0; // 重置计数
            }
        }
        else
        {
            // 如果没有检测到或面积不够，重置计数
            cout << " (无效或面积不足)" << endl;
            if (blue_detect_count > 0) 
            {
                blue_detect_count = 0;
            }
        }
    }
    else
    {
        // 如果没找到轮廓，重置计数
        if (blue_detect_count > 0) 
        {
            blue_detect_count = 0;
        }
    }
}

// 检测蓝色挡板是否移开
// 功能: 检测蓝色挡板是否移开（无有效轮廓即视为移开），触发发车
void blue_card_remove(void) // 输入为mask图像
{
    cout << "进入 蓝色挡板移开 进程！" << endl; // 输出进入移除蓝色挡板的过程

    if (frame.empty()) {
        cerr << "[错误] blue_card_remove: frame为空" << endl;
        return;
    }

    Mat change_frame; // 存储颜色空间转换后的图像
    cvtColor(frame, change_frame, COLOR_BGR2HSV); // 转换颜色空间

    Mat mask; // 存储掩码图像

    // 定义HSV范围 hsv颜色空间特点：色调H、饱和度S、亮度V（使用与blue_card_find相同的参数）
    Scalar scalarl = Scalar(BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN); // HSV的低值
    Scalar scalarH = Scalar(BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX);  // HSV的高值 
    inRange(change_frame, scalarl, scalarH, mask); // 创建掩码

    // 使用与blue_card_find相同的ROI区域进行裁剪
    cv::Rect roi_blue(BLUE_ROI_X, BLUE_ROI_Y, BLUE_ROI_WIDTH, BLUE_ROI_HEIGHT);
    
    // 边界检查
    if (roi_blue.x + roi_blue.width > mask.cols || roi_blue.y + roi_blue.height > mask.rows) {
        cerr << "[错误] blue_card_remove: ROI超出边界" << endl;
        return;
    }
    
    Mat mask_roi = mask(roi_blue).clone(); // 克隆以避免findContours修改原图

    vector<vector<Point>> contours; // 定义轮廓向量
    vector<Vec4i> hierarcy; // 定义层次结构向量
    
    try {
        findContours(mask_roi, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE); // 查找轮廓
    } catch (const cv::Exception& e) {
        cerr << "[错误] blue_card_remove: findContours失败: " << e.what() << endl;
        return;
    }

    // 过滤出"有效蓝色轮廓"（只检查面积，位置已由ROI限制）
    vector<vector<Point>> validContours;
    for (const auto &contour : contours) 
    {
        // 过滤面积过小的干扰
        double area = contourArea(contour);
        if (area >= BLUE_REMOVE_AREA_MIN) 
        {
            validContours.push_back(contour);
        }
    }

    // 判断是否存在"有效蓝色轮廓"：若不存在，说明挡板已移开
    if (validContours.empty()) 
    {
        fache_sign = 1;
        cout << "蓝色挡板已移开，开始巡线！" << endl;
        usleep(500000);  
    } 
    else 
    {
        cout << "仍检测到蓝色物体（面积：" << contourArea(validContours[0]) << "），等待移开..." << endl;
    }
}

// 功能: 先裁剪斑马线ROI，再在ROI内做白色提取与形态学，统计矩形条纹数量
int banma_get(cv::Mat &frame) {
    // 先裁剪感兴趣区域，减少后续处理数据量
    int roiWidth = std::min(BANMA_ROI_WIDTH, frame.cols - BANMA_ROI_X);
    int roiHeight = std::min(BANMA_ROI_HEIGHT, frame.rows - BANMA_ROI_Y);
    if (roiWidth <= 0 || roiHeight <= 0) {
        return 0;
    }
    cv::Rect roi(BANMA_ROI_X, BANMA_ROI_Y, roiWidth, roiHeight);
    cv::Mat roiFrame = frame(roi);

    // 将ROI图像转换为HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(roiFrame, hsv, cv::COLOR_BGR2HSV);

    // 定义白色的下界和上界（使用常量）
    cv::Scalar lower_white(BANMA_WHITE_H_MIN, BANMA_WHITE_S_MIN, BANMA_WHITE_V_MIN);
    cv::Scalar upper_white(BANMA_WHITE_H_MAX, BANMA_WHITE_S_MAX, BANMA_WHITE_V_MAX);

    // 创建白色掩码
    cv::Mat mask1;
    cv::inRange(hsv, lower_white, upper_white, mask1);

    // 创建形态学处理的结构元素（使用常量）
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(BANMA_MORPH_KERNEL_SIZE, BANMA_MORPH_KERNEL_SIZE));
    // 对掩码进行膨胀和腐蚀操作
    cv::dilate(mask1, mask1, kernel);
    cv::erode(mask1, mask1, kernel);

    // 查找图像中的轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask1, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 创建一个副本以便绘制轮廓
    cv::Mat contour_img = mask1.clone();

    int count_BMX = 0;  // 斑马线计数器

    // 遍历每个找到的轮廓
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);  // 获取当前轮廓的外接矩形 rect

        // 筛选符合尺寸的矩形（使用常量）
        if (BANMA_RECT_MIN_HEIGHT <= rect.height && rect.height < BANMA_RECT_MAX_HEIGHT &&
            BANMA_RECT_MIN_WIDTH <= rect.width && rect.width < BANMA_RECT_MAX_WIDTH) {
            // 过滤赛道外的轮廓
            cv::rectangle(contour_img, rect, cv::Scalar(255), 2);
            count_BMX++;
        }
    }

    // 最终返回值（使用常量）
    if (count_BMX >= BANMA_MIN_COUNT) {
        cout << "检测到斑马线（白色矩形数量：" << count_BMX << "）" << endl;
        return 1;
    }
    else {
        return 0;
    }
}



// 功能: 常规巡线PD控制器，基于中线偏差计算舵机PWM
float servo_pd(int target) { // 赛道巡线控制

    // 安全检查：确保mid向量有足够的元素
    if (mid.size() < 26) {
        cerr << "[警告] servo_pd: mid向量元素不足 (" << mid.size() << " < 26)，返回中值" << endl;
        return servo_pwm_mid;
    }

    int pidx = int((mid[23].x + mid[25].x) / 2); // 计算中线中点的x坐标

    float kp = 1.0; // 比例系数
    float kd = 2.0; // 微分系数

    error_first = target - pidx; // 计算误差

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); // 计算舵机PWM差值

    last_error = error_first; // 更新上一次误差

    servo_pwm = servo_pwm_mid + servo_pwm_diff; // 计算舵机PWM值

    if (servo_pwm > 1000) // 如果PWM值大于900
    {
        servo_pwm = 1000; // 限制PWM值为900
    }
    else if (servo_pwm < 580) // 如果PWM值小于600
    {
        servo_pwm = 580; // 限制PWM值为600
    }
    return servo_pwm; // 返回舵机PWM值
}

// 功能: 避障巡线PD控制器，权重更大，响应更快
float servo_pd_bz(int target) { // 避障巡线控制

    // 安全检查：确保mid_bz向量不为空
    if (mid_bz.empty()) {
        cerr << "[警告] servo_pd_bz: mid_bz向量为空，返回中值" << endl;
        return servo_pwm_mid;
    }

    int pidx = mid_bz[(int)(mid_bz.size() / 2)].x;

    // float kp = 1.5; // 比例系数
    float kp = 2.0; // 比例系数
    float kd = 5.0; // 微分系数

    error_first = target - pidx; // 计算误差

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); // 计算舵机PWM差值
    last_error = error_first; // 更新上一次误差

    servo_pwm = servo_pwm_mid + servo_pwm_diff; // 计算舵机PWM值
    if (servo_pwm > 1000) // 如果PWM值大于900
    {
        servo_pwm = 1000; // 限制PWM值为900
    }
    else if (servo_pwm < 600) // 如果PWM值小于600
    {
        servo_pwm = 600; // 限制PWM值为600
    }
    return servo_pwm; // 返回舵机PWM值
}

// 功能: 停车阶段的PD控制器，基于车库检测中点
float servo_pd_AB(int target) { // 避障巡线控制

    int pidx = park_mid; // 计算中点的x坐标

    float kp = 3.0; // 比例系数
    float kd = 3.0; // 微分系数

    error_first = target - pidx; // 计算误差

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); // 计算舵机PWM差值
    last_error = error_first; // 更新上一次误差

    servo_pwm = servo_pwm_mid + servo_pwm_diff; // 计算舵机PWM值
    
    if (servo_pwm > 1000) // 如果PWM值大于900
    {
        servo_pwm = 1000; // 限制PWM值为900
    }
    else if (servo_pwm < 600) // 如果PWM值小于600
    {
        servo_pwm = 600; // 限制PWM值为600
    }
    return servo_pwm; // 返回舵机PWM值
}


// 功能: 根据最近识别的车库ID（A=1/B=2）执行入库动作序列
void gohead(int parkchose){
    if(parkchose == 1 ){ //try to find park A
        std::cout << "[停车调试] 前往A车库目标，执行前进动作" << std::endl;
        gpioPWM(13, motor_pwm_mid + 2800);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 690); // 设置舵机PW0M
        sleep(1);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 780); // 设置舵机PW0M
        // sleep(2);
        usleep(2200000);
        gpioPWM(13, motor_pwm_mid);
    }
    else if(parkchose == 2){ //try to find park B
        cout << "[停车调试] 前往B车库目标，执行前进动作" << endl;
        gpioPWM(13, motor_pwm_mid + 2800);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 690); // 设置舵机PWM
        sleep(1);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 650); // 设置舵机PW0M
        // usleep(1800000);
        sleep(2);
        gpioPWM(13, motor_pwm_mid);
    }
}

// 功能: 斑马线触发停车：电机回中、舵机回中并输出日志
void banma_stop(){
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock); // 解锁状态，即停车
    gpioPWM(servo_pin, servo_pwm_mid); // 舵机回中
    cout << "[流程] 检测到斑马线，车辆停车3秒等待指令" << endl;
}

// 功能: 按照 `changeroad` 状态执行左/右变道动作序列
void motor_changeroad(){
    if(changeroad == 1){ // 向左变道----------------------------------------------------------------
        gpioPWM(12, 825); // 设置舵机PWM
        gpioPWM(13, motor_pwm_mid + 1300); // 设置电机PWM
        usleep(1200000); // RIGHT弯道
        // usleep(1000000); // LEFT弯道
        gpioPWM(12, 610); // 设置舵机PWM
        gpioPWM(13, motor_pwm_mid + 1300); // 设置电机PWM
        usleep(800000); // 延时550毫秒
        gpioPWM(12, servo_pwm_mid); // 设置舵机PWM
        gpioPWM(13, motor_pwm_mid + 1400); // 设置电机PWM
    }
    else if(changeroad == 2){ //向右变道----------------------------------------------------------------
        gpioPWM(12, 620); // 设置舵机PWM
        gpioPWM(13, motor_pwm_mid + 1300); // 设置电机PWM
        usleep(1400000);
        gpioPWM(12, 820); // 设置舵机PWM
        gpioPWM(13, motor_pwm_mid + 1300); // 设置电机PWM
        usleep(500000); // 延时550毫秒
        gpioPWM(12, servo_pwm_mid); // 设置舵机PWM
        gpioPWM(13, motor_pwm_mid + 1400); // 设置电机PWM
    }
}


// 控制舵机电机
// 功能: 根据状态机切换控制策略（巡线/避障/停车），并下发PWM
void motor_servo_contral()
{
    float servo_pwm_now;

    // 如果正在停车，则由主循环的计时器逻辑控制，这里不执行任何操作
    if (is_stopping_at_zebra) {
        return;
    }

    // 安全检查：如果还没发车，不执行控制
    if (fache_sign == 0) {
        return;
    }

    if (is_parking_phase)
    {
        // 状态4: 寻找并进入车库
        servo_pwm_now = servo_pd_AB(160); // 使用为停车优化的PD控制
        gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_PARK); // 停车时使用稳定速度
    }
    else if (is_in_avoidance) { // 使用避障状态锁来决定控制策略
        // 状态：正在主动避障
        servo_pwm_now = servo_pd_bz(160); // 使用为避障优化的PD控制
        gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_AVOID); // 避障时使用较慢速度
    } else {
        // 状态：常规巡线（包括寻找斑马线，或避障间隙）
        servo_pwm_now = servo_pd(160); // 使用常规PD控制
        gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_CRUISE); // 使用常规速度
    }
    gpioPWM(servo_pin, servo_pwm_now);
}

//-----------------------------------------------------------------------------------主函数-----------------------------------------------
// 功能: 主循环，完成相机初始化、状态机执行与控制闭环
int main(int argc, char* argv[])
{
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--no-fps") {
            SHOW_FPS = false;
            cout << "[参数] FPS显示已禁用" << endl;
        } else if (std::string(argv[i]) == "--show-fps") {
            SHOW_FPS = true;
            cout << "[参数] FPS显示已启用" << endl;
        }
    }

    // 初始化检测模型
    cout << "[初始化] 加载障碍物检测模型..." << endl;
    try {
        fastestdet_obs = new FastestDet(model_param_obs, model_bin_obs, num_classes_obs, labels_obs, 352, 0.6f, 0.4f, 4, false);
        cout << "[初始化] 障碍物检测模型加载成功!" << endl;
    } catch (const std::exception& e) {
        cerr << "[错误] 障碍物检测模型加载失败: " << e.what() << endl;
        return -1;
    }

    cout << "[初始化] 加载转向标志检测模型..." << endl;
    try {
        fastestdet_lr = new FastestDet(model_param_lr, model_bin_lr, num_classes_lr, labels_lr, 352, 0.6f, 0.5f, 4, false);
        cout << "[初始化] 转向标志检测模型加载成功!" << endl;
    } catch (const std::exception& e) {
        cerr << "[错误] 转向标志检测模型加载失败: " << e.what() << endl;
        delete fastestdet_obs;
        return -1;
    }

    cout << "[初始化] 加载车库检测模型..." << endl;
    try {
        fastestdet_ab = new FastestDet(model_param_ab, model_bin_ab, num_classes_ab, labels_ab, 352, 0.6f, 0.5f, 4, false);
        cout << "[初始化] 车库检测模型加载成功!" << endl;
    } catch (const std::exception& e) {
        cerr << "[错误] 车库检测模型加载失败: " << e.what() << endl;
        delete fastestdet_obs;
        delete fastestdet_lr;
        return -1;
    }

    cout << "[初始化] 所有模型加载完成!\n";
    cout << "===========================================\n";

    gpioTerminate();           // 终止GPIO操作
    servo_motor_pwmInit();     // 初始化舵机PWM

//----------------打开摄像头------------------------------------------------
    VideoCapture capture;       // 视频捕获对象
    capture.open(0);           // 打开默认摄像头

    // 设置视频属性
    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(cv::CAP_PROP_FPS, 30); // 设置帧率
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 320); // 设置帧宽
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 240); // 设置帧高

    if (!capture.isOpened())   // 检查摄像头是否成功打开
    {
        cout << "无法打开摄像头，请检查设备连接！" << endl;
        cout << "按任意键退出程序" << endl;
        cin.ignore();// 等待用户输入
        return -1;            // 返回错误代码
    }

    // 输出摄像头的属性
    cout << "摄像头帧率: " << capture.get(cv::CAP_PROP_FPS) << endl;
    cout << "摄像头宽度: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << endl;
    cout << "摄像头高度: " << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;
    //---------------------------------------------------

    auto lastDebugRefresh = std::chrono::steady_clock::now();
    cv::Mat lastDebugOverlay;

    while (capture.read(frame)){

        // 安全检查：确保frame有效
        if (frame.empty()) {
            cerr << "[错误] 读取到空帧，跳过处理" << endl;
            continue;
        }

        try {
            // 1. 图像预处理：畸变校正
            frame = undistort(frame);

            // 记录单帧处理起始时间
            auto start = std::chrono::high_resolution_clock::now();
            
            // 2. 发车逻辑：检测蓝色挡板
            if (fache_sign == 0) // 发车标志为0，说明还未发车
            {
                if (find_first == 0) // 若还未找到过挡板
                {
                    blue_card_find(); // 持续寻找蓝色挡板
                }
                else // 若已找到过挡板，则进入移开检测阶段
                {
                    blue_card_remove(); // 检测蓝色挡板是否已移开
                }

            }
            else // 发车标志为1，车辆启动
            {
                number++; // 帧计数器累加

                // 1. 图像处理与车道线识别
                const auto now = std::chrono::steady_clock::now();
                const bool shouldRefreshDebug = SHOW_SOBEL_DEBUG &&
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - lastDebugRefresh).count() >= SOBEL_DEBUG_REFRESH_INTERVAL_MS;

                cv::Mat debugOverlay;
                cv::Mat* debugPtr = (SHOW_SOBEL_DEBUG && shouldRefreshDebug) ? &debugOverlay : nullptr;
                
                bin_image = ImageSobel(frame, debugPtr); // Sobel等处理提取二值化图像

            // (可选) 显示调试图像
            if (SHOW_SOBEL_DEBUG && shouldRefreshDebug)
            {
                if (!debugOverlay.empty()) lastDebugOverlay = debugOverlay;
                if (!lastDebugOverlay.empty()) cv::imshow("TrackLine Overlay", lastDebugOverlay);
                lastDebugRefresh = now;
            }
            if (SHOW_SOBEL_DEBUG) cv::waitKey(1);

            // 2. 主状态机逻辑
            if (is_stopping_at_zebra)
            {
                // 状态2: 在斑马线处停车，并检测转向标志
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - zebra_stop_start_time).count();
                if (elapsed < 3) {
                    // 3秒停车时间内，持续检测转向标志
                    result.clear();
                    result = fastestdet_lr->detect(frame);
                    if (!result.empty()) {
                        changeroad = result[0].label + 1; // label 0 -> left (1), label 1 -> right (2)
                        has_detected_turn_sign = true; // 标记已成功识别
                        cout << "[流程] 检测到转向标识：" << (changeroad == 1 ? "左转" : "右转") << endl;
                    }
                } else {
                    // 3秒结束，检查是否已识别到转向标识
                    if (has_detected_turn_sign) {
                        // 已识别到转向标识，执行转向
                        is_stopping_at_zebra = false;
                        cout << "[流程] 停车时间结束，执行" << (changeroad == 1 ? "左转" : "右转") << "动作" << endl;
                        motor_changeroad(); // 执行转向动作
                        flag_turn_done = 1; // 标记转向完成
                        is_parking_phase = true; // 进入寻找车库阶段
                        cout << "[流程] 转向完成，开始寻找并识别A/B车库" << endl;
                    } else {
                        // 未识别到转向标识，继续等待并检测
                        cout << "[警告] 未检测到转向标识，继续等待识别..." << endl;
                        zebra_stop_start_time = std::chrono::steady_clock::now(); // 重置计时器，继续等待
                    }
                }
            }
            else if (is_parking_phase)
            {
                // 状态4: 寻找并进入车库
                Tracking(bin_image); // 继续基础巡线以保持姿态
                
                result_ab.clear();
                result_ab = fastestdet_ab->detect(frame); // 使用FastestDet模型检测A/B

                DetectObject closest_box = {cv::Rect_<float>(0, 0, 0, 0), -1, 0.0f};
                
                if (!result_ab.empty())
                {
                    // 找到y2最大的那个检测框，即离得最近的
                    for(const auto& box : result_ab) {
                        float box_y2 = box.rect.y + box.rect.height;
                        float closest_y2 = closest_box.rect.y + closest_box.rect.height;
                        if (box_y2 > closest_y2) {
                            closest_box = box;
                        }
                    }
                    
                    float closest_y2 = closest_box.rect.y + closest_box.rect.height;
                    latest_park_id = closest_box.label + 1; // 0 for A -> 1, 1 for B -> 2
                    cout << "[停车] 检测到最近车库: " << (latest_park_id == 1 ? "A" : "B") 
                         << "，底部位置: " << (int)closest_y2 << "/" << PARKING_Y_THRESHOLD << endl;

                    // 检查是否达到入库阈值
                    if (closest_y2 >= PARKING_Y_THRESHOLD) {
                        cout << "[停车] 已达到入库阈值，执行入库 -> " << (latest_park_id == 1 ? "A" : "B") << endl;
                        gohead(latest_park_id);
                        is_parking_phase = false; // 避免重复执行
                    }
                }
            }
            else if (is_in_avoidance)
            {
                // 状态3: 正在执行避障
                Tracking(bin_image); // 仍然需要常规巡线来获取左右边界参考
                
                bz_get = 0;
                result = fastestdet_obs->detect(frame);
                if (!result.empty()) {
                    DetectObject box = result.at(0);
                    int box_y2 = static_cast<int>(box.rect.y + box.rect.height);
                    if (box_y2 < bz_y2) {
                        bz_get = 1;
                        int box_x1 = static_cast<int>(box.rect.x);
                        int box_x2 = static_cast<int>(box.rect.x + box.rect.width);
                        int box_y1 = static_cast<int>(box.rect.y);
                        last_known_bz_xcenter = (box_x1 + box_x2) / 2;
                        last_known_bz_bottom = box_y2;
                        last_known_bz_heighest = box_y1;
                        bz_disappear_count = 0; // 障碍物可见，重置消失计数
                    }
                }

                if (bz_get == 0) {
                    bz_disappear_count++; // 障碍物不可见，累加消失计数
                }

                // 只要在避障状态，就始终使用最后记录的位置进行补线
                bz_heighest = last_known_bz_heighest; // 确保Tracking_bz使用正确的边界
                if (last_known_bz_xcenter > 160) {
                    bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_xcenter, last_known_bz_bottom), cv::Point(int((right_line[0].x + right_line[1].x + right_line[2].x) / 3), 155), 8);
                } else {
                    bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_xcenter, last_known_bz_bottom), cv::Point(int((left_line[0].x + left_line[1].x + left_line[2].x) / 3), 155), 8);
                }
                Tracking_bz(bin_image);

                // 检查是否满足退出避障的条件
                if (bz_disappear_count >= BZ_DISAPPEAR_THRESHOLD) {
                    is_in_avoidance = false;
                    count_bz++;
                    bz_disappear_count = 0;
                    cout << "[流程] 障碍物已安全绕过，退出避障模式" << endl;
                }
            }
            else
            {
                // 状态0/1: 默认巡航状态 (寻找障碍物或斑马线)
                Tracking(bin_image); // 识别常规车道线

                if (count_bz >= 1 && flag_turn_done == 0)
                {
                    // 状态1: 已完成至少一次避障，且尚未完成转向，此时寻找斑马线
                    banma = banma_get(frame);
                    if (banma == 1) {
                        is_stopping_at_zebra = true; //切换到停车状态
                        has_detected_turn_sign = false; // 重置转向标识检测标志
                        changeroad = 0; // 重置转向方向
                        zebra_stop_start_time = std::chrono::steady_clock::now();
                        cout << "[流程] 避障结束，检测到斑马线，准备停车识别" << endl;
                        banma_stop(); // 执行停车
                        system("mpg123 /home/pi/dev_ws/月半猫.mp3"); // 播放斑马线提示音
                    }
                }
                else
                {
                    // 状态0: 默认状态，执行障碍物检测以启动避障
                    bz_get = 0;
                    result = fastestdet_obs->detect(frame); 

                    if (result.size() > 0) { 
                        DetectObject box = result.at(0);
                        int box_y2 = static_cast<int>(box.rect.y + box.rect.height);
                        if (box_y2 < bz_y2) { 
                            bz_get = 1; 
                            is_in_avoidance = true; // 启动并锁定避障状态
                            cout << "[流程] 检测到障碍物，进入避障模式" << endl;
                            
                            // 记录障碍物的初始位置
                            int box_x1 = static_cast<int>(box.rect.x);
                            int box_x2 = static_cast<int>(box.rect.x + box.rect.width);
                            int box_y1 = static_cast<int>(box.rect.y);
                            last_known_bz_xcenter = (box_x1 + box_x2) / 2;
                            last_known_bz_bottom = box_y2;
                            last_known_bz_heighest = box_y1;
                            bz_heighest = last_known_bz_heighest;

                            // 立即执行第一次补线和避障巡线
                            if (last_known_bz_xcenter > 160) { 
                                bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_xcenter, last_known_bz_bottom), cv::Point(int((right_line[0].x + right_line[1].x + right_line[2].x) / 3), 155), 8);
                            } else { 
                                bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_xcenter, last_known_bz_bottom), cv::Point(int((left_line[0].x + left_line[1].x + left_line[2].x) / 3), 155), 8);
                            }
                            Tracking_bz(bin_image); 
                        }
                    }
                }
            }
        }

        // 3. 电机与舵机控制
        motor_servo_contral(); // 根据当前状态（常规/避障）控制车辆运动

        // 计算并打印FPS
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double instantFps = (elapsed.count() > 0 ? 1.0 / elapsed.count() : 0.0);

        // 根据SHOW_FPS设置决定是否输出FPS信息
        if (SHOW_FPS) {
            std::cout << "[性能] 当前FPS: " << std::fixed << std::setprecision(1) << instantFps
                      << " | 已处理帧数: " << number << std::endl;
        }
        
        } catch (const cv::Exception& e) {
            cerr << "[错误] OpenCV异常: " << e.what() << endl;
            cerr << "  位置: " << e.file << ":" << e.line << endl;
            // 继续下一帧
            continue;
        } catch (const std::exception& e) {
            cerr << "[错误] 标准异常: " << e.what() << endl;
            // 继续下一帧
            continue;
        } catch (...) {
            cerr << "[错误] 未知异常，跳过当前帧" << endl;
            // 继续下一帧
continue;
        }

    }

    // 清理资源
    cout << "\n[清理] 释放模型资源..." << endl;
    if (fastestdet_obs) {
        delete fastestdet_obs;
        fastestdet_obs = nullptr;
    }
    if (fastestdet_lr) {
        delete fastestdet_lr;
        fastestdet_lr = nullptr;
    }
    if (fastestdet_ab) {
        delete fastestdet_ab;
        fastestdet_ab = nullptr;
    }
    
    gpioTerminate(); // 终止GPIO
    cout << "[清理] 系统退出完成" << endl;
    
    return 0;
}
