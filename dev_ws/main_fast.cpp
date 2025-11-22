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
#include <ctime> // 时间格式化
#include <sstream> // 字符串流
#include <sys/stat.h> // 目录操作

#include "fastestdet.hpp" // FastestDet库

using namespace std; // 使用标准命名空间
using namespace cv; // 使用OpenCV命名空间

bool program_finished = false; // 控制主循环退出的标志

//------------速度参数配置------------------------------------------------------------------------------------------
const int MOTOR_SPEED_DELTA_CRUISE = 1300; // 常规巡航速度增量
const int MOTOR_SPEED_DELTA_AVOID = 1100;  // 避障阶段速度增量
const int MOTOR_SPEED_DELTA_PARK = 1000;   // 车库阶段速度增量
const int MOTOR_SPEED_DELTA_BRAKE = -3000; // 瞬时反转/刹停增量

const float BRIEF_STOP_REVERSE_DURATION = 0.5f; // 反转阶段持续时间（秒）
const float BRIEF_STOP_HOLD_DURATION = 0.1f;    // 刹停保持时间（秒）

//---------------调试选项-------------------------------------------------
const bool SHOW_SOBEL_DEBUG = false; // 是否显示Sobel调试窗口
const int SOBEL_DEBUG_REFRESH_INTERVAL_MS = 120; // 调试窗口刷新间隔，减轻imshow开销

//---------------性能统计---------------------------------------------------
int number = 0; // 已处理帧计数
bool SHOW_FPS = false; // 是否显示FPS信息，可通过命令行参数控制

//------------有关的全局变量定义------------------------------------------------------------------------------------------

std::vector<DetectObject> result; // 存储FastestDet检测结果 
std::vector<DetectObject> result_ab; // 存储FastestDet检测结果

enum class BriefStopType { None, Obstacle, Parking };
enum class BriefStopNextAction { None, ResumeAvoidance, EnterPreParking };

bool is_brief_stop_active = false;
BriefStopNextAction brief_stop_next_action = BriefStopNextAction::None;
std::chrono::steady_clock::time_point brief_stop_start_time;
int pending_pre_parking_label = -1;

// 模型路径配置
std::string model_param_obs = "models/obs.param";
std::string model_bin_obs = "models/obs.bin";
// 障碍物检测类别：0 = blue, 1 = yellow
int num_classes_obs = 2;
std::vector<std::string> labels_obs{"blue", "yellow"};

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
bool has_detected_turn_sign = false; // 是否已成功识别到转向标识

//----------------避障相关---------------------------------------------------
int bz_heighest = 0; // 避障高度
int bz_get = 0;
std::vector<cv::Point> mid_bz; // 存储中线
std::vector<cv::Point> left_line_bz; // 存储左线条
std::vector<cv::Point> right_line_bz; // 存储右线条
std::vector<cv::Point> last_mid_bz; // 存储上一帧避障中线
bool is_in_avoidance = false; // 是否处于避障状态锁
int last_known_bz_xcenter = 0; // 最后一次检测到的障碍物位置
int last_known_bz_x1 = 0;      // 最后一次检测到的障碍物左边界
int last_known_bz_x2 = 0;      // 最后一次检测到的障碍物右边界
int last_known_bz_bottom = 0;
int last_known_bz_heighest = 0;
int count_bz = 0; // 避障计数器
int bz_disappear_count = 0; // 障碍物连续消失计数
const int BZ_DISAPPEAR_THRESHOLD = 3; // 确认障碍物消失的帧数阈值
const int BZ_Y_UPPER_THRESHOLD = 200; // 可见障碍物底部阈值 (上限)
const int BZ_Y_LOWER_THRESHOLD = 40; // 触发避障的Y轴下限阈值 (下限)

int bz_detect_count = 0; // 障碍物连续检测计数
const int BZ_DETECT_THRESHOLD = 3; // 确认障碍物出现的帧数阈值

//----------------停车相关---------------------------------------------------
int flag_turn_done = 0; // 转向完成标志
std::chrono::steady_clock::time_point zebra_stop_start_time;
bool is_stopping_at_zebra = false;
std::chrono::steady_clock::time_point post_zebra_delay_start_time; // Timer for delay after zebra crossing
bool is_in_post_zebra_delay = false; // Flag for delay state after zebra crossing
bool is_parking_phase = false; // 是否进入寻找车库阶段
bool is_pre_parking = false; // 是否在预入库阶段
int latest_park_id = 0; // 最近检测到的车库ID (1=A, 2=B)
int park_A_count = 0; // A车库累计识别次数
int park_B_count = 0; // B车库累计识别次数
const int PARKING_Y_THRESHOLD = 120; // 触发入库的Y轴阈值
int final_target_label = -1;       // 最终锁定的AB标志的label (0 for A, 1 for B)

// 发车延时相关：挡板移开后等待3秒再开始电机/舵机控制
bool is_start_delay = false; // 挡板移开后的发车延时标志
std::chrono::steady_clock::time_point start_delay_time; // 挡板移开时间戳

//----------------图像保存相关---------------------------------------------------
std::chrono::steady_clock::time_point last_save_time; // 上次保存图像的时间
const int SAVE_INTERVAL_SECONDS = 30; // 保存间隔（秒）
const std::string SAVE_DIR = "captured_images"; // 保存目录

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
const float yuntai_UD_pwm_duty_cycle_unlock = 55.0; //大上下小

//---------------平滑滤波相关-------------------------------------------------
std::vector<cv::Point> last_mid; // 存储上一次的中线，用于平滑滤波
int blue_detect_count = 0; // 蓝色挡板连续检测计数
const int BLUE_DETECT_THRESHOLD = 5; // 需要连续检测到的帧数才能确认找到蓝色挡板

// 预入库阶段参数
std::chrono::steady_clock::time_point pre_parking_start_time; // 预入库阶段开始时间
const int PARKING_DETECT_MISS_THRESHOLD = 5; // 检测不到的帧数阈值，达到后停车

// 预入库阶段跟随目标相关
int parking_target_not_detected_count = 0; // 连续检测不到目标的帧数
int parking_follow_x = 160; // 当前跟随目标的x坐标（默认中心）
bool parking_target_detected_this_frame = false; // 当前帧是否检测到目标


//---------------蓝色检测参数------------------------------------------
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

// 蓝色面积阈值
const double BLUE_AREA_VALID = 2000.0; // 有效面积阈值

// 蓝色挡板移开检测参数
const double BLUE_REMOVE_AREA_MIN = 500.0; // 移开检测的最小面积阈值（过滤小噪点）

//---------------斑马线检测参数（可调节）------------------------------------------
// 斑马线检测ROI区域
const int BANMA_ROI_X = 10;           // ROI左上角X坐标
const int BANMA_ROI_Y = 130;          // ROI左上角Y坐标 (下移)
const int BANMA_ROI_WIDTH = 300;      // ROI宽度
const int BANMA_ROI_HEIGHT = 60;     // ROI高度 (减小)

// 斑马线矩形筛选尺寸
const int BANMA_RECT_MIN_WIDTH = 15;   // 矩形最小宽度 (调高以过滤噪点)
const int BANMA_RECT_MAX_WIDTH = 100;  // 矩形最大宽度
const int BANMA_RECT_MIN_HEIGHT = 15;   // 矩形最小高度
const int BANMA_RECT_MAX_HEIGHT = 100;  // 矩形最大高度 (调低以排除车道线)

// 判定为斑马线需要的最少白色矩形数量 (根据实际情况调整)
const int BANMA_MIN_COUNT = 4;

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
        resizedFrame = frame; // 无需克隆，后续操作不会修改原始frame
    }

    const cv::Rect roiRect(1, 109, 318, 46); // 巡线ROI区域
    cv::Mat roi = resizedFrame(roiRect); // 直接使用ROI视图

    cv::Mat grayRoi;
    cv::cvtColor(roi, grayRoi, cv::COLOR_BGR2GRAY); // ROI灰度化

    int kernelSize = 5;
    cv::Mat blurredRoi;
    cv::blur(grayRoi, blurredRoi, cv::Size(kernelSize, kernelSize)); // ROI均值滤波降噪

    cv::Mat sobelX, sobelY;
    // 使用CV_16S以提高性能，避免使用昂贵的CV_64F浮点运算
    cv::Sobel(blurredRoi, sobelX, CV_16S, 1, 0, 3);
    cv::Sobel(blurredRoi, sobelY, CV_16S, 0, 1, 3);

    // 转换回CV_8U并计算梯度
    cv::Mat absSobelX, absSobelY;
    cv::convertScaleAbs(sobelX, absSobelX);
    cv::convertScaleAbs(sobelY, absSobelY);
    
    // 组合梯度，权重偏向Y方向
    cv::Mat gradientMagnitude8U;
    cv::addWeighted(absSobelY, 1.0, absSobelX, 0.5, 0, gradientMagnitude8U);

    // 顶帽操作减弱阴影
    cv::Mat topHat;
    static cv::Mat kernel_tophat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 3));
    cv::morphologyEx(blurredRoi, topHat, cv::MORPH_TOPHAT, kernel_tophat);

    cv::Mat adaptiveMask;
    cv::threshold(topHat, adaptiveMask, 10, 255, cv::THRESH_BINARY);

    cv::Mat gradientMask;
    cv::threshold(gradientMagnitude8U, gradientMask, 50, 255, cv::THRESH_BINARY); // 梯度二值掩码
    static cv::Mat kernel_gradient_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(gradientMask, gradientMask, kernel_gradient_dilate);

    cv::Mat binaryMask;
    cv::bitwise_and(adaptiveMask, gradientMask, binaryMask); // 亮度+梯度联合约束

    cv::medianBlur(binaryMask, binaryMask, 3); // 中值去椒盐噪声
    // cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_OPEN, noiseKernel); // 小结构开运算 - 1x1内核无效，已移除

    // 原地执行形态学操作，避免binaryMask.clone()的开销
    static cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 5)); // 闭运算连接断裂
    cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_CLOSE, kernel_close);
    static cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); // 膨胀加粗车道线
    cv::dilate(binaryMask, binaryMask, kernel_dilate, cv::Point(-1, -1), 1);

    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binaryMask, labels, stats, centroids, 8, CV_32S); // 连通域分析
    cv::Mat filteredMorph = cv::Mat::zeros(binaryMask.size(), CV_8U);
    for (int i = 1; i < numLabels; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= MIN_COMPONENT_AREA)
        {
            filteredMorph.setTo(255, labels == i);
        }
    }

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(filteredMorph, lines, 1, CV_PI / 180, 20, 15, 8);

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
        // 挡板移开后开始计时，延时2秒再允许控制函数运行
        is_start_delay = true;
        start_delay_time = std::chrono::steady_clock::now();
    } 
    else 
    {
        cout << "仍检测到蓝色物体（面积：" << contourArea(validContours[0]) << "），等待移开..." << endl;
    }
}

// 功能: 使用顶帽变换与宽高比筛选检测斑ma线，增强光照鲁棒性
int banma_get(cv::Mat &frame) {
    // 1. 裁剪调整后的ROI
    int roiWidth = std::min(BANMA_ROI_WIDTH, frame.cols - BANMA_ROI_X);
    int roiHeight = std::min(BANMA_ROI_HEIGHT, frame.rows - BANMA_ROI_Y);
    if (roiWidth <= 0 || roiHeight <= 0) {
        return 0; // ROI无效
    }
    cv::Rect roiRect(BANMA_ROI_X, BANMA_ROI_Y, roiWidth, roiHeight);
    cv::Mat roiFrame = frame(roiRect).clone();

    // 2. 灰度化
    cv::Mat grayRoi;
    cv::cvtColor(roiFrame, grayRoi, cv::COLOR_BGR2GRAY);

    // 3. 顶帽变换 - 核心步骤，用于在复杂光照下突出白色条纹
    cv::Mat topHat;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 3));
    cv::morphologyEx(grayRoi, topHat, cv::MORPH_TOPHAT, kernel);

    // 4. 二值化
    cv::Mat binaryMask;
    cv::threshold(topHat, binaryMask, 40, 255, cv::THRESH_BINARY);

    // 5. 形态学开运算（先腐蚀再膨胀），去除小的噪声点
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_OPEN, openKernel);

    // 6. 查找轮廓并应用尺寸筛选
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int count_BMX = 0;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);

        // 应用尺寸筛选
        bool size_ok = (rect.width >= BANMA_RECT_MIN_WIDTH && rect.width <= BANMA_RECT_MAX_WIDTH &&
                        rect.height >= BANMA_RECT_MIN_HEIGHT && rect.height <= BANMA_RECT_MAX_HEIGHT);

        if (size_ok) {
            count_BMX++;
        }
    }
    
    // 7. 最终判定
    if (count_BMX >= BANMA_MIN_COUNT) {
        cout << "检测到斑马线（白色矩形数量：" << count_BMX << "）" << endl;
        return 1;
    } else {
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

    float kp = 0.8; // 比例系数
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
    float kp = 1.5; // 比例系数
    float kd = 4.0; // 微分系数

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

// 功能: 预入库阶段跟随AB目标的PD控制器，P和D参数较大，响应更灵敏
float servo_pd_parking(int ab_center_x) { // 跟随AB目标控制，ab_center_x是AB中心点x坐标

    const int IMAGE_CENTER_X = 160; // 图像中心x坐标，作为目标位置（类似巡线时的target）
    int target = IMAGE_CENTER_X; // 目标位置（类似巡线时的target=160）
    int pidx = ab_center_x; // AB中心点位置（类似巡线时的pidx）

    float kp = 2.0; 
    float kd = 4.0; 

    error_first = target - pidx; // 计算误差：目标位置(160) - AB位置(pidx)

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); // 计算舵机PWM差值

    last_error = error_first; // 更新上一次误差

    servo_pwm = servo_pwm_mid + servo_pwm_diff; // 计算舵机PWM值

    return servo_pwm; // 返回舵机PWM值
}



// 功能: 斑马线触发停车：电机回中、舵机回中并输出日志
void banma_stop(){
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock - 3000); // 解锁状态，即停车
    usleep(500000);
    cout << "[流程] 检测到斑马线，车辆停车3秒等待指令" << endl;
}

// 功能: 在图像上绘制时间戳（加8小时时差）并保存
void save_frame_with_timestamp(const cv::Mat& frame) {
    // 创建保存目录（如果不存在）
    struct stat info;
    if (stat(SAVE_DIR.c_str(), &info) != 0) {
        // 目录不存在，创建它
        mkdir(SAVE_DIR.c_str(), 0755);
    }

    // 获取当前时间（系统时间）
    auto now = std::chrono::system_clock::now();
    // 加上8小时时差（8小时 = 8 * 3600秒）
    auto china_time = now + std::chrono::hours(8);
    
    // 转换为time_t以便格式化
    std::time_t time_t_china = std::chrono::system_clock::to_time_t(china_time);
    
    // 格式化为字符串：YYYY-MM-DD HH:MM:SS
    std::tm* tm_info = std::gmtime(&time_t_china);
    std::ostringstream oss;
    oss << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S");
    std::string timestamp_str = oss.str();

    // 克隆图像以便绘制时间戳
    cv::Mat frame_with_timestamp = frame.clone();

    // 在图像上绘制时间戳（左上角，绿色文字，带黑色边框以提高可读性）
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.6;
    int thickness = 2;
    cv::Scalar text_color(0, 255, 0); // 绿色
    cv::Scalar outline_color(0, 0, 0); // 黑色边框
    
    cv::Point text_pos(10, 30); // 左上角位置
    
    // 先绘制黑色边框（稍微偏移）
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx != 0 || dy != 0) {
                cv::putText(frame_with_timestamp, timestamp_str, 
                           cv::Point(text_pos.x + dx, text_pos.y + dy),
                           font_face, font_scale, outline_color, thickness + 1);
            }
        }
    }
    
    // 再绘制绿色文字
    cv::putText(frame_with_timestamp, timestamp_str, text_pos,
                font_face, font_scale, text_color, thickness);

    // 生成文件名（使用时间戳作为文件名）
    std::ostringstream filename_oss;
    filename_oss << std::put_time(tm_info, "%Y%m%d_%H%M%S");
    std::string filename = SAVE_DIR + "/" + filename_oss.str() + ".jpg";

    // 保存图像（如果文件已存在会自动覆盖）
    if (cv::imwrite(filename, frame_with_timestamp)) {
        cout << "[图像保存] 已保存: " << filename << " (时间戳: " << timestamp_str << ")" << endl;
    } else {
        cerr << "[错误] 保存图像失败: " << filename << endl;
    }
}

void start_brief_stop(BriefStopType type, BriefStopNextAction next_action)
{
    is_brief_stop_active = true;
    brief_stop_next_action = next_action;
    brief_stop_start_time = std::chrono::steady_clock::now();

    std::string reason;
    if (type == BriefStopType::Obstacle) {
        reason = "障碍物";
    } else if (type == BriefStopType::Parking) {
        reason = "入库阈值";
    } else {
        reason = "未知";
    }
    cout << "[流程] " << reason << "触发短暂停车，执行反向刹停..." << endl;
}

int decide_parking_label_from_counts()
{
    if (park_A_count > park_B_count) return 0;
    if (park_B_count > park_A_count) return 1;
    if (latest_park_id != 0) return latest_park_id - 1;
    return -1;
}

void finalize_brief_stop_action()
{
    if (brief_stop_next_action == BriefStopNextAction::EnterPreParking)
    {
        int decided_label = pending_pre_parking_label;
        if (decided_label == -1)
        {
            decided_label = decide_parking_label_from_counts();
        }

        if (decided_label != -1)
        {
            is_pre_parking = true;
            pre_parking_start_time = std::chrono::steady_clock::now();
            final_target_label = decided_label;
            // 初始化预入库阶段的变量
            parking_target_not_detected_count = 0;
            // 如果parking_follow_x还是默认值（160），说明短暂停车前没有检测到目标，保持默认值
            // 否则使用短暂停车前保存的目标位置
            if (parking_follow_x == 160) {
                cout << "[流程] 短暂停车结束，综合计数结果，开始预入库阶段 -> "
                     << (final_target_label == 0 ? "A(左)" : "B(右)") 
                     << "，将跟随最远的" << (final_target_label == 0 ? "A" : "B") << "目标（使用默认中心位置）" << endl;
            } else {
                cout << "[流程] 短暂停车结束，综合计数结果，开始预入库阶段 -> "
                     << (final_target_label == 0 ? "A(左)" : "B(右)") 
                     << "，将跟随最远的" << (final_target_label == 0 ? "A" : "B") << "目标（已保存位置 x=" << parking_follow_x << "）" << endl;
            }
            parking_target_detected_this_frame = false;
            pending_pre_parking_label = -1;
        }
        else
        {
            cout << "[警告] 短暂停车后仍无法确认A/B车库，继续保持寻找车库状态" << endl;
            is_parking_phase = true;
        }
    }
    else if (brief_stop_next_action == BriefStopNextAction::ResumeAvoidance)
    {
        cout << "[流程] 短暂停车结束，继续避障巡线" << endl;
    }

    brief_stop_next_action = BriefStopNextAction::None;
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

    // 挡板移开后需要延时2秒再开始运动控制
    if (is_start_delay) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(now - start_delay_time).count();
        if (elapsed_sec < 2) {
            // 延时阶段保持电机停止、舵机回中
            gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock);
            gpioPWM(servo_pin, servo_pwm_mid);
            return;
        } else {
            is_start_delay = false; // 延时结束，后续正常控制
        }
    }

    if (is_brief_stop_active)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - brief_stop_start_time).count() / 1000.0f;

        if (elapsed < BRIEF_STOP_REVERSE_DURATION)
        {
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_BRAKE); // 瞬时反转
            return;
        }
        else if (elapsed < (BRIEF_STOP_REVERSE_DURATION + BRIEF_STOP_HOLD_DURATION))
        {
            gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock); // 原地等待
            return;
        }
        else
        {
            is_brief_stop_active = false;
            finalize_brief_stop_action();
            // 短暂停车完成后继续执行后续控制逻辑
        }
    }

    if (is_pre_parking)
    {
        // 状态: 预入库阶段 - 跟随最远的A或B目标
        // 始终使用上次检测到的目标x坐标（parking_follow_x）
        // 如果目标消失，继续使用上次记录的位置，不启用巡线
        int target_x = parking_follow_x;
        
        // 使用PD控制跟随目标x坐标（使用专门的parking PD控制器，P和D参数较大）
        servo_pwm_now = servo_pd_parking(target_x);
        gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_PARK);
    }
    else if (is_parking_phase)
    {
        // 状态4: 寻找并进入车库
        if (latest_park_id != 0) {
            // 已识别到车库，跟随最近的A或B标识
            servo_pwm_now = servo_pd_parking(parking_follow_x); 
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_PARK);
        } else {
            // 未识别到车库，继续常规巡线
            servo_pwm_now = servo_pd(160);
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_CRUISE);
        }
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
        fastestdet_obs = new FastestDet(model_param_obs, model_bin_obs, num_classes_obs, labels_obs, 352, 0.6f, 0.6f, 4, false);
        cout << "[初始化] 障碍物检测模型加载成功!" << endl;
    } catch (const std::exception& e) {
        cerr << "[错误] 障碍物检测模型加载失败: " << e.what() << endl;
        return -1;
    }

    cout << "[初始化] 加载转向标志检测模型..." << endl;
    try {
        fastestdet_lr = new FastestDet(model_param_lr, model_bin_lr, num_classes_lr, labels_lr, 352, 0.6f, 0.6f, 4, false);
        cout << "[初始化] 转向标志检测模型加载成功!" << endl;
    } catch (const std::exception& e) {
        cerr << "[错误] 转向标志检测模型加载失败: " << e.what() << endl;
        delete fastestdet_obs;
        return -1;
    }

    cout << "[初始化] 加载车库检测模型..." << endl;
    try {
        fastestdet_ab = new FastestDet(model_param_ab, model_bin_ab, num_classes_ab, labels_ab, 352, 0.8f, 0.8f, 4, false);
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
    
    // 初始化图像保存时间（设置为过去时间，这样发车后第一次检查就会立即保存）
    last_save_time = std::chrono::steady_clock::now() - std::chrono::seconds(SAVE_INTERVAL_SECONDS);

    while (capture.read(frame) && !program_finished){

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
            
            // 检查是否需要保存图像（每隔30秒，程序启动后就开始计时）
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - last_save_time).count();
            if (elapsed_seconds >= SAVE_INTERVAL_SECONDS) {
                save_frame_with_timestamp(frame);
                last_save_time = current_time; // 更新上次保存时间
            }
            
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
            // 短暂停车期间，根据后续动作继续执行相应逻辑（巡线、避障、车库检测等）
            // 但不会触发新的短暂停车，控制逻辑由motor_servo_contral()处理
            if (is_stopping_at_zebra)
            {
                // 状态2: 在斑马线处停车，并检测转向标志
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - zebra_stop_start_time).count();

                // 在3秒停车时间内，持续检测转向标志
                if (elapsed < 3)
                {
                    if (!has_detected_turn_sign)
                    { // 避免重复检测和打印
                        result.clear();
                        result = fastestdet_lr->detect(frame);
                        if (!result.empty())
                        {
                            has_detected_turn_sign = true;    // 标记已成功识别
                            cout << "[流程] 检测到转向标识：" << (result[0].label == 0 ? "左转" : "右转") << endl;
                        }
                    }
                }
                else
                {
                    // 3秒结束，无论是否识别到转向标识，都直接继续巡线
                    is_stopping_at_zebra = false;
                    flag_turn_done = 1;      // 标记"变道"阶段已完成（跳过）
                    is_in_post_zebra_delay = true; // 进入巡线延迟阶段
                    post_zebra_delay_start_time = std::chrono::steady_clock::now(); // 启动延迟计时器
                    cout << "[流程] 停车结束，开始2秒常规巡线..." << endl;
                }
            }
            else if (is_in_post_zebra_delay)
            {
                // 状态: 斑马线后延迟巡线
                Tracking(bin_image); // 正常巡线

                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - post_zebra_delay_start_time).count();
                if (elapsed >= 2)
                {
                    // 1秒延迟结束，开始寻找车库
                    is_in_post_zebra_delay = false;
                    is_parking_phase = true;
                    cout << "[流程] 2秒巡线结束，开始寻找并识别A/B车库" << endl;
                }
            }
            else if (is_pre_parking)
            {
                // 状态: 预入库阶段 - 跟随最远的A或B目标
                // 不进行巡线，直接跟随A/B目标位置
                
                // 检测A/B目标
                result_ab.clear();
                result_ab = fastestdet_ab->detect(frame);
                
                parking_target_detected_this_frame = false;
                DetectObject farthest_target; // 最远的目标（y值最小）
                farthest_target.label = -1;
                float farthest_y = 9999.0f; // 初始化为很大的值
                
                if (!result_ab.empty()) {
                    // 找到最远的A或B（根据final_target_label选择，0=A，1=B）
                    for(const auto& box : result_ab) {
                        // 只选择匹配的目标标签
                        if (box.label == final_target_label) {
                            float box_y = box.rect.y; // y值越小表示越远
                            if (box_y < farthest_y) {
                                farthest_y = box_y;
                                farthest_target = box;
                                parking_target_detected_this_frame = true;
                            }
                        }
                    }
                    
                    if (parking_target_detected_this_frame) {
                        // 检测到目标，更新跟随坐标并重置计数器
                        float target_x = farthest_target.rect.x + farthest_target.rect.width / 2.0f;
                        parking_follow_x = static_cast<int>(target_x);
                        parking_target_not_detected_count = 0; // 重置计数
                        
                        cout << "[预入库] 检测到" << (final_target_label == 0 ? "A" : "B") 
                             << "目标，跟随x坐标: " << parking_follow_x 
                             << "，y坐标: " << (int)farthest_y << endl;
                    } else {
                        // 没有检测到匹配的目标，累加计数（继续使用上次记录的parking_follow_x）
                        parking_target_not_detected_count++;
                        cout << "[预入库] 未检测到" << (final_target_label == 0 ? "A" : "B") 
                             << "目标，使用上次检测位置(x=" << parking_follow_x << ")，未检测计数: " 
                             << parking_target_not_detected_count << "/" << PARKING_DETECT_MISS_THRESHOLD << endl;
                    }
                } else {
                    // 没有检测到任何目标，累加计数（继续使用上次记录的parking_follow_x）
                    parking_target_not_detected_count++;
                    cout << "[预入库] 未检测到任何目标，使用上次检测位置(x=" << parking_follow_x << ")，未检测计数: " 
                         << parking_target_not_detected_count << "/" << PARKING_DETECT_MISS_THRESHOLD << endl;
                }
                
                // 检查是否达到停车阈值
                if (parking_target_not_detected_count >= PARKING_DETECT_MISS_THRESHOLD) {
                    cout << "[流程] 预入库完成（连续" << PARKING_DETECT_MISS_THRESHOLD 
                         << "帧未检测到目标），刹车！" << endl;
                    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock); // 停车
                    
                    // 关闭当前摄像头
                    capture.release();
                    
                    // 重新打开摄像头，设置为最高分辨率
                    cout << "[流程] 重新打开摄像头，设置为最高分辨率(1280x720)..." << endl;
                    capture.open(0);
                    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                    capture.set(cv::CAP_PROP_FPS, 30);
                    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
                    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
                    
                    if (!capture.isOpened()) {
                        cerr << "[错误] 无法重新打开摄像头" << endl;
                    } else {
                        cout << "[流程] 摄像头已设置为: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) 
                             << "x" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;
                        cout << "[流程] 开始显示高分辨率图像，按'q'键退出..." << endl;
                        
                        // 持续显示高分辨率图像
                        cv::Mat high_res_frame;
                        while (capture.read(high_res_frame) && !high_res_frame.empty()) {
                            cv::imshow("Parking Complete - High Resolution", high_res_frame);
                            if ((cv::waitKey(1) & 0xFF) == 'q') {
                                break;
                            }
                        }
                        cv::destroyAllWindows();
                    }
                    
                    program_finished = true;
                    continue;
                }
            }
            else if (is_parking_phase || (is_brief_stop_active && brief_stop_next_action == BriefStopNextAction::EnterPreParking))
            {
                // 状态4: 寻找并进入车库（包括短暂停车期间继续检测）
                Tracking(bin_image); // 继续基础巡线以保持姿态
                
                result_ab.clear();
                result_ab = fastestdet_ab->detect(frame); // 使用FastestDet模型检测A/B

                DetectObject closest_box = {cv::Rect_<float>(0, 0, 0, 0), -1, 0.0f};
                
                if (!result_ab.empty())
                {
                    // 1. 找到y2最大的那个检测框（离得最近），且其底部满足阈值要求
                    for(const auto& box : result_ab) {
                        float box_y2 = box.rect.y + box.rect.height;
                        // 寻找到的AB底部需要大于障碍物阈值才算有效
                        if (box_y2 > BZ_Y_LOWER_THRESHOLD) {
                            float closest_y2 = closest_box.rect.y + closest_box.rect.height;
                            if (box_y2 > closest_y2) {
                                closest_box = box;
                            }
                        }
                    }
                    
                    // 如果经过筛选后，找到了有效的车库目标
                    if (closest_box.label != -1)
                    {
                        // 2. 仅以最近的为准，累加A和B的计数
                        if (closest_box.label == 0) { // 'A'
                            park_A_count++;
                        } else if (closest_box.label == 1) { // 'B'
                            park_B_count++;
                        }

                        float closest_y2 = closest_box.rect.y + closest_box.rect.height;
                        latest_park_id = closest_box.label + 1; // 0 for A -> 1, 1 for B -> 2
                        
                        // 保存最近的AB标识中心x坐标，用于跟随控制
                        float target_x = closest_box.rect.x + closest_box.rect.width / 2.0f;
                        parking_follow_x = static_cast<int>(target_x);
                        
                        cout << "[停车] 最近: " << (latest_park_id == 1 ? "A" : "B") 
                             << " | 计数 A:" << park_A_count << ", B:" << park_B_count
                             << " | Y:" << (int)closest_y2 << "/" << PARKING_Y_THRESHOLD
                             << " | 跟随x:" << parking_follow_x << endl;

                        // 检查是否达到入库阈值（短暂停车期间不触发新的停车）
                        if (closest_y2 >= PARKING_Y_THRESHOLD && !is_brief_stop_active) {
                            // 目标位置已在上面保存
                            is_parking_phase = false; // 寻找阶段结束
                            is_pre_parking = false;
                            pending_pre_parking_label = -1; // 停车后再根据总计数决定
                            start_brief_stop(BriefStopType::Parking, BriefStopNextAction::EnterPreParking);
                            cout << "[流程] 达到入库阈值，先短暂停车收集更多A/B计数再决策，已保存目标位置 x=" << parking_follow_x << endl;
                        }
                    }
                    else 
                    {
                        latest_park_id = 0; // 未检测到有效目标，重置
                    }
                }
                else
                {
                    latest_park_id = 0; // 未检测到，重置
                }
            }
            else if (is_in_avoidance)
            {
                // 状态3: 正在执行避障
                Tracking(bin_image); // 仍然需要常规巡线来获取左右边界参考
                
                bz_get = 0;
                result = fastestdet_obs->detect(frame);
                DetectObject blue_box;
                bool blue_obstacle_found = false;
                for (const auto& box : result)
                {
                    int box_y2 = static_cast<int>(box.rect.y + box.rect.height);
                    // 仅当类别为 blue(label=0) 且在有效高度范围内时才认为是障碍物
                    if (box.label == 0 &&
                        box_y2 < BZ_Y_UPPER_THRESHOLD && box_y2 > BZ_Y_LOWER_THRESHOLD)
                    {
                        blue_box = box;
                        blue_obstacle_found = true;
                        break;
                    }
                }

                if (blue_obstacle_found) {
                    bz_get = 1;
                    int box_x1 = static_cast<int>(blue_box.rect.x);
                    int box_x2 = static_cast<int>(blue_box.rect.x + blue_box.rect.width);
                    int box_y1 = static_cast<int>(blue_box.rect.y);
                    int box_y2 = static_cast<int>(blue_box.rect.y + blue_box.rect.height);
                    last_known_bz_xcenter = (box_x1 + box_x2) / 2;
                    last_known_bz_x1 = box_x1;
                    last_known_bz_x2 = box_x2;
                    last_known_bz_bottom = box_y2;
                    last_known_bz_heighest = box_y1;
                    bz_disappear_count = 0; // 障碍物可见，重置消失计数
                }

                if (bz_get == 0) {
                    bz_disappear_count++; // 障碍物不可见，累加消失计数
                }

                // 只要在避障状态，就始终使用最后记录的位置进行补线
                bz_heighest = last_known_bz_heighest; // 确保Tracking_bz使用正确的边界
                if (last_known_bz_xcenter > 160) {
                    bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_x1, last_known_bz_bottom), cv::Point(int((right_line[0].x + right_line[1].x + right_line[2].x) / 3), 155), 8);
                } else {
                    bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_x2, last_known_bz_bottom), cv::Point(int((left_line[0].x + left_line[1].x + left_line[2].x) / 3), 155), 8);
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
                        zebra_stop_start_time = std::chrono::steady_clock::now();
                        cout << "[流程] 避障结束，检测到斑马线，准备停车识别" << endl;
                        banma_stop(); // 执行停车
                        system("mpg123 /home/pi/dev_ws/月半猫.mp3 &"); // 播放斑马线提示音（后台播放）
                    }
                }
                else
                {
                    // 状态0: 默认状态，执行障碍物检测以启动避障
                    result = fastestdet_obs->detect(frame);
                    bool obstacle_found_this_frame = false;
                    DetectObject blue_box;

                    if (!result.empty()) {
                        for (const auto& box : result) {
                            int box_y2 = static_cast<int>(box.rect.y + box.rect.height);
                            // 仅当类别为 blue(label=0) 且在有效高度范围内时才认为是障碍物
                            if (box.label == 0 &&
                                box_y2 < BZ_Y_UPPER_THRESHOLD && box_y2 > BZ_Y_LOWER_THRESHOLD) {
                                obstacle_found_this_frame = true;
                                blue_box = box;
                                break;
                            }
                        }
                    }

                    if (obstacle_found_this_frame) {
                        bz_detect_count++;
                        cout << "[避障检测] 发现蓝色障碍物，计数: " << bz_detect_count << "/" << BZ_DETECT_THRESHOLD << endl;
                    } else {
                        if (bz_detect_count > 0) {
                            cout << "[避障检测] 障碍物消失或为非蓝色目标，重置计数" << endl;
                        }
                        bz_detect_count = 0;
                    }

                    if (bz_detect_count >= BZ_DETECT_THRESHOLD) {
                        is_in_avoidance = true; // 启动并锁定避障状态
                        cout << "[流程] 确认蓝色障碍物，进入避障模式（直接变速避障）" << endl;
                        
                        // 记录障碍物的初始位置（仅使用蓝色目标）
                        int box_y2 = static_cast<int>(blue_box.rect.y + blue_box.rect.height);
                        int box_x1 = static_cast<int>(blue_box.rect.x);
                        int box_x2 = static_cast<int>(blue_box.rect.x + blue_box.rect.width);
                        int box_y1 = static_cast<int>(blue_box.rect.y);
                        last_known_bz_xcenter = (box_x1 + box_x2) / 2;
                        last_known_bz_x1 = box_x1;
                        last_known_bz_x2 = box_x2;
                        last_known_bz_bottom = box_y2;
                        last_known_bz_heighest = box_y1;
                        bz_heighest = last_known_bz_heighest;

                        // 立即执行第一次补线和避障巡线
                        if (last_known_bz_xcenter > 160) { 
                            bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_x1, last_known_bz_bottom), cv::Point(int((right_line[0].x + right_line[1].x + right_line[2].x) / 3), 155), 8);
                        } else { 
                            bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_x2, last_known_bz_bottom), cv::Point(int((left_line[0].x + left_line[1].x + left_line[2].x) / 3), 155), 8);
                        }
                        Tracking_bz(bin_image); 
                        
                        bz_detect_count = 0; // 重置计数器，避免重复进入
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
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock); // 确保电机停止
    gpioPWM(servo_pin, servo_pwm_mid);               // 舵机回中
    usleep(100000);                                  // 短暂延时

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
