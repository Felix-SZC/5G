#include <iostream> // 标准输入输出流库
#include <cstdlib> // 标准库


#include <opencv2/opencv.hpp> // OpenCV主头文件
#include <opencv2/core/core.hpp> // OpenCV核心功能
#include <opencv2/highgui.hpp> // OpenCV高层GUI功能
#include <opencv2/imgproc/imgproc_c.h> // OpenCV图像处理功能

#include <string> // 字符串库
#include <pigpio.h> // GPIO控制库
#include <thread> // 线程库
#include <vector> // 向量容器库
#include <algorithm> // 算法库（用于排序）
#include <chrono> // 时间库
#include <iomanip> // 格式化输出
#include <ctime> // 时间格式化
#include <sstream> // 字符串流
#include <sys/stat.h> // 目录操作

#include "fastestdet.hpp" // FastestDet库

using namespace std; // 使用标准命名空间
using namespace cv; // 使用OpenCV命名空间

bool program_finished = false; // 控制主循环退出的标志

//---------------调试选项-------------------------------------------------
const bool SHOW_SOBEL_DEBUG = false; // 是否显示Sobel调试窗口
const int SOBEL_DEBUG_REFRESH_INTERVAL_MS = 120; // 调试窗口刷新间隔，减轻imshow开销

//---------------性能统计---------------------------------------------------
int number = 0; // 已处理帧计数
bool SHOW_FPS = false; // 是否显示FPS信息，可通过命令行参数控制

//------------速度参数配置------------------------------------------------------------------------------------------
const int MOTOR_SPEED_DELTA_CRUISE = 1300;           // 常规巡航速度增量
const int MOTOR_SPEED_DELTA_AVOID = 1100;            // 避障阶段速度增量
const int MOTOR_SPEED_DELTA_POST_ZEBRA = 1300;       // 斑马线后巡线速度增量
const int MOTOR_SPEED_DELTA_CONE_GUIDANCE = 1100;    // 锥桶引导阶段速度增量（有锥桶目标时）
const int MOTOR_SPEED_DELTA_CONE_CRUISE = 1300;      // 锥桶引导后备巡线速度增量（无锥桶目标时）
const int MOTOR_SPEED_DELTA_PARKING_SEARCH = 1300;   // 寻找车库速度增量（未激活A/B识别时）
const int MOTOR_SPEED_DELTA_PARKING_FOLLOW = 1100;   // 车库跟随速度增量（已激活A/B识别时）
const int MOTOR_SPEED_DELTA_PRE_PARKING = 1100;      // 预入库阶段速度增量
const int MOTOR_SPEED_DELTA_BRAKE = -3000;           // 瞬时反转/刹停增量

//------------时间参数配置（单位：秒）------------------------------------------------------------------------------------------
const float BRIEF_STOP_REVERSE_DURATION = 0.5f; // 反转阶段持续时间（秒）
const float BRIEF_STOP_HOLD_DURATION = 0.1f;    // 刹停保持时间（秒）
const float START_DELAY_SECONDS = 2.0f;              // 发车延时时间（秒）
const float ZEBRA_STOP_DURATION_SECONDS = 4.0f;      // 斑马线停车持续时间（秒）
const float POST_ZEBRA_DELAY_SECONDS = 1.0f;        // 斑马线后巡线延迟时间（秒）
const float BANMA_STOP_SLEEP_SECONDS = 0.5f;        // 斑马线停车后的延时（秒）
const float LANE_CHANGE_DURATION_SECONDS = 1.5f;    // 变道持续时间（秒）
const float POST_CONE_CRUISE_DURATION_SECONDS = 3.0f; // 锥桶引导后巡线延迟时间（秒）
const int SERVO_PWM_LEFT_TURN = 780;                // 左转PWM值
const int SERVO_PWM_RIGHT_TURN = 690;               // 右转PWM值
const int MOTOR_SPEED_DELTA_LANE_CHANGE = 1300;     // 变道速度增量

//------------有关的全局变量定义------------------------------------------------------------------------------------------

std::vector<DetectObject> result; // 存储FastestDet检测结果 
std::vector<DetectObject> result_ab; // 存储FastestDet检测结果

enum class CarState {
    Idle,           // 等待发车
    StartDelay,     // 发车延时
    Cruise,         // 正常行驶
    Avoidance,      // 避障
    ZebraStop,      // 在斑马线处等待
    PostZebra,      // 斑马线后恢复
    LaneChange,     // 变道
    ConeGuidance,   // 引导锥桶区域
    ParkingApproachDetection,  // 蓝框（蓝点）识别
    ParkingSearch,  // AB识别和计数
    BriefStopForABActivation,  // 激活AB识别前的停车
    BriefStopForParking,       // 入库前停车（进行AB识别和计数）
    PreParking,     // 预入库（最终接近）
    ParkingComplete // 完成
};

CarState current_state = CarState::Idle;

// 功能: 将CarState枚举转换为可读字符串
std::string carStateToString(CarState state) {
    switch (state) {
        case CarState::Idle:           return "Idle (等待发车)";
        case CarState::StartDelay:     return "StartDelay (发车延时)";
        case CarState::Cruise:         return "Cruise (正常行驶)";
        case CarState::Avoidance:      return "Avoidance (避障)";
        case CarState::ZebraStop:      return "ZebraStop (在斑马线处等待)";
        case CarState::PostZebra:      return "PostZebra (斑马线后恢复)";
        case CarState::LaneChange:     return "LaneChange (变道)";
        case CarState::ConeGuidance:    return "ConeGuidance (引导锥桶区域)";
        case CarState::ParkingApproachDetection: return "ParkingApproachDetection (蓝框识别)";
        case CarState::ParkingSearch:  return "ParkingSearch (AB识别)";
        case CarState::BriefStopForABActivation: return "BriefStopForABActivation (激活AB识别前停车)";
        case CarState::BriefStopForParking:      return "BriefStopForParking (入库前停车)";
        case CarState::PreParking:     return "PreParking (预入库)";
        case CarState::ParkingComplete: return "ParkingComplete (完成)";
        default:                       return "Unknown";
    }
}

// 功能: 设置车辆状态并打印状态变更日志
void setCarState(CarState newState) {
    if (current_state != newState) {
        std::cout << "[状态变更] " << carStateToString(current_state) 
                  << " -> " << carStateToString(newState) << std::endl;

        current_state = newState;
    }
}

// 状态上下文变量
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

//-----------------图像处理与保存----------------------------------------------
Mat frame; // 存储视频帧
Mat bin_image; // 存储二值化图像--Sobel检测后图像
std::chrono::steady_clock::time_point last_save_time; // 上次保存图像的时间
const int SAVE_INTERVAL_SECONDS = 30; // 保存间隔（秒）
const std::string SAVE_DIR = "captured_images"; // 保存目录
const int MIN_COMPONENT_AREA = 400; // 连通区域最小面积阈值（用于过滤噪声）

//---------------蓝色挡板发车相关----------------------------------------------
int find_first = 0; // 标记是否第一次找到蓝色挡板
int blue_detect_count = 0; // 蓝色挡板连续检测计数
const int BLUE_DETECT_THRESHOLD = 5; // 需要连续检测到的帧数才能确认找到蓝色挡板

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
std::chrono::steady_clock::time_point start_delay_time; // 挡板移开时间戳

//-----------------巡线相关-----------------------------------------------
std::vector<cv::Point> mid; // 存储中线
std::vector<cv::Point> left_line; // 存储左线条
std::vector<cv::Point> right_line; // 存储右线条
std::vector<cv::Point> last_mid; // 存储上一次的中线，用于平滑滤波

int error_first; // 存储第一次误差
int last_error;
float servo_pwm_diff; // 存储舵机PWM差值
float servo_pwm; // 存储舵机PWM值

//----------------避障相关---------------------------------------------------
enum class AvoidanceDirection {
    Auto,        // 自动判断: 障碍物在右，向左避；障碍物在左，向右避
    ForceLeft,   // 强制向左避障
    ForceRight   // 强制向右避障
};
const AvoidanceDirection AVOIDANCE_STRATEGY = AvoidanceDirection::Auto; // 在此配置避障策略

int bz_heighest = 0; // 避障高度
int bz_get = 0;
std::vector<cv::Point> mid_bz; // 存储中线
std::vector<cv::Point> left_line_bz; // 存储左线条
std::vector<cv::Point> right_line_bz; // 存储右线条
std::vector<cv::Point> last_mid_bz; // 存储上一帧避障中线
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

//---------------斑马线检测相关------------------------------------------
int banma = 0; // 斑马线检测结果
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


const int BANMA_MIN_COUNT = 4;  // 判定为斑马线需要的最少白色矩形数量 

//----------------变道相关---------------------------------------------------
int flag_turn_done = 0; // 转向完成标志
std::chrono::steady_clock::time_point zebra_stop_start_time;
std::chrono::steady_clock::time_point post_zebra_delay_start_time; // 斑马线后延迟计时器
std::chrono::steady_clock::time_point lane_change_start_time;      // 变道计时器
std::chrono::steady_clock::time_point post_cone_cruise_start_time; // 锥桶引导后巡线延迟计时器
int turn_signal_label = -1;                                        // 转向标志 (0=left, 1=right)

// ----------------锥桶引导相关---------------------------------------------------
int cone_outer_color = 1; // 0=蓝色为外侧边界, 1=黄色为外侧边界
const int CONE_ENTER_THRESHOLD = 5; // 确认锥桶出现的帧数阈值
const int CONE_BOTTOM_Y_THRESHOLD = 140; // 进入锥桶引导的底部高度阈值
const int CONE_EXIT_THRESHOLD = 3; // 确认锥桶消失的帧数阈值
bool has_seen_cones = false; // 是否已确认进入锥桶引导模式
int cones_detect_count = 0; // 锥桶连续检测计数
int cones_lost_count = 0; // 锥桶连续丢失计数
int cone_target_x = -1; // 锥桶引导目标点X坐标（-1表示未检测到）
bool need_post_cone_cruise_delay = false; // 是否需要锥桶引导后巡线延迟
bool from_cone_guidance = false; // 是否从锥桶引导来的（用于区分处理逻辑）
std::vector<int> lane_widths; // 存储不同高度对应的赛道宽度

//---------------入库相关------------------------------------------

// 新增：入库前蓝点引导相关
const int PARKING_APPROACH_ROI_X = 10;
const int PARKING_APPROACH_ROI_Y = 85;
const int PARKING_APPROACH_ROI_WIDTH = 300;
const int PARKING_APPROACH_ROI_HEIGHT = 100;
const double PARKING_APPROACH_BLUE_AREA_MIN = 30.0; // 判定为接近车库需要的最小蓝色面积
const int PARKING_APPROACH_CONFIRM_THRESHOLD = 2; // 连续检测到N帧才确认
int parking_approach_detect_count = 0; // 蓝点接近区域连续检测计数

int latest_park_id = 0; // 最近检测到的车库ID (1=A, 2=B)
int park_A_count = 0; // A车库累计识别次数
int park_B_count = 0; // B车库累计识别次数
int turn_left_count = 0;    // 斑马线左转累计识别次数
int turn_right_count = 0;   // 斑马线右转累计识别次数
const int PARKING_Y_THRESHOLD = 120; // 触发入库的Y轴阈值
int final_target_label = -1;       // 最终锁定的AB标志的标签（0表示A，1表示B）
std::chrono::steady_clock::time_point pre_parking_start_time; // 预入库阶段开始时间
const int PARKING_DETECT_MISS_THRESHOLD = 5; // 检测不到的帧数阈值，达到后停车

int parking_target_not_detected_count = 0; // 连续检测不到目标的帧数
int parking_follow_x = 160; // 当前跟随目标的x坐标（默认中心）
bool parking_target_detected_this_frame = false; // 当前帧是否检测到目标

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
cv::Mat ImageSobel(cv::Mat &frame, CarState state, cv::Mat *debugOverlay = nullptr) 
{
    const cv::Size targetSize(320, 240);
    cv::Mat resizedFrame;
    if (frame.size() != targetSize)
    {
        cv::resize(frame, resizedFrame, targetSize);
    }
    else
    {
        resizedFrame = frame; // 无需克隆，后续操作不会修改原始图像
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
    cv::addWeighted(absSobelY, 1.0, absSobelX, 1.0, 0, gradientMagnitude8U);

    // 顶帽操作减弱阴影
    cv::Mat topHat;
    static cv::Mat kernel_tophat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 3));
    cv::morphologyEx(blurredRoi, topHat, cv::MORPH_TOPHAT, kernel_tophat);

    cv::Mat adaptiveMask;
    cv::threshold(topHat, adaptiveMask, 5, 255, cv::THRESH_BINARY);

    cv::Mat gradientMask;
    cv::threshold(gradientMagnitude8U, gradientMask, 30, 255, cv::THRESH_BINARY); // 梯度二值掩码
    static cv::Mat kernel_gradient_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(gradientMask, gradientMask, kernel_gradient_dilate);

    cv::Mat binaryMask;
    cv::bitwise_and(adaptiveMask, gradientMask, binaryMask); // 亮度+梯度联合约束

    cv::medianBlur(binaryMask, binaryMask, 3); // 中值去椒盐噪声
    // cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_OPEN, noiseKernel); // 小结构开运算 - 1x1内核无效，已移除

    // 先进行连通域分析，过滤掉不符合面积要求的连通域
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binaryMask, labels, stats, centroids, 8, CV_32S); // 连通域分析
    cv::Mat filteredByArea = cv::Mat::zeros(binaryMask.size(), CV_8U);
    for (int i = 1; i < numLabels; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= MIN_COMPONENT_AREA)
        {
            filteredByArea.setTo(255, labels == i);
        }
    }

    // 在面积过滤后的结果上执行形态学操作，避免binaryMask.clone()的开销
    static cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 9)); // 闭运算连接断裂
    cv::morphologyEx(filteredByArea, filteredByArea, cv::MORPH_CLOSE, kernel_close);
    static cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)); // 膨胀加粗车道线
    cv::dilate(filteredByArea, filteredByArea, kernel_dilate, cv::Point(-1, -1), 1);

    cv::Mat filteredMorph = filteredByArea;

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(filteredMorph, lines, 1, CV_PI / 180, 8, 10, 12);

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

        float angle_threshold = 5.0f;
        if (state == CarState::Cruise || state == CarState::Avoidance)
        {
            angle_threshold = 15.0f;
        }
        else if (state == CarState::ConeGuidance && has_seen_cones)
        {
            angle_threshold = 60.0f;
        }

        if (std::abs(angle) > angle_threshold && length > 8)
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

// 功能: 计算锥桶引导目标点
// 策略: 
// 1. 如果蓝黄锥桶都被检测到，则分别取最近的1-2个计算中心，再取中心的中点作为目标。
// 2. 如果只有单边（蓝或黄），则使用该侧所有锥桶的平均位置，再根据内外侧设置进行偏移，计算虚拟中点。
// 参数: objects - 检测到的锥桶对象列表
//      target_x - 输出的目标点X坐标
//      cone_outer_color - 配置变量，指定外侧边界颜色（0=蓝色为外侧边界, 1=黄色为外侧边界）
//      last_turn_signal - 之前的转向动作（0=左转/左道，1=右转/右道）
// 返回: 是否成功计算出目标点
bool calculate_cone_target(const std::vector<DetectObject>& objects, int& target_x, int cone_outer_color, int last_turn_signal) {
    // 如果单侧锥桶数量小于等于2，则该侧无效

    // 分别收集blue和yellow锥桶
    std::vector<DetectObject> blue_cones;
    std::vector<DetectObject> yellow_cones;

    for (const auto& obj : objects) {
        if (obj.label == 0) { // Blue
            blue_cones.push_back(obj);
        } else if (obj.label == 1) { // Yellow
            yellow_cones.push_back(obj);
        }
    }

    // 辅助函数：按y2（底部y坐标）降序排序的比较函数
    auto compare_by_y2 = [](const DetectObject& a, const DetectObject& b) {
        float y2_a = a.rect.y + a.rect.height;
        float y2_b = b.rect.y + b.rect.height;
        return y2_a > y2_b; // 降序：y2大的在前（更近的在前）
    };

    // 对锥桶列表进行排序
    if (!blue_cones.empty()) {
        std::sort(blue_cones.begin(), blue_cones.end(), compare_by_y2);
    }
    if (!yellow_cones.empty()) {
        std::sort(yellow_cones.begin(), yellow_cones.end(), compare_by_y2);
    }

    bool has_blue = blue_cones.size() > 2;
    bool has_yellow = yellow_cones.size() > 2;

    // 3. 根据锥桶分布情况计算目标点
    if (has_blue && has_yellow) {
        // --- 情况一: 双边都识别到 ---
        // 策略: 只使用每侧最近的最多3个锥桶
        float avg_blue = 0.0f;
        int take_count_blue = std::min(3, (int)blue_cones.size());
        float sum_blue_x = 0.0f;
        for (int i = 0; i < take_count_blue; i++) {
            sum_blue_x += blue_cones[i].rect.x + blue_cones[i].rect.width / 2.0f;
        }
        avg_blue = sum_blue_x / take_count_blue;

        float avg_yellow = 0.0f;
        int take_count_yellow = std::min(3, (int)yellow_cones.size());
        float sum_yellow_x = 0.0f;
        for (int i = 0; i < take_count_yellow; i++) {
            sum_yellow_x += yellow_cones[i].rect.x + yellow_cones[i].rect.width / 2.0f;
        }
        avg_yellow = sum_yellow_x / take_count_yellow;
        
        // 目标点取两侧中心位置的平均值，作为车道中心
        target_x = static_cast<int>((avg_blue + avg_yellow) / 2.0f);
        return true;

    } else if (has_blue) {
        // --- 情况二: 只有蓝色锥桶 ---
        // 策略: 使用所有检测到的蓝色锥桶计算平均位置和平均底部高度
        float sum_blue_x = 0.0f;
        float sum_blue_y = 0.0f;
        for (const auto& cone : blue_cones) {
            sum_blue_x += cone.rect.x + cone.rect.width / 2.0f;
            sum_blue_y += cone.rect.y + cone.rect.height;
        }
        float avg_blue_x = sum_blue_x / blue_cones.size();
        int avg_blue_y = static_cast<int>(sum_blue_y / blue_cones.size());
        
        // 越界保护
        if (avg_blue_y < 0) avg_blue_y = 0;
        if (avg_blue_y >= lane_widths.size()) avg_blue_y = lane_widths.size() - 1;
        int dynamic_offset = lane_widths[avg_blue_y] / 2;

        // 判断蓝色锥桶在左侧还是右侧
        bool is_blue_left = (cone_outer_color == 0 && last_turn_signal == 0) || 
                            (cone_outer_color == 1 && last_turn_signal == 1);

        if (is_blue_left) {
            // 蓝色在左侧：车道中心在蓝色右侧，目标点 = 蓝色位置 + 偏移量
            target_x = static_cast<int>(avg_blue_x) + dynamic_offset;
        } else {
            // 蓝色在右侧：车道中心在蓝色左侧，目标点 = 蓝色位置 - 偏移量
            target_x = static_cast<int>(avg_blue_x) - dynamic_offset;
        }
        
    } else if (has_yellow) {
        // --- 情况三: 只有黄色锥桶 ---
        // 策略: 使用所有检测到的黄色锥桶计算平均位置和平均底部高度
        float sum_yellow_x = 0.0f;
        float sum_yellow_y = 0.0f;
        for (const auto& cone : yellow_cones) {
            sum_yellow_x += cone.rect.x + cone.rect.width / 2.0f;
            sum_yellow_y += cone.rect.y + cone.rect.height;
        }
        float avg_yellow_x = sum_yellow_x / yellow_cones.size();
        int avg_yellow_y = static_cast<int>(sum_yellow_y / yellow_cones.size());

        // 越界保护
        if (avg_yellow_y < 0) avg_yellow_y = 0;
        if (avg_yellow_y >= lane_widths.size()) avg_yellow_y = lane_widths.size() - 1;
        int dynamic_offset = lane_widths[avg_yellow_y] / 2;

        // 判断黄色锥桶在左侧还是右侧
        bool is_yellow_left = (cone_outer_color == 1 && last_turn_signal == 0) || 
                              (cone_outer_color == 0 && last_turn_signal == 1);

        if (is_yellow_left) {
            // 黄色在左侧：车道中心在黄色右侧，目标点 = 黄色位置 + 偏移量
            target_x = static_cast<int>(avg_yellow_x) + dynamic_offset;
        } else {
            // 黄色在右侧：车道中心在黄色左侧，目标点 = 黄色位置 - 偏移量
            target_x = static_cast<int>(avg_yellow_x) - dynamic_offset;
        }

    } else {
        // --- 情况四: 未检测到任何锥桶 ---
        return false;
    }

    // 对单边情况计算出的目标点进行边界保护
    if (target_x > 320) target_x = 320; 
    if (target_x < 0) target_x = 0;
    return true;
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
        setCarState(CarState::StartDelay);
        start_delay_time = std::chrono::steady_clock::now();
        // 挡板移开后开始计时，延时指定时间再允许控制函数运行
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


// 功能: 检测入库前引导区域（通过最大蓝色面积）
bool detect_parking_approach_points(cv::Mat &frame) {
    // 1. 裁剪ROI
    int roiWidth = std::min(PARKING_APPROACH_ROI_WIDTH, frame.cols - PARKING_APPROACH_ROI_X);
    int roiHeight = std::min(PARKING_APPROACH_ROI_HEIGHT, frame.rows - PARKING_APPROACH_ROI_Y);
    if (roiWidth <= 0 || roiHeight <= 0) {
        return false; // ROI无效
    }
    cv::Rect roiRect(PARKING_APPROACH_ROI_X, PARKING_APPROACH_ROI_Y, roiWidth, roiHeight);
    cv::Mat roiFrame = frame(roiRect);

    // 2. 转换到HSV并进行阈值分割
    cv::Mat hsvRoi;
    cv::cvtColor(roiFrame, hsvRoi, cv::COLOR_BGR2HSV);
    
    cv::Mat mask;
    Scalar scalarl = Scalar(BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN);
    Scalar scalarH = Scalar(BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX);
    inRange(hsvRoi, scalarl, scalarH, mask);

    // 3. 形态学闭运算连接区域，消除白色箭头/字母分割造成的影响
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, closeKernel, Point(-1, -1), 3);
    
    // 4. 查找轮廓并找出最大面积
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return false;
    }

    // 按面积排序并获取最大面积
    sort(contours.begin(), contours.end(), Contour_Area);
    double max_area = contourArea(contours[0]);

    // 5. 最终判定
    if (max_area >= PARKING_APPROACH_BLUE_AREA_MIN) {
        cout << "检测到入库引导区域 (最大蓝色面积: " << (int)max_area << ")" << endl;
        return true;
    }
    
    return false;
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
    float kd = 3.5; // 微分系数

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

// 功能: 斑马线后巡线PD控制器，基于中线偏差计算舵机PWM
float servo_pd_post_zebra(int target) { // 赛道巡线控制

    // 安全检查：确保mid向量有足够的元素
    if (mid.size() < 26) {
        cerr << "[警告] servo_pd_post_zebra: mid向量元素不足 (" << mid.size() << " < 26)，返回中值" << endl;
        return servo_pwm_mid;
    }

    int pidx = int((mid[23].x + mid[25].x) / 2); // 计算中线中点的x坐标

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
    else if (servo_pwm < 580) // 如果PWM值小于600
    {
        servo_pwm = 580; // 限制PWM值为600
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

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); 

    last_error = error_first; // 更新上一次误差

    servo_pwm = servo_pwm_mid + servo_pwm_diff; // 计算舵机PWM值

    return servo_pwm; // 返回舵机PWM值
}



// 功能: 锥桶引导PD控制器
float servo_pd_cone(int target_x) {
    int target = 160; // 图像中心作为目标
    int pidx = target_x; // 当前锥桶引导计算出的目标点（实际上是我们希望车头对准的点）
    // 注意：calculate_cone_target返回的是"车道中心在图像中的X坐标"
    // 所以这里的error应该是 target(160) - pidx(lane_center_x)
    // 如果 lane_center_x > 160 (在右边)，error < 0，需要右转（根据舵机方向调整）
    
    // 使用与常规巡线相似的参数
    float kp = 0.6; 
    float kd = 3.5; 

    error_first = target - pidx; 

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); 

    last_error = error_first; 

    servo_pwm = servo_pwm_mid + servo_pwm_diff; 

    // 限制幅值
    if (servo_pwm > 1000) servo_pwm = 1000;
    else if (servo_pwm < 580) servo_pwm = 580;
    
    return servo_pwm; 
}

// 功能: 寻找车库阶段的巡线PD控制器（参数更平缓以适应低帧率）
float servo_pd_parking_cruise(int target) { 
    if (mid.size() < 26) {
        return servo_pwm_mid;
    }
    int pidx = int((mid[23].x + mid[25].x) / 2);

    float kp = 0.6; 
    float kd = 3.5; 

    error_first = target - pidx;
    servo_pwm_diff = kp * error_first + kd * (error_first - last_error);
    last_error = error_first;
    servo_pwm = servo_pwm_mid + servo_pwm_diff;

    if (servo_pwm > 1000) servo_pwm = 1000;
    else if (servo_pwm < 580) servo_pwm = 580;
    
    return servo_pwm;
}

// 功能: 锥桶引导阶段的后备巡线PD控制器（参数更平缓以适应低帧率）
float servo_pd_cone_cruise(int target) { 
    if (mid.size() < 26) {
        return servo_pwm_mid;
    }
    int pidx = int((mid[23].x + mid[25].x) / 2);

    float kp = 0.6; 
    float kd = 3.5; 

    error_first = target - pidx;
    servo_pwm_diff = kp * error_first + kd * (error_first - last_error);
    last_error = error_first;
    servo_pwm = servo_pwm_mid + servo_pwm_diff;

    if (servo_pwm > 1000) servo_pwm = 1000;
    else if (servo_pwm < 580) servo_pwm = 580;
    
    return servo_pwm;
}


// 功能: 斑马线触发停车：电机回中、舵机回中并输出日志
void banma_stop(){
    gpioPWM(servo_pin, servo_pwm_mid);
    gpioPWM(servo_pin, servo_pwm_mid);
    gpioPWM(servo_pin, servo_pwm_mid);
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock - 5000); // 解锁状态，即停车
    gpioPWM(servo_pin, servo_pwm_mid);
    gpioPWM(servo_pin, servo_pwm_mid);
    gpioPWM(servo_pin, servo_pwm_mid);
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long long>(BANMA_STOP_SLEEP_SECONDS * 1000000))); // 转换为微秒
    cout << "[流程] 检测到斑马线，车辆停车" << static_cast<int>(ZEBRA_STOP_DURATION_SECONDS) << "秒等待指令" << endl;
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


// 功能: 根据A/B车库累计计数决定最终目标标签
int decide_parking_label_from_counts()
{
    if (park_A_count > park_B_count) return 0;
    if (park_B_count > park_A_count) return 1;
    if (latest_park_id != 0) return latest_park_id - 1;
    return -1;
}


// 控制舵机电机
// 功能: 根据状态机切换控制策略（巡线/避障/停车），并下发PWM
void motor_servo_contral()
{
    float servo_pwm_now = servo_pwm_mid;

    switch (current_state)
    {
        case CarState::Idle:
        case CarState::ZebraStop:
        case CarState::ParkingComplete:
            // 停车状态，不做主动控制（由相关逻辑或默认状态保持电机和舵机）
            return;

        case CarState::StartDelay:
            // 发车延时阶段：保持电机停止、舵机回中
            gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock);
            gpioPWM(servo_pin, servo_pwm_mid);
            return;

        case CarState::BriefStopForABActivation:
        case CarState::BriefStopForParking:
            {
                // 计算短暂停车已持续的时间
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - brief_stop_start_time).count() / 1000.0f;

                if (elapsed < BRIEF_STOP_REVERSE_DURATION)
                {
                    gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_BRAKE); // 瞬时反转阶段
                }
                else
                {
                    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock); // 刹停保持阶段
                }
                // 舵机保持中值
                gpioPWM(servo_pin, servo_pwm_mid);
            }
            return;

        case CarState::PreParking:
            // 状态：预入库阶段 - 跟随最远的A或B目标
            servo_pwm_now = servo_pd_parking(parking_follow_x);
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_PRE_PARKING);
            break;

        case CarState::ParkingApproachDetection:
            // 状态：蓝框识别，使用寻找车库速度
            servo_pwm_now = servo_pd_parking_cruise(160);
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_PARKING_SEARCH);
            break;

        case CarState::ParkingSearch:
            // 状态：AB识别和计数，使用慢速前进
            if (latest_park_id != 0) {
                // 已识别到车库，跟随最近的A或B标识
                servo_pwm_now = servo_pd_parking(parking_follow_x); 
            } else {
                // 未识别到车库，平缓巡线
                servo_pwm_now = servo_pd_parking_cruise(160);
            }
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_PARKING_FOLLOW);
            break;

        case CarState::Avoidance:
            // 状态：正在执行避障
            servo_pwm_now = servo_pd_bz(160); // 使用为避障优化的PD控制
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_AVOID); // 避障时使用较慢速度
            break;

        case CarState::LaneChange:
            // 状态：盲跑变道
            if (turn_signal_label == 0) { // Left
                servo_pwm_now = SERVO_PWM_LEFT_TURN;
            } else if (turn_signal_label == 1) { // Right
                servo_pwm_now = SERVO_PWM_RIGHT_TURN;
            } else {
                servo_pwm_now = servo_pwm_mid; 
            }
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_LANE_CHANGE);
            break;

        case CarState::ConeGuidance:
            // 状态：锥桶引导
            if (cone_target_x != -1) {
                // 如果计算出了锥桶目标，使用专门的锥桶PD控制
                servo_pwm_now = servo_pd_cone(cone_target_x);
                gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_CONE_GUIDANCE);
            } else {
                // 否则使用为锥桶引导优化的平缓巡线参数
                servo_pwm_now = servo_pd_cone_cruise(160);
                gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_CONE_CRUISE);

            }
            break;

        case CarState::Cruise:
            // 状态：常规巡线（包括寻找斑马线或避障间隙）
            servo_pwm_now = servo_pd(160); // 使用常规PD控制
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_CRUISE); // 使用常规速度
            break;

        case CarState::PostZebra:
            // 状态：常规巡线（包括寻找斑马线或避障间隙）
            servo_pwm_now = servo_pd_post_zebra(160); // 使用常规PD控制
            gpioPWM(motor_pin, motor_pwm_mid + MOTOR_SPEED_DELTA_POST_ZEBRA); // 使用斑马线后巡线速度
            break;
            
        default:
            // 默认状态：停止电机和舵机
            gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock);
            gpioPWM(servo_pin, servo_pwm_mid);
            return;
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
        fastestdet_obs = new FastestDet(model_param_obs, model_bin_obs, num_classes_obs, labels_obs, 352, 0.5f, 0.5f, 4, false);
        cout << "[初始化] 障碍物检测模型加载成功!" << endl;
    } catch (const std::exception& e) {
        cerr << "[错误] 障碍物检测模型加载失败: " << e.what() << endl;
        return -1;
    }

    cout << "[初始化] 加载转向标志检测模型..." << endl;
    try {
        fastestdet_lr = new FastestDet(model_param_lr, model_bin_lr, num_classes_lr, labels_lr, 352, 0.4f, 0.4f, 4, false);
        cout << "[初始化] 转向标志检测模型加载成功!" << endl;
    } catch (const std::exception& e) {
        cerr << "[错误] 转向标志检测模型加载失败: " << e.what() << endl;
        delete fastestdet_obs;
        return -1;
    }

    cout << "[初始化] 加载车库检测模型..." << endl;
    try {
        fastestdet_ab = new FastestDet(model_param_ab, model_bin_ab, num_classes_ab, labels_ab, 352, 0.7f, 0.7f, 4, false);
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
    initialize_lane_widths();  // 初始化赛道宽度查找表

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
        cin.ignore(); // 等待用户输入
        return -1;    // 返回错误代码
    }

    // 输出摄像头的属性
    cout << "摄像头帧率: " << capture.get(cv::CAP_PROP_FPS) << endl;
    cout << "摄像头宽度: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << endl;
    cout << "摄像头高度: " << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;
    //---------------------------------------------------

    auto lastDebugRefresh = std::chrono::steady_clock::now();
    cv::Mat lastDebugOverlay;
    
    // 初始化图像保存时间（设置为过去时间，确保发车后第一次检查就会立即保存）
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
            
            // 2. 状态更新与预处理
            
            // 特殊状态的时间检查与自动状态跃迁
            if (current_state == CarState::StartDelay) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed_sec = std::chrono::duration_cast<std::chrono::microseconds>(now - start_delay_time).count() / 1000000.0;
                if (elapsed_sec >= START_DELAY_SECONDS) {
                    cout << "[流程] 发车延时结束，开始巡航" << endl;
                    setCarState(CarState::Cruise);
                }
            }

            // 图像处理（空闲状态单独处理，其他驾驶状态统一计算车道线）
            if (current_state == CarState::Idle) {
                if (find_first == 0) {
                    blue_card_find();
                } else {
                    blue_card_remove();
                }
            }
            else if (current_state != CarState::ParkingComplete && current_state != CarState::BriefStopForABActivation && current_state != CarState::BriefStopForParking && current_state != CarState::PreParking) 
            {
                // 发车后的状态，累加帧计数
                number++; 
                // 计算车道线（适用于巡航、避障、斑马线停车、斑马线后、寻找车库等状态）
                const auto now = std::chrono::steady_clock::now();
                const bool shouldRefreshDebug = SHOW_SOBEL_DEBUG &&
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - lastDebugRefresh).count() >= SOBEL_DEBUG_REFRESH_INTERVAL_MS;

                cv::Mat debugOverlay;
                cv::Mat* debugPtr = (SHOW_SOBEL_DEBUG && shouldRefreshDebug) ? &debugOverlay : nullptr;
                
                bin_image = ImageSobel(frame, current_state, debugPtr);

                // 可选：显示调试图像
                if (SHOW_SOBEL_DEBUG && shouldRefreshDebug)
                {
                    if (!debugOverlay.empty()) lastDebugOverlay = debugOverlay;
                    if (!lastDebugOverlay.empty()) cv::imshow("TrackLine Overlay", lastDebugOverlay);
                    lastDebugRefresh = now;
                }
                if (SHOW_SOBEL_DEBUG) cv::waitKey(1);
            }

            // 3. 各状态具体逻辑处理
            switch (current_state)
            {
            case CarState::Idle:
            case CarState::StartDelay:
                // 已在预处理阶段处理，此处无需额外操作
                break;

            case CarState::Cruise:
                Tracking(bin_image);

                // 检查是否检测到斑马线
                if (count_bz >= 1 && flag_turn_done == 0)
                {
                    banma = banma_get(frame);
                    if (banma == 1) {
                        cout << "[流程] 避障结束，检测到斑马线，准备停车识别" << endl;
                        banma_stop(); 
                        system("mpg123 /home/pi/dev_ws/月半猫.mp3 &");
                        zebra_stop_start_time = std::chrono::steady_clock::now();
                        turn_left_count = 0;
                        turn_right_count = 0;
                        turn_signal_label = -1;
                        setCarState(CarState::ZebraStop);
                    }
                }
                else // 检查障碍物
                {
                    result = fastestdet_obs->detect(frame);
                    bool obstacle_found_this_frame = false;
                    DetectObject blue_box;

                    if (!result.empty()) {
                        // 查找蓝色障碍物（label=0，且在有效高度范围内）
                        for (const auto& box : result) {
                            int box_y2 = static_cast<int>(box.rect.y + box.rect.height);
                            if (box.label == 0 && box.prob > 0.6f &&
                                box_y2 < BZ_Y_UPPER_THRESHOLD && box_y2 > BZ_Y_LOWER_THRESHOLD) {
                                obstacle_found_this_frame = true;
                                blue_box = box;
                                break;
                            }
                        }
                    }

                    if (obstacle_found_this_frame) {
                        // 检测到障碍物，累加计数
                        bz_detect_count++;
                        cout << "[避障检测] 发现蓝色障碍物，计数: " << bz_detect_count << "/" << BZ_DETECT_THRESHOLD << endl;
                    } else {
                        // 未检测到障碍物，重置计数
                        if (bz_detect_count > 0) cout << "[避障检测] 障碍物消失，重置计数" << endl;
                        bz_detect_count = 0;
                    }

                    // 连续检测到足够帧数，确认障碍物并进入避障模式
                    if (bz_detect_count >= BZ_DETECT_THRESHOLD) {
                        cout << "[流程] 确认蓝色障碍物，进入避障模式" << endl;
                        setCarState(CarState::Avoidance);
                        
                        // 初始化避障相关参数
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

                        // 执行第一次补线操作
                        // 根据避障策略决定补线方向
                        bool avoid_left;
                        switch (AVOIDANCE_STRATEGY) {
                            case AvoidanceDirection::ForceLeft:
                                avoid_left = true;
                                break;
                            case AvoidanceDirection::ForceRight:
                                avoid_left = false;
                                break;
                            case AvoidanceDirection::Auto:
                            default:
                                // 自动模式：基于障碍物位置判断
                                avoid_left = (last_known_bz_xcenter > 160); 
                                break;
                        }

                        // 执行补线
                        if (avoid_left) {
                            // 向左避障: 在障碍物左侧补线，引导车辆向左
                            bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_x1, last_known_bz_bottom), cv::Point(int((right_line[0].x + right_line[1].x + right_line[2].x) / 3), 155), 8);
                        } else {
                            // 向右避障: 在障碍物右侧补线，引导车辆向右
                            bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_x2, last_known_bz_bottom), cv::Point(int((left_line[0].x + left_line[1].x + left_line[2].x) / 3), 155), 8);
                        }
                        Tracking_bz(bin_image); 
                        bz_detect_count = 0; // 重置计数器，避免重复进入
                    }
                }
                break;

            case CarState::Avoidance:
                Tracking(bin_image); // 常规巡线作为参考
                
                bz_get = 0;
                result = fastestdet_obs->detect(frame);
                {
                    DetectObject blue_box;
                    bool blue_obstacle_found = false;
                    // 查找蓝色障碍物（label=0，且在有效高度范围内）
                    for (const auto& box : result) {
                        int box_y2 = static_cast<int>(box.rect.y + box.rect.height);
                        if (box.label == 0 && box.prob > 0.6f && box_y2 < BZ_Y_UPPER_THRESHOLD && box_y2 > BZ_Y_LOWER_THRESHOLD) {
                            blue_box = box;
                            blue_obstacle_found = true;
                            break;
                        }
                    }

                    if (blue_obstacle_found) {
                        // 更新障碍物位置信息
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
                    } else {
                        bz_disappear_count++; // 障碍物不可见，累加消失计数
                    }

                    // 使用最后记录的障碍物位置进行补线
                    bz_heighest = last_known_bz_heighest;
                    // 根据避障策略决定补线方向
                    bool avoid_left;
                    switch (AVOIDANCE_STRATEGY) {
                        case AvoidanceDirection::ForceLeft:
                            avoid_left = true;
                            break;
                        case AvoidanceDirection::ForceRight:
                            avoid_left = false;
                            break;
                        case AvoidanceDirection::Auto:
                        default:
                            // 自动模式：基于障碍物位置判断
                            avoid_left = (last_known_bz_xcenter > 160); 
                            break;
                    }

                    // 执行补线
                    if (avoid_left) {
                        // 向左避障: 在障碍物左侧补线，引导车辆向左
                        bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_x1, last_known_bz_bottom), cv::Point(int((right_line[0].x + right_line[1].x + right_line[2].x) / 3), 155), 8);
                    } else {
                        // 向右避障: 在障碍物右侧补线，引导车辆向右
                        bin_image = drawWhiteLine(bin_image, cv::Point(last_known_bz_x2, last_known_bz_bottom), cv::Point(int((left_line[0].x + left_line[1].x + left_line[2].x) / 3), 155), 8);
                    }
                    Tracking_bz(bin_image);

                    // 检查是否满足退出避障的条件
                    if (bz_disappear_count >= BZ_DISAPPEAR_THRESHOLD) {
                        cout << "[流程] 障碍物已安全绕过，退出避障模式" << endl;
                        count_bz++;
                        bz_disappear_count = 0;
                        setCarState(CarState::Cruise);
                    }
                }
                break;

            case CarState::ZebraStop:
                {
                    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - zebra_stop_start_time).count() / 1000000.0;
                    if (elapsed < ZEBRA_STOP_DURATION_SECONDS) {
                        // 在停车时间内，持续检测并计数转向标志
                        result.clear();
                        result = fastestdet_lr->detect(frame);
                        if (!result.empty()) {
                            for (const auto& obj : result) {
                                if (obj.label == 0) { // left
                                    turn_left_count++;
                                } else if (obj.label == 1) { // right
                                    turn_right_count++;
                                }
                            }
                        }
                        // 为了调试，可以打印计数值
                        cout << "\r[Zebra] 计数中... 左: " << turn_left_count << ", 右: " << turn_right_count << "  " << std::flush;
                    } else {
                        // 停车时间结束，根据计数值决定转向
                        cout << endl; // 换行
                        if (turn_left_count > turn_right_count) {
                            turn_signal_label = 0; // Left
                            cout << "[流程] 转向决策: 左转 (L:" << turn_left_count << " > R:" << turn_right_count << ")" << endl;
                        } else if (turn_right_count > turn_left_count) {
                            turn_signal_label = 1; // Right
                            cout << "[流程] 转向决策: 右转 (R:" << turn_right_count << " > L:" << turn_left_count << ")" << endl;
                        } else {
                            turn_signal_label = -1; // No turn if counts are equal or zero
                            cout << "[流程] 转向决策: 直行 (计数 L:" << turn_left_count << ", R:" << turn_right_count << ")" << endl;
                        }

                        // 无论是否识别到转向标识，都继续巡线
                        cout << "[流程] 停车结束，开始" << static_cast<int>(POST_ZEBRA_DELAY_SECONDS) << "秒常规巡线..." << endl;
                        flag_turn_done = 1;
                        post_zebra_delay_start_time = std::chrono::steady_clock::now();
                        setCarState(CarState::PostZebra);
                    }
                }
                break;

            case CarState::PostZebra:
                Tracking(bin_image); // 正常巡线
                {
                    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - post_zebra_delay_start_time).count() / 1000000.0;
                    if (elapsed >= POST_ZEBRA_DELAY_SECONDS) {
                        // 延迟时间结束，根据是否有转向标志决定下一步
                        if (turn_signal_label != -1) {
                             cout << "[流程] " << static_cast<int>(POST_ZEBRA_DELAY_SECONDS) << "秒巡线结束，开始变道 -> " << (turn_signal_label == 0 ? "左" : "右") << endl;
                             lane_change_start_time = std::chrono::steady_clock::now();
                             setCarState(CarState::LaneChange);
                        } else {
                             cout << "[流程] " << static_cast<int>(POST_ZEBRA_DELAY_SECONDS) << "秒巡线结束，未检测到转向标志，开始蓝框识别" << endl;
                             parking_approach_detect_count = 0;
                             from_cone_guidance = false; // 确保标志为false（从PostZebra路径来）
                             setCarState(CarState::ParkingApproachDetection);
                        }
                    }
                }
                break;

            case CarState::LaneChange:
                // 盲跑变道，不进行视觉处理（或者仅做记录），依靠时间控制
                {
                    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - lane_change_start_time).count() / 1000000.0;
                    if (elapsed >= LANE_CHANGE_DURATION_SECONDS) {
                        cout << "[流程] 变道结束，进入锥桶引导区域" << endl;
                        setCarState(CarState::ConeGuidance);
                        has_seen_cones = false;
                        cones_detect_count = 0; // 重置检测计数器
                        cones_lost_count = 0;
                        cone_target_x = -1;
                    }
                }
                break;

            case CarState::ConeGuidance:
                // 1. 常规巡线 (Fallback)
                Tracking(bin_image);
                
                // 2. 检测锥桶
                result = fastestdet_obs->detect(frame);
                
                // 3. 计算目标并更新状态
                {
                    int target_tmp = 0;
                    bool cone_found_this_frame = calculate_cone_target(result, target_tmp, cone_outer_color, turn_signal_label);
                    float max_cone_bottom_y = 0;

                    // 查找最下方锥桶的底部Y坐标
                    if (cone_found_this_frame) {
                        for (const auto& obj : result) {
                            float bottom_y = obj.rect.y + obj.rect.height;
                            if (bottom_y > max_cone_bottom_y) {
                                max_cone_bottom_y = bottom_y;
                            }
                        }
                    }

                    // 检查是否满足进入引导模式的条件
                    bool entry_conditions_met = cone_found_this_frame && (max_cone_bottom_y >= CONE_BOTTOM_Y_THRESHOLD);

                    if (cone_found_this_frame) {
                        cone_target_x = target_tmp;
                        cones_lost_count = 0; // 重置丢失计数
                        if (!has_seen_cones) {
                            if(entry_conditions_met) {
                                cones_detect_count++; // 累加检测计数
                                cout << "[锥桶搜索] 发现有效锥桶 (高度满足)，计数: " << cones_detect_count << "/" << CONE_ENTER_THRESHOLD << " (y=" << (int)max_cone_bottom_y << ")" << endl;
                            } else {
                                if (cones_detect_count > 0) {
                                    cout << "[锥桶搜索] 锥桶高度不足 (y=" << (int)max_cone_bottom_y << ")，重置计数" << endl;
                                    cones_detect_count = 0;
                                }
                            }
                        }
                    } else {
                        cone_target_x = -1; // 未检测到，使用后备巡线
                        if (cones_detect_count > 0) {
                             cout << "[锥桶搜索] 锥桶丢失，重置检测计数" << endl;
                             cones_detect_count = 0; // 如果中断，则重置检测计数
                        }
                        if (has_seen_cones) {
                            cones_lost_count++; // 如果已在引导模式，累加丢失计数
                            cout << "[锥桶引导] 锥桶丢失计数: " << cones_lost_count << "/" << CONE_EXIT_THRESHOLD << endl;
                        }
                    }
                }

                // 4. 状态切换判定
                if (!has_seen_cones && cones_detect_count >= CONE_ENTER_THRESHOLD) {
                    cout << "[流程] 确认锥桶引导，进入引导模式" << endl;
                    has_seen_cones = true;
                }
                
                if (has_seen_cones && cones_lost_count >= CONE_EXIT_THRESHOLD) {
                    cout << "[流程] 通过锥桶区域，开始" << static_cast<int>(POST_CONE_CRUISE_DURATION_SECONDS) << "秒巡线延迟..." << endl;
                    parking_approach_detect_count = 0;
                    post_cone_cruise_start_time = std::chrono::steady_clock::now(); // 启动锥桶引导后巡线延迟计时器
                    need_post_cone_cruise_delay = true; // 设置延迟标志
                    from_cone_guidance = true; // 标记来自锥桶引导
                    setCarState(CarState::ParkingApproachDetection);
                }
                break;

            case CarState::ParkingApproachDetection:
                // 蓝框（蓝点）识别状态
                Tracking(bin_image);

                // 检查是否在锥桶引导后的巡线延迟时间内
                if (need_post_cone_cruise_delay) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - post_cone_cruise_start_time).count() / 1000000.0;
                    if (elapsed < POST_CONE_CRUISE_DURATION_SECONDS) {
                        // 在延迟时间内，只执行巡线，不进行任何检测
                        break; // 直接退出，不执行后续检测逻辑
                    } else {
                        // 延迟时间结束，开启蓝框检测
                        cout << "[流程] " << static_cast<int>(POST_CONE_CRUISE_DURATION_SECONDS) << "秒巡线延迟结束，开始蓝框检测..." << endl;
                        need_post_cone_cruise_delay = false; // 清除延迟标志，继续执行后续蓝框检测逻辑
                    }
                }

                // 检测蓝点（蓝框）
                if (detect_parking_approach_points(frame)) {
                    parking_approach_detect_count++;
                    cout << "[蓝框识别] 检测到引导蓝点, 计数: " << parking_approach_detect_count << "/" << PARKING_APPROACH_CONFIRM_THRESHOLD << endl;
                } else {
                    if (parking_approach_detect_count > 0) {
                       cout << "[蓝框识别] 引导蓝点丢失, 重置计数" << endl;
                    }
                    parking_approach_detect_count = 0;
                }

                if (parking_approach_detect_count >= PARKING_APPROACH_CONFIRM_THRESHOLD) {
                    // 蓝框检测有效后，进入BriefStopForABActivation状态，停车后再激活AB识别
                    cout << "[流程] 蓝框检测有效，准备执行短暂停车以稳定车辆" << endl;
                    brief_stop_start_time = std::chrono::steady_clock::now();
                    from_cone_guidance = false; // 清除标志（不再需要区分）
                    setCarState(CarState::BriefStopForABActivation);
                }
                break;

            case CarState::ParkingSearch:
                // AB识别和计数状态
                Tracking(bin_image);

                // 执行A/B识别逻辑
                result_ab.clear();
                result_ab = fastestdet_ab->detect(frame);
                {
                    // 初始化最近检测框（用于查找最近的A/B标识）
                    DetectObject closest_box = {cv::Rect_<float>(0, 0, 0, 0), -1, 0.0f};
                    if (!result_ab.empty()) {
                        // 查找最近的A/B标识（y2最大，且满足阈值要求）
                        for(const auto& box : result_ab) {
                            float box_y2 = box.rect.y + box.rect.height;
                            if (box_y2 > BZ_Y_LOWER_THRESHOLD) {
                                float closest_y2 = closest_box.rect.y + closest_box.rect.height;
                                if (box_y2 > closest_y2) closest_box = box;
                            }
                        }
                    }

                    if (closest_box.label != -1) {
                        // 累加A/B车库识别计数
                        if (closest_box.label == 0) park_A_count++;
                        else if (closest_box.label == 1) park_B_count++;

                        // 更新最近检测到的车库信息和跟随坐标
                        float closest_y2 = closest_box.rect.y + closest_box.rect.height;
                        latest_park_id = closest_box.label + 1;
                        float target_x = closest_box.rect.x + closest_box.rect.width / 2.0f;
                        parking_follow_x = static_cast<int>(target_x);

                        cout << "[AB识别] 最近: " << (latest_park_id == 1 ? "A" : "B") 
                             << " | 计数 A:" << park_A_count << ", B:" << park_B_count
                             << " | Y:" << (int)closest_y2 << "/" << PARKING_Y_THRESHOLD
                             << " | 跟随x:" << parking_follow_x << endl;

                        // 检查是否达到入库阈值
                        if (closest_y2 >= PARKING_Y_THRESHOLD) {
                            cout << "[流程] 达到入库阈值，触发短暂停车，位置 x=" << parking_follow_x << endl;
                            pending_pre_parking_label = -1;
                            brief_stop_start_time = std::chrono::steady_clock::now();
                            setCarState(CarState::BriefStopForParking);
                        }
                    } else {
                        // 未检测到有效目标，重置车库ID
                        latest_park_id = 0;
                    }
                }
                break;

            case CarState::BriefStopForABActivation:
                // 激活AB识别前的停车，不进行AB识别和计数
                // motor_servo_contral会处理反转和刹停
                {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - brief_stop_start_time).count() / 1000.0f;
                    
                    if (elapsed >= (BRIEF_STOP_REVERSE_DURATION + BRIEF_STOP_HOLD_DURATION)) {
                        // 停车时间结束，激活AB识别并进入ParkingSearch状态
                        cout << "[流程] 短暂停车结束，减速并激活A/B识别" << endl;
                        from_cone_guidance = false; // 确保标志为false（从BriefStopForABActivation路径来）
                        setCarState(CarState::ParkingSearch);
                    }
                }
                break;

            case CarState::BriefStopForParking:
                // 入库前停车，在停车期间继续AB识别和计数
                // motor_servo_contral会处理反转和刹停
                {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - brief_stop_start_time).count() / 1000.0f;
                    
                    // 在停车期间（未达到停车时间）或停车时间结束后，继续AB识别和计数
                    result_ab.clear();
                    result_ab = fastestdet_ab->detect(frame);
                    {
                        // 初始化最近检测框（用于查找最近的A/B标识）
                        DetectObject closest_box = {cv::Rect_<float>(0, 0, 0, 0), -1, 0.0f};
                        if (!result_ab.empty()) {
                            // 查找最近的A/B标识（y2最大，且满足阈值要求）
                            for(const auto& box : result_ab) {
                                float box_y2 = box.rect.y + box.rect.height;
                                if (box_y2 > BZ_Y_LOWER_THRESHOLD) {
                                    float closest_y2 = closest_box.rect.y + closest_box.rect.height;
                                    if (box_y2 > closest_y2) closest_box = box;
                                }
                            }
                        }

                        if (closest_box.label != -1) {
                            // 累加A/B车库识别计数
                            if (closest_box.label == 0) park_A_count++;
                            else if (closest_box.label == 1) park_B_count++;

                            // 更新最近检测到的车库信息和跟随坐标
                            float closest_y2 = closest_box.rect.y + closest_box.rect.height;
                            latest_park_id = closest_box.label + 1;
                            float target_x = closest_box.rect.x + closest_box.rect.width / 2.0f;
                            parking_follow_x = static_cast<int>(target_x);

                            cout << "[停车] 最近: " << (latest_park_id == 1 ? "A" : "B") 
                                 << " | 计数 A:" << park_A_count << ", B:" << park_B_count
                                 << " | Y:" << (int)closest_y2 << "/" << PARKING_Y_THRESHOLD
                                 << " | 跟随x:" << parking_follow_x << endl;
                        } else {
                            // 未检测到有效目标，重置车库ID
                            latest_park_id = 0;
                        }
                    }
                    
                    if (elapsed >= (BRIEF_STOP_REVERSE_DURATION + BRIEF_STOP_HOLD_DURATION)) {
                        // 停车时间结束后，根据计数决定入库方向
                        int decided_label = pending_pre_parking_label;
                        if (decided_label == -1) decided_label = decide_parking_label_from_counts();

                        if (decided_label != -1) {
                            // 成功确定目标车库，进入预入库阶段
                            final_target_label = decided_label;
                            parking_target_not_detected_count = 0;
                            parking_target_detected_this_frame = false;
                            
                            cout << "[流程] 短暂停车结束，开始预入库 -> "
                                 << (final_target_label == 0 ? "A(左)" : "B(右)") << endl;
                            setCarState(CarState::PreParking);
                        } else {
                            // 无法确认目标，重新开始蓝框识别
                            cout << "[警告] 无法确认A/B，重新开始蓝框识别" << endl;
                            parking_approach_detect_count = 0;
                            from_cone_guidance = false; // 确保标志为false（从BriefStopForParking路径来）
                            setCarState(CarState::ParkingApproachDetection);
                        }
                    }
                }
                break;

            case CarState::PreParking:
                result_ab.clear();
                result_ab = fastestdet_ab->detect(frame);
                parking_target_detected_this_frame = false;
                {
                    // 初始化最远目标（用于查找最远的A或B）
                    DetectObject farthest_target;
                    farthest_target.label = -1;
                    float farthest_y = 9999.0f;

                    if (!result_ab.empty()) {
                        // 查找最远的目标（y值最小）
                        for(const auto& box : result_ab) {
                            if (box.label == final_target_label) {
                                float box_y = box.rect.y;
                                if (box_y < farthest_y) {
                                    farthest_y = box_y;
                                    farthest_target = box;
                                    parking_target_detected_this_frame = true;
                                }
                            }
                        }
                    }

                    if (parking_target_detected_this_frame) {
                        // 检测到目标，更新跟随坐标并重置计数器
                        float target_x = farthest_target.rect.x + farthest_target.rect.width / 2.0f;
                        parking_follow_x = static_cast<int>(target_x);
                        parking_target_not_detected_count = 0;
                        cout << "[预入库] 跟随 " << (final_target_label == 0 ? "A" : "B") 
                             << " x=" << parking_follow_x << " y=" << (int)farthest_y << endl;
                    } else {
                        // 未检测到目标，累加丢失计数
                        parking_target_not_detected_count++;
                        cout << "[预入库] 目标丢失 (" << parking_target_not_detected_count 
                             << "/" << PARKING_DETECT_MISS_THRESHOLD << ")" << endl;
                    }

                    // 检查是否达到停车阈值
                    if (parking_target_not_detected_count >= PARKING_DETECT_MISS_THRESHOLD) {
                        cout << "[流程] 预入库完成，停车！" << endl;
                        gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock);
                        setCarState(CarState::ParkingComplete);
                        
                        // 切换到高分辨率显示模式
                        capture.release();
                        capture.open(0);
                        capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                        capture.set(cv::CAP_PROP_FPS, 30);
                        capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
                        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
                        
                        cv::Mat high_res_frame;
                        cout << "[流程] 显示高分辨率图像，按'q'退出..." << endl;
                        while (capture.read(high_res_frame) && !high_res_frame.empty()) {
                            cv::imshow("Parking Complete", high_res_frame);
                            if ((cv::waitKey(1) & 0xFF) == 'q') break;
                        }
                        program_finished = true;
                    }
                }
                break;
            
            default:
                // 未知状态，不做处理
                break;
            }


        // 3. 电机与舵机控制
        motor_servo_contral(); // 根据当前状态控制车辆运动

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
            continue; // 继续处理下一帧
        } catch (const std::exception& e) {
            cerr << "[错误] 标准异常: " << e.what() << endl;
            continue; // 继续处理下一帧
        } catch (...) {
            cerr << "[错误] 未知异常，跳过当前帧" << endl;
            continue; // 继续处理下一帧
        }

    }

    // 清理资源
    cout << "\n[清理] 释放模型资源..." << endl;
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock); // 确保电机停止
    gpioPWM(servo_pin, servo_pwm_mid);               // 舵机回中
    std::this_thread::sleep_for(std::chrono::milliseconds(100));                                  // 短暂延时

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
