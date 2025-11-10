// Last update: 2024/12/5 
// 安徽芜湖----国赛----------

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

using namespace std; // 使用标准命名空间
using namespace cv; // 使用OpenCV命名空间

//------------有关的全局变量定义------------------------------------------------------------------------------------------

//-----------------图像相关----------------------------------------------
Mat frame; // 存储视频帧
Mat frame_a; // 存储视频帧
Mat bin_image; // 存储二值化图像--Sobel检测后图像

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
int flag_banma = 0; // 斑马线标志

//----------------变道相关---------------------------------------------------

int changeroad = 1; // 变道检测结果
int flag_changeroad = 0; // 变道标志



// 定义舵机和电机引脚号、PWM范围、PWM频率、PWM占空比解锁值
const int servo_pin = 12; // 存储舵机引脚号
const float servo_pwm_range = 10000.0; // 存储舵机PWM范围
const float servo_pwm_frequency = 50.0; // 存储舵机PWM频率
const float servo_pwm_duty_cycle_unlock = 730.0; // 存储舵机PWM占空比解锁值

//---------------------------------------------------------------------------------------------------
float servo_pwm_mid = servo_pwm_duty_cycle_unlock; // 存储舵机中值
//---------------------------------------------------------------------------------------------------

const int motor_pin = 13; // 存储电机引脚号
const float motor_pwm_range = 40000; // 存储电机PWM范围
const float motor_pwm_frequency = 200.0; // 存储电机PWM频率
const float motor_pwm_duty_cycle_unlock = 11400.0; // 存储电机PWM占空比解锁值


//---------------------------------------------------------------------------------------------------
float motor_pwm_mid = 11400.0; // 存储电机解锁值
//---------------------------------------------------------------------------------------------------

const int yuntai_LR_pin = 22; // 存储云台引脚号
const float yuntai_LR_pwm_range = 1000.0; // 存储云台PWM范围
const float yuntai_LR_pwm_frequency = 50.0; // 存储云台PWM频率
const float yuntai_LR_pwm_duty_cycle_unlock = 63.0; //大左小右 

const int yuntai_UD_pin = 23; // 存储云台引脚号
const float yuntai_UD_pwm_range = 1000.0; // 存储云台PWM范围
const float yuntai_UD_pwm_frequency = 50.0; // 存储云台PWM频率
const float yuntai_UD_pwm_duty_cycle_unlock = 58.0; //大上下小

int first_bz_get = 0;
int number = 0;
int number1 = 0;

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
const int BANMA_ROI_Y = 110;         // ROI左上角Y坐标
const int BANMA_ROI_WIDTH = 318;     // ROI宽度
const int BANMA_ROI_HEIGHT = 200;    // ROI高度

// 斑马线矩形筛选尺寸（斑马线由多个白色矩形组成）
const int BANMA_RECT_MIN_WIDTH = 10;   // 矩形最小宽度
const int BANMA_RECT_MAX_WIDTH = 50;   // 矩形最大宽度
const int BANMA_RECT_MIN_HEIGHT = 10;  // 矩形最小高度
const int BANMA_RECT_MAX_HEIGHT = 50;  // 矩形最大高度

// 斑马线判定阈值
const int BANMA_MIN_COUNT = 6;  // 判定为斑马线需要的最少白色矩形数量

// 形态学处理参数
const int BANMA_MORPH_KERNEL_SIZE = 3;  // 形态学处理kernel大小（3x3）

//---------------性能优化选项-------------------------------------------------
// 如果树莓派性能不足，可以设置为1以使用更快的处理方式（可能会略微降低效果）
const int MIN_COMPONENT_AREA = 400;
const bool SHOW_SOBEL_DEBUG = false;

// 定义舵机和电机PWM初始化函数
void servo_motor_pwmInit(void) 
{
    if (gpioInitialise() < 0) // 初始化GPIO，如果失败则返回
    {
        std::cout << "GPIO failed ! Please use sudo !" << std::endl; // 输出失败信息
        return; // 返回
    }
    else
        std::cout << "GPIO ok. Good !!" << std::endl; // 输出成功信息

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
cv::Mat undistort(const cv::Mat &frame) 
{
    double k1 = 0.0439656098483248; // 畸变系数k1
    double k2 = -0.0420991522460257; // 畸变系数k2
    double p1 = 0.0; // 畸变系数p1
    double p2 = 0.0; // 畸变系数p2
    double k3 = 0.0; // 畸变系数k3

    // 相机内参矩阵
    cv::Mat K = (cv::Mat_<double>(3, 3) << 176.842468665091, 0.0, 159.705914860981,
                 0.0, 176.990910857055, 120.557953465790,
                 0.0, 0.0, 1.0);

    // 畸变系数矩阵
    cv::Mat D = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
    cv::Mat mapx, mapy; // 映射矩阵
    cv::Mat undistortedFrame; // 去畸变后的图像帧

    // 初始化去畸变映射
    cv::initUndistortRectifyMap(K, D, cv::Mat(), K, frame.size(), CV_32FC1, mapx, mapy);
    // 应用映射，得到去畸变后的图像
    cv::remap(frame, undistortedFrame, mapx, mapy, cv::INTER_LINEAR);

    return undistortedFrame; // 返回去畸变后的图像
}

// 定义自定义直方图均衡化函数，输入为图像和alpha值   在ImagePreprocessing函数中调用
Mat customEqualizeHist(const Mat &inputImage, float alpha) 
{
    Mat enhancedImage; // 定义增强后的图像
    equalizeHist(inputImage, enhancedImage); // 对输入图像进行直方图均衡化

    // 减弱对比度增强的效果
    return alpha * enhancedImage + (1 - alpha) * inputImage; // 返回调整后的图像
}

// 后续避障补线用
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

cv::Mat ImageSobel(cv::Mat &frame, cv::Mat *debugOverlay = nullptr) 
{
    (void)debugOverlay;

    // 定义图像宽度和高度
    const int width = 320;
    const int height = 240;

    // 初始化二值输出图像
    Mat binaryImage = Mat::zeros(height, width, CV_8U);
    Mat binaryImage_1 = Mat::zeros(height, width, CV_8U);

    // 转换输入图像为灰度图像
    Mat grayImage;
    cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);

    int kernelSize = 5;
    double sigma = 1.0;
    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(kernelSize, kernelSize), sigma);

    // Sobel 边缘检测
    Mat sobelY;
    Sobel(blurredImage, sobelY, CV_64F, 0, 1, 3); // y方向梯度

    // 计算梯度幅值并转换为 8 位图像
    Mat gradientMagnitude = abs(sobelY);
    convertScaleAbs(gradientMagnitude, gradientMagnitude);

    // 阈值分割并膨胀操作
    cv::threshold(gradientMagnitude, binaryImage, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binaryImage, binaryImage, kernel,cv::Point(-1, -1), 1);

    // 定义感兴趣区域 (ROI)
    const int x_roi = 1, y_roi = 109, width_roi = 318, height_roi = 46;
    Rect roi(x_roi, y_roi, width_roi, height_roi);
    Mat croppedImage = binaryImage(roi);

    // 使用概率霍夫变换检测直线
    vector<Vec4i> lines;
    HoughLinesP(croppedImage, lines, 1, CV_PI / 180, 25, 15, 10);

    // 遍历直线并筛选有效线段
    for (const auto &l : lines) 
    {
        // 计算直线角度和长度
        double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
        double length = hypot(l[3] - l[1], l[2] - l[0]);

        // 筛选条件：角度范围、最小长度
        if (abs(angle) > 15) 
        {
            // 调整坐标以适应全图
            Vec4i adjustedLine = l;
            adjustedLine[0] += x_roi;
            adjustedLine[1] += y_roi;
            adjustedLine[2] += x_roi;
            adjustedLine[3] += y_roi;

            // 绘制白线
            line(binaryImage_1, Point(adjustedLine[0], adjustedLine[1]),
                Point(adjustedLine[2], adjustedLine[3]), Scalar(255), 2, LINE_AA);
        }
    }

    // 返回最终的处理图像
    return binaryImage_1;
}

void Tracking(cv::Mat &dilated_image) 
{
    // 参数检查
    if (dilated_image.empty() || dilated_image.type() != CV_8U) 
    {
        std::cerr << "Invalid input image for Tracking!" << std::endl;
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
        int left = begin;  // 左侧搜索起点
        int right = begin; // 右侧搜索起点
        bool left_found = false; // 标记是否找到左线
        bool right_found = false; // 标记是否找到右线

        // 搜索左线
        while (left > 1) 
        {
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
        while (right < 318) 
        {
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
            right_line.emplace_back(318, i); // 右线未找到，默认记录最右侧点
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

// 比较两个轮廓的面积
bool Contour_Area(vector<Point> contour1, vector<Point> contour2)
{
    return contourArea(contour1) > contourArea(contour2); // 返回轮廓1是否大于轮廓2
}

// 定义蓝色挡板 寻找函数
void blue_card_find(void)  // 输入为mask图像
{   
    Mat change_frame; // 存储颜色空间转换后的图像
    cvtColor(frame, change_frame, COLOR_BGR2HSV); // 转换颜色空间

    Mat mask; // 存储掩码图像

    // 定义HSV范围 hsv颜色空间特点：色调H、饱和度S、亮度V
    Scalar scalarl = Scalar(BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN); // HSV的低值
    Scalar scalarH = Scalar(BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX);  // HSV的高值
    inRange(change_frame, scalarl, scalarH, mask); // 创建掩码

    // 限制检测区域到画面中央区域，减少边缘干扰
    cv::Rect roi_blue(BLUE_ROI_X, BLUE_ROI_Y, BLUE_ROI_WIDTH, BLUE_ROI_HEIGHT);
    Mat mask_roi = mask(roi_blue);

    vector<vector<Point>> contours; // 存储轮廓的向量
    vector<Vec4i> hierarcy; // 存储层次结构的向量
    findContours(mask_roi, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE); // 查找轮廓
    
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
void blue_card_remove(void) // 输入为mask图像
{
    cout << "进入 蓝色挡板移开 进程！" << endl; // 输出进入移除蓝色挡板的过程

    Mat change_frame; // 存储颜色空间转换后的图像
    cvtColor(frame, change_frame, COLOR_BGR2HSV); // 转换颜色空间

    Mat mask; // 存储掩码图像

    // 定义HSV范围 hsv颜色空间特点：色调H、饱和度S、亮度V（使用与blue_card_find相同的参数）
    Scalar scalarl = Scalar(BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN); // HSV的低值
    Scalar scalarH = Scalar(BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX);  // HSV的高值 
    inRange(change_frame, scalarl, scalarH, mask); // 创建掩码

    // 使用与blue_card_find相同的ROI区域进行裁剪
    cv::Rect roi_blue(BLUE_ROI_X, BLUE_ROI_Y, BLUE_ROI_WIDTH, BLUE_ROI_HEIGHT);
    Mat mask_roi = mask(roi_blue);

    vector<vector<Point>> contours; // 定义轮廓向量
    vector<Vec4i> hierarcy; // 定义层次结构向量
    findContours(mask_roi, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE); // 查找轮廓

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

    // 判断是否存在“有效蓝色轮廓”：若不存在，说明挡板已移开
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

int banma_get(cv::Mat &frame) {
    // 将输入图像转换为HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

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

    // 裁剪ROI区域（使用常量，并确保不超出图像边界）
    cv::Rect roi(BANMA_ROI_X, BANMA_ROI_Y, 
                 std::min(BANMA_ROI_WIDTH, mask1.cols - BANMA_ROI_X), 
                 std::min(BANMA_ROI_HEIGHT, mask1.rows - BANMA_ROI_Y));
    cv::Mat src = mask1(roi);

    // 查找图像中的轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 创建一个副本以便绘制轮廓
    cv::Mat contour_img = src.clone();

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

// 找停车位
int find_parking(cv::Mat frame) {
    // 将图像转换为HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // 定义红色的HSV范围
    cv::Scalar lower_red1(0, 43, 46);
    cv::Scalar upper_red1(10, 255, 255);

    cv::Scalar lower_red2(156, 43, 46);
    cv::Scalar upper_red2(180, 255, 255);

    // 创建两个红色掩码并合并
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, lower_red1, upper_red1, mask1);
    cv::inRange(hsv, lower_red2, upper_red2, mask2);
    cv::bitwise_or(mask1, mask2, mask);

    // 执行形态学操作，可以根据实际情况调整参数
    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1); // 5x5的卷积核
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);  // 开运算 去除外部小噪点
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel); // 闭运算 去除内部小洞

    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 仅保留y轴在100-218范围内的轮廓  
    std::vector<std::vector<cv::Point>> filtered_contours;
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect bounding_box = cv::boundingRect(contours[i]);
        if (bounding_box.y >= 100 && bounding_box.y <= 218) {
            filtered_contours.push_back(contours[i]);
        }
    }

    // 计算大于4000面积的轮廓数量
    int large_contours_count = 0;
    for (size_t i = 0; i < filtered_contours.size(); i++) {
        double area = cv::contourArea(filtered_contours[i]);
        if (area > 1000) {
            large_contours_count++;
        }
    }

    // 如果有两个以上的轮廓的面积大于4000，返回1
    if (large_contours_count >= 1) {
        return 1;
    } else {
        return 0;
    }
}

float servo_pd(int target) { // 赛道巡线控制

    int pidx = int((mid[23].x + mid[25].x) / 2); // 计算中线中点的x坐标

    cout << " PIDX: " << pidx << endl;  

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

void gohead(int parkchose){
    if(parkchose == 1 ){ //try to find park A
        gpioPWM(motor_pin, 12800);
        gpioPWM(motor_pin, 11000); // 设置电机PWM
        gpioPWM(servo_pin, 870); // 设置舵机PWM
        sleep(2);
        cout << "gohead--------------------------------------------------------------Try To Find Park AAAAAAAAAAAAAAA" << endl;
    }
    else if(parkchose == 2){ //try to find park B
        gpioPWM(motor_pin, 12800);
        gpioPWM(motor_pin, 11000); // 设置电机PWM
        gpioPWM(servo_pin, 750); // 设置舵机PWM
        sleep(2);
        cout << "gohead--------------------------------------------------------------Try To Find Park BBBBBBBBBBBBBBB" << endl;
    }
}

void banma_stop(){
    gpioPWM(motor_pin, 10800);
    gpioPWM(motor_pin, 10400);
    gpioPWM(motor_pin, 10000); // 设置电机PWM
    usleep(500000); // 延时300毫秒
    gpioPWM(servo_pin, servo_pwm_mid); // 设置舵机PWM
    gpioPWM(motor_pin, 10000); // 设置电机PWM
}

void motor_changeroad(){
    if(changeroad == 1){ // 向左变道----------------------------------------------------------------
        gpioPWM(servo_pin, 810); // 设置舵机PWM
        gpioPWM(motor_pin, 12000); // 设置电机PWM
        usleep(900000);
        gpioPWM(servo_pin, 610); // 设置舵机PWM
        gpioPWM(motor_pin, 11500); // 设置电机PWM
        usleep(900000); // 延时550毫秒
        gpioPWM(servo_pin, servo_pwm_mid); // 设置舵机PWM
        gpioPWM(motor_pin, 11400); // 设置电机PWM
    }else if(changeroad == 2){ //向右变道----------------------------------------------------------------
        gpioPWM(servo_pin, 630); // 设置舵机PWM
        gpioPWM(motor_pin, 12000); // 设置电机PWM
        usleep(900000);
        gpioPWM(servo_pin, 770); // 设置舵机PWM
        gpioPWM(motor_pin, 11500); // 设置电机PWM
        usleep(900000); // 延时550毫秒
        gpioPWM(servo_pin, servo_pwm_mid); // 设置舵机PWM
        gpioPWM(motor_pin, 11400); // 设置电机PWM
    }
}

void motor_park(){
    gpioPWM(motor_pin, 10800); // 设置电机PWM
    gpioPWM(motor_pin, 10000); // 设置电机PWM
    gpioPWM(motor_pin, 9100); // 设置电机PWM
    usleep(200000); // 延时300毫秒
    gpioPWM(servo_pin, servo_pwm_mid); // 设置舵机PWM
    gpioPWM(motor_pin, 10000); // 设置电机PWM
}

// 控制舵机电机
void motor_servo_contral()
{   
    float servo_pwm_now; 
    // 只有发车信号激活（fache_sign = 1）时，才执行巡线控制
    if (fache_sign == 1) { 
        if (banma == 0 && flag_banma == 0 ){
            // 确保 mid 向量有足够数据（避免越界）
            if (mid.size() < 26) {
                cerr << "mid 向量数据不足，使用默认角度" << endl;
                servo_pwm_now = servo_pwm_mid;
            } else {
                servo_pwm_now = servo_pd(160); 
                cerr << "找到有效线" << endl;
            }
            if(number < 10){
                gpioPWM(motor_pin, motor_pwm_mid + 1800); 
            }else if (number < 500){
                gpioPWM(motor_pin, motor_pwm_mid + 1500); 
                cout << "巡线-----------------------弯道1 舵机PWM:  " << servo_pwm_now << endl;
            }else if (number < 550){
                gpioPWM(motor_pin, motor_pwm_mid + 1500); 
                cout << "巡线-----------------------弯道2 舵机PWM:  " << servo_pwm_now << endl;
            }else if (number < 600){
                gpioPWM(motor_pin, motor_pwm_mid + 1500); 
                cout << "巡线-----------------------弯道3 舵机PWM:  " << servo_pwm_now << endl;
            }else{
                gpioPWM(motor_pin, motor_pwm_mid + 1500); 
                cout << "巡线-----------------------弯道4 舵机PWM:  " << servo_pwm_now << endl;
            }
            gpioPWM(servo_pin, servo_pwm_now);
        }
        else if(banma == 1 && flag_banma == 0){ 
            flag_banma = 1;
            banma_stop();
            //system("sudo -u pi /home/pi/.nvm/versions/node/v12.22.12/bin/node /home/pi/network-rc/we2hdu.js");
            number = 0;
        }
    } else {
        // 发车前（fache_sign = 0）：停止电机和舵机，避免无效操作
        gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock); // 电机解锁但不转动
        gpioPWM(servo_pin, servo_pwm_mid); // 舵机回中
    }
}

//-----------------------------------------------------------------------------------主函数-----------------------------------------------
int main(void)
{
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
        cout << "Can not open camera!" << endl;
        cout << "please enter any key to exit" << endl;
        cin.ignore();// 等待用户输入
        return -1;            // 返回错误代码
    }

    // 输出摄像头的属性
    cout << "FPS: " << capture.get(cv::CAP_PROP_FPS) << endl;
    cout << "Frame Width: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << endl;
    cout << "Frame Height: " << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;
    //---------------------------------------------------

    while (capture.read(frame)){

        frame = undistort(frame); // 对帧进行去畸变处理

        auto start = std::chrono::high_resolution_clock::now();

        // 处理发车逻辑
        if (fache_sign == 0) // 如果开始标志为0
        {
            // 根据条件调用不同的函数
            if (find_first == 0) //   find_first = 0; 标记是否找到第一个目标  1为找到 0为未找到 默认值为0 找到后进入检测是否移开挡板
            {
                blue_card_find(); // 查找蓝卡
            }
            else
            {
                blue_card_remove(); // 移除蓝卡
            }

        }
        else // 如果开始标志不为1
        {

            number++; // 计数器加1

            if ( banma == 0 ){

                cv::Mat debugOverlay;
                bin_image = ImageSobel(frame, SHOW_SOBEL_DEBUG ? &debugOverlay : nullptr); // 图像预处理
                if (SHOW_SOBEL_DEBUG && !debugOverlay.empty())
                {
                    cv::imshow("TrackLine Overlay", debugOverlay);
                    cv::waitKey(1);
                }
                Tracking(bin_image); // 进行巡线识别

                if(number > 600){
                    banma = banma_get(frame);
                    cout << "斑马线检测---------------------------:   " << banma << endl;
                }

            }
        }

        motor_servo_contral(); // 控制舵机电机

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time per frame: " << elapsed.count() << " s    FPS: " << (elapsed.count() > 0 ? 1.0 / elapsed.count() : 0.0) << "    number: " << number << std::endl;

    }
}
