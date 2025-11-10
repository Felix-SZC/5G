#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <deque>

using namespace std;

// 检测对象结构体
struct Object
{
    cv::Rect_<float> rect;      // 边界框
    int label;                  // 类别ID
    float prob;                 // 置信度
};

// 计算两个边界框的交集面积
static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

// 快速排序
static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

// NMS
static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // 类别内NMS
            if (a.label != b.label)
                continue;

            // 计算IoU
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

static inline float fast_tanh(float x)
{
    return tanh(x);
}

// 生成检测候选框
// FastestDet输出格式: [1, C, H, W]
// C = 1 + 4 + num_class
// 通道0: 前背景分类分数
// 通道1-4: 边界框回归
// 通道5+: 类别分类分数
static void generate_proposals(const ncnn::Mat& out, int img_w, int img_h, 
                               float prob_threshold, std::vector<Object>& objects)
{
    const int channels = out.c;
    const int height = out.h;
    const int width = out.w;
    
    const int num_class = channels - 5; // 类别数量
    
    // 遍历特征图的每个位置
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int base_idx = i * width + j;
            
            // 从通道0获取前背景分数
            const float* pobj_ptr = out.channel(0);
            float obj_score = pobj_ptr[base_idx];
            
            // 从通道1-4获取边界框回归值
            const float* preg_ptr0 = out.channel(1);
            const float* preg_ptr1 = out.channel(2);
            const float* preg_ptr2 = out.channel(3);
            const float* preg_ptr3 = out.channel(4);
            
            float tx = preg_ptr0[base_idx];
            float ty = preg_ptr1[base_idx];
            float tw = preg_ptr2[base_idx];
            float th = preg_ptr3[base_idx];
            
            // 从通道5+找到最大类别分数和对应索引
            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                const float* pcls_ptr = out.channel(5 + k);
                float score = pcls_ptr[base_idx];
                if (score > class_score)
                {
                    class_score = score;
                    class_index = k;
                }
            }
            
            // 计算置信度: obj^0.6 * cls^0.4
            float confidence = pow(obj_score, 0.6f) * pow(class_score, 0.4f);
            
            if (confidence < prob_threshold)
                continue;
            
            // 解码边界框
            // bcx = (tanh(tx) + gx) / W
            // bcy = (tanh(ty) + gy) / H
            // bw = sigmoid(tw)
            // bh = sigmoid(th)
            
            float bcx = (fast_tanh(tx) + j) / width;
            float bcy = (fast_tanh(ty) + i) / height;
            float bw = sigmoid(tw);
            float bh = sigmoid(th);
            
            // 转换为 x1, y1, x2, y2
            float x1 = bcx - 0.5f * bw;
            float y1 = bcy - 0.5f * bh;
            float x2 = bcx + 0.5f * bw;
            float y2 = bcy + 0.5f * bh;
            
            // 裁剪到 [0, 1]
            x1 = std::max(0.0f, std::min(1.0f, x1));
            y1 = std::max(0.0f, std::min(1.0f, y1));
            x2 = std::max(0.0f, std::min(1.0f, x2));
            y2 = std::max(0.0f, std::min(1.0f, y2));
            
            // 转换为像素坐标
            x1 *= img_w;
            y1 *= img_h;
            x2 *= img_w;
            y2 *= img_h;
            
            Object obj;
            obj.rect.x = x1;
            obj.rect.y = y1;
            obj.rect.width = x2 - x1;
            obj.rect.height = y2 - y1;
            obj.label = class_index;
            obj.prob = confidence;
            
            objects.push_back(obj);
        }
    }
}

// 对单帧进行检测
static int detect_frame(ncnn::Net& fastestdet, const cv::Mat& bgr, std::vector<Object>& objects,
                        int target_size = 352, float prob_threshold = 0.65f, 
                        float nms_threshold = 0.45f)
{
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // 图像预处理
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, 
                                                  img_w, img_h, target_size, target_size);

    // 归一化到 [0, 1]
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = fastestdet.create_extractor();

    ex.input("in0", in);

    // 执行推理
    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<Object> proposals;

    // 后处理
    generate_proposals(out, img_w, img_h, prob_threshold, proposals);

    // 按置信度排序
    qsort_descent_inplace(proposals);

    // 应用NMS
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
    }

    return 0;
}

// 在图像上绘制检测结果
static void draw_objects(cv::Mat& image, const std::vector<Object>& objects, 
                        const std::vector<std::string>& class_names)
{
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        
        int x1 = static_cast<int>(obj.rect.x);
        int y1 = static_cast<int>(obj.rect.y);
        int x2 = static_cast<int>(obj.rect.x + obj.rect.width);
        int y2 = static_cast<int>(obj.rect.y + obj.rect.height);
        
        std::string label;
        if (obj.label < (int)class_names.size())
        {
            label = class_names[obj.label];
        }
        else
        {
            label = "Class_" + std::to_string(obj.label);
        }

        // 绘制黄色边界框
        cv::Scalar color = cv::Scalar(0, 255, 255);  // 黄色 (BGR)
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // 准备标签文本
        char text[256];
        sprintf(text, "%s: %.2f", label.c_str(), obj.prob);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = x1;
        int y = y1 - 5;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        // 绘制文本背景
        cv::rectangle(image, cv::Point(x, y - label_size.height - baseLine),
                     cv::Point(x + label_size.width, y + baseLine),
                     cv::Scalar(0, 0, 0), -1);

        // 绘制文本
        cv::putText(image, text, cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

// FPS计算类
class FPSCounter
{
private:
    std::deque<double> frame_times;
    const size_t max_samples = 30;  // 使用最近30帧计算平均FPS
    
public:
    FPSCounter() {}
    
    void update(double frame_time_ms)
    {
        frame_times.push_back(frame_time_ms);
        if (frame_times.size() > max_samples)
        {
            frame_times.pop_front();
        }
    }
    
    float getFPS() const
    {
        if (frame_times.empty())
            return 0.0f;
        
        double sum = 0.0;
        for (double t : frame_times)
        {
            sum += t;
        }
        double avg_time_ms = sum / frame_times.size();
        return 1000.0f / avg_time_ms;
    }
    
    float getAvgTime() const
    {
        if (frame_times.empty())
            return 0.0f;
        
        double sum = 0.0;
        for (double t : frame_times)
        {
            sum += t;
        }
        return sum / frame_times.size();
    }
};

void print_usage()
{
    cout << "\n使用方法: ./infer_camera [选项]\n" << endl;
    cout << "选项:" << endl;
    cout << "  --camera <int>         摄像头设备ID (默认: 0)" << endl;
    cout << "  --model <path>         模型param文件路径 (默认: FastestDet.ncnn.param)" << endl;
    cout << "  --size <int>           模型输入尺寸 (默认: 352)" << endl;
    cout << "  --conf <float>         置信度阈值 (默认: 0.65)" << endl;
    cout << "  --nms <float>          NMS阈值 (默认: 0.45)" << endl;
    cout << "  --width <int>          摄像头宽度 (默认: 1280)" << endl;
    cout << "  --height <int>         摄像头高度 (默认: 720)" << endl;
    cout << "  --fps <int>            摄像头帧率 (默认: 30)" << endl;
    cout << "  --threads <int>        NCNN线程数 (默认: 4)" << endl;
    cout << "  --save                 保存输出视频" << endl;
    cout << "  --output <path>        输出视频文件路径 (默认: output_camera.avi)" << endl;
    cout << "  --headless             无显示模式（自动录制，无GUI）" << endl;
    cout << "  --duration <int>       录制时长（秒，仅headless模式，0为无限）" << endl;
    cout << "  -h, --help             显示帮助信息" << endl;
    cout << "\n键盘控制（仅GUI模式）:" << endl;
    cout << "  q 或 ESC              退出程序" << endl;
    cout << "  s                     截图保存" << endl;
    cout << "  r                     开始/停止录制视频" << endl;
    cout << endl;
}

int main(int argc, char** argv)
{
    // 默认参数
    int camera_id = 0;
    std::string model_path = "FastestDet.ncnn.param";
    std::string bin_path = "FastestDet.ncnn.bin";
    int target_size = 352;
    float prob_threshold = 0.65f;
    float nms_threshold = 0.45f;
    int camera_width = 1280;
    int camera_height = 720;
    int camera_fps = 30;
    int num_threads = 4;
    bool save_output = false;
    std::string output_path = "output_camera.avi";
    bool headless_mode = false;
    int duration_seconds = 0;  // 0 表示无限录制

    // 解析命令行参数
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            print_usage();
            return 0;
        }
        else if (arg == "--camera" && i + 1 < argc)
        {
            camera_id = std::stoi(argv[++i]);
        }
        else if (arg == "--model" && i + 1 < argc)
        {
            model_path = argv[++i];
            bin_path = model_path;
            size_t pos = bin_path.find(".param");
            if (pos != std::string::npos)
            {
                bin_path.replace(pos, 6, ".bin");
            }
        }
        else if (arg == "--size" && i + 1 < argc)
        {
            target_size = std::stoi(argv[++i]);
        }
        else if (arg == "--conf" && i + 1 < argc)
        {
            prob_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--nms" && i + 1 < argc)
        {
            nms_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--width" && i + 1 < argc)
        {
            camera_width = std::stoi(argv[++i]);
        }
        else if (arg == "--height" && i + 1 < argc)
        {
            camera_height = std::stoi(argv[++i]);
        }
        else if (arg == "--fps" && i + 1 < argc)
        {
            camera_fps = std::stoi(argv[++i]);
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            num_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--save")
        {
            save_output = true;
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            output_path = argv[++i];
        }
        else if (arg == "--headless")
        {
            headless_mode = true;
            save_output = true;  // headless模式自动保存
        }
        else if (arg == "--duration" && i + 1 < argc)
        {
            duration_seconds = std::stoi(argv[++i]);
        }
    }

    // 打印配置信息
    cout << "======================================" << endl;
    cout << "FastestDet 摄像头实时推理" << endl;
    cout << "======================================" << endl;
    cout << "模型: " << model_path << endl;
    cout << "摄像头ID: " << camera_id << endl;
    cout << "摄像头分辨率: " << camera_width << "x" << camera_height << "@" << camera_fps << "fps" << endl;
    cout << "输入尺寸: " << target_size << endl;
    cout << "置信度阈值: " << prob_threshold << endl;
    cout << "NMS阈值: " << nms_threshold << endl;
    cout << "线程数: " << num_threads << endl;
    cout << "运行模式: " << (headless_mode ? "无显示模式 (Headless)" : "GUI模式") << endl;
    cout << "保存视频: " << (save_output ? "是 (" + output_path + ")" : "否") << endl;
    if (headless_mode && duration_seconds > 0)
    {
        cout << "录制时长: " << duration_seconds << " 秒" << endl;
    }
    cout << "======================================\n" << endl;

    // 打开摄像头
    cout << "正在打开摄像头..." << endl;
    cv::VideoCapture cap(camera_id);
    
    if (!cap.isOpened())
    {
        fprintf(stderr, "错误: 无法打开摄像头 %d\n", camera_id);
        return -1;
    }

    // 设置摄像头参数
    cap.set(cv::CAP_PROP_FRAME_WIDTH, camera_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height);
    cap.set(cv::CAP_PROP_FPS, camera_fps);

    // 获取实际摄像头参数
    int actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = cap.get(cv::CAP_PROP_FPS);

    cout << "摄像头已打开!" << endl;
    cout << "实际分辨率: " << actual_width << "x" << actual_height << endl;
    cout << "实际帧率: " << actual_fps << endl;
    cout << endl;

    // 初始化视频写入器
    cv::VideoWriter writer;
    bool is_recording = false;

    // 加载模型
    cout << "正在加载模型..." << endl;
    ncnn::Net fastestdet;

    // 设置NCNN选项
    fastestdet.opt.use_vulkan_compute = false;  // 不使用GPU加速
    fastestdet.opt.num_threads = num_threads;   // 设置线程数

    // 加载模型参数
    if (fastestdet.load_param(model_path.c_str()))
    {
        fprintf(stderr, "加载模型参数失败: %s\n", model_path.c_str());
        return -1;
    }
    if (fastestdet.load_model(bin_path.c_str()))
    {
        fprintf(stderr, "加载模型权重失败: %s\n", bin_path.c_str());
        return -1;
    }

    cout << "模型加载成功!\n" << endl;

    // 类别名称
    std::vector<std::string> class_names = {"A", "B"};

    // FPS计数器
    FPSCounter fps_counter;
    
    // 帧计数器
    int frame_count = 0;
    int screenshot_count = 0;

    cout << "开始实时推理..." << endl;
    if (!headless_mode)
    {
        cout << "按 'q' 或 ESC 退出, 按 's' 截图, 按 'r' 开始/停止录制" << endl;
    }
    else
    {
        cout << "无显示模式运行中..." << endl;
        if (duration_seconds > 0)
        {
            cout << "将录制 " << duration_seconds << " 秒" << endl;
        }
        else
        {
            cout << "按 Ctrl+C 停止录制" << endl;
        }
    }
    cout << "======================================\n" << endl;

    cv::Mat frame;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (true)
    {
        // 读取帧
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        cap >> frame;
        if (frame.empty())
        {
            fprintf(stderr, "错误: 无法读取摄像头帧\n");
            break;
        }

        frame_count++;

        // 执行检测
        auto infer_start = std::chrono::high_resolution_clock::now();
        
        std::vector<Object> objects;
        detect_frame(fastestdet, frame, objects, target_size, prob_threshold, nms_threshold);
        
        auto infer_end = std::chrono::high_resolution_clock::now();
        float inference_time = std::chrono::duration<float, std::milli>(infer_end - infer_start).count();

        // 绘制检测结果
        draw_objects(frame, objects, class_names);

        // 计算总帧时间(包括读取、推理、显示等)
        auto frame_end = std::chrono::high_resolution_clock::now();
        float total_frame_time = std::chrono::duration<float, std::milli>(frame_end - frame_start).count();
        
        // 更新FPS计数器
        fps_counter.update(total_frame_time);

        // 添加信息文本
        char fps_text[256];
        sprintf(fps_text, "FPS: %.1f | Infer: %.1f ms | Objects: %zu", 
                fps_counter.getFPS(), inference_time, objects.size());
        
        cv::putText(frame, fps_text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        // 添加分辨率信息
        char res_text[128];
        sprintf(res_text, "Resolution: %dx%d", actual_width, actual_height);
        cv::putText(frame, res_text, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);

        // 添加录制状态
        if (is_recording || save_output)
        {
            cv::circle(frame, cv::Point(actual_width - 30, 30), 10, cv::Scalar(0, 0, 255), -1);
            cv::putText(frame, "REC", cv::Point(actual_width - 80, 35),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }

        // 检查录制时长（headless模式）
        if (headless_mode && duration_seconds > 0)
        {
            auto current_time = std::chrono::high_resolution_clock::now();
            float elapsed_seconds = std::chrono::duration<float>(current_time - start_time).count();
            if (elapsed_seconds >= duration_seconds)
            {
                cout << "\n达到设定录制时长 " << duration_seconds << " 秒，停止录制." << endl;
                break;
            }
        }

        // 保存视频帧
        if (save_output || is_recording)
        {
            if (!writer.isOpened())
            {
                int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                writer.open(output_path, fourcc, camera_fps, cv::Size(actual_width, actual_height), true);
                if (!writer.isOpened())
                {
                    fprintf(stderr, "警告: 无法创建输出视频文件: %s\n", output_path.c_str());
                    save_output = false;
                    is_recording = false;
                }
                else
                {
                    is_recording = true;
                    if (headless_mode)
                    {
                        cout << "开始录制到: " << output_path << endl;
                    }
                }
            }
            if (writer.isOpened())
            {
                writer.write(frame);
            }
        }

        // 显示结果（非headless模式）
        if (!headless_mode)
        {
            cv::imshow("FastestDet Camera Inference", frame);

            // 处理键盘输入
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27)  // 'q' 或 ESC
            {
                cout << "\n用户停止程序." << endl;
                break;
            }
            else if (key == 's' || key == 'S')  // 截图
            {
                char filename[128];
                sprintf(filename, "screenshot_%04d.jpg", screenshot_count++);
                cv::imwrite(filename, frame);
                cout << "截图保存: " << filename << endl;
            }
            else if (key == 'r' || key == 'R')  // 开始/停止录制
            {
                if (!is_recording)
                {
                    is_recording = true;
                    cout << "开始录制视频: " << output_path << endl;
                }
                else
                {
                    is_recording = false;
                    if (writer.isOpened())
                    {
                        writer.release();
                        cout << "停止录制, 视频已保存: " << output_path << endl;
                    }
                }
            }
        }

        // 每100帧打印一次统计信息
        if (frame_count % 100 == 0)
        {
            cout << "已处理帧数: " << frame_count 
                 << " | 平均FPS: " << std::fixed << std::setprecision(1) << fps_counter.getFPS()
                 << " | 平均推理时间: " << std::setprecision(1) << inference_time << " ms"
                 << endl;
        }
    }

    // 释放资源
    cap.release();
    if (writer.isOpened())
    {
        writer.release();
        cout << "视频已保存: " << output_path << endl;
    }
    if (!headless_mode)
    {
        cv::destroyAllWindows();
    }

    // 打印统计信息
    cout << "\n======================================" << endl;
    cout << "推理完成!" << endl;
    cout << "======================================" << endl;
    cout << "总处理帧数: " << frame_count << endl;
    cout << "平均FPS: " << std::fixed << std::setprecision(1) << fps_counter.getFPS() << endl;
    cout << "平均帧处理时间: " << std::setprecision(1) << fps_counter.getAvgTime() << " ms" << endl;
    if (screenshot_count > 0)
    {
        cout << "截图数量: " << screenshot_count << endl;
    }
    cout << "======================================" << endl;

    return 0;
}
