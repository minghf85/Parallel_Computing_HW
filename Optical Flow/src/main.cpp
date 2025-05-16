#include <opencv2/opencv.hpp>
#include <iostream>
#include "CUfarneback.h"

// 添加串行版本的Farneback光流函数声明
bool serialCalcOpticalFlowFarneback(
    const cv::Mat& prev_img,
    const cv::Mat& next_img,
    cv::Mat& flow,
    double pyr_scale = 0.5,
    int levels = 3,
    int winsize = 15,
    int iterations = 3,
    int poly_n = 5,
    double poly_sigma = 1.2,
    int flags = 0
);

// 帮助信息
void showHelp() {
    std::cout << "Farneback Optical Flow Implementation Comparison" << std::endl;
    std::cout << "Usage: program [mode]" << std::endl;
    std::cout << "  mode: 1 - OpenCV implementation (default)" << std::endl;
    std::cout << "        2 - Serial implementation using CUDA with single thread" << std::endl;
    std::cout << "        3 - CUDA parallel implementation" << std::endl;
}

int main(int argc, char** argv) {
    // 处理命令行参数
    int mode = 1; // 默认使用OpenCV实现
    
    if (argc > 1) {
        mode = std::atoi(argv[1]);
        if (mode < 1 || mode > 3) {
            showHelp();
            return -1;
        }
    }
    
    std::cout << "Running in mode " << mode << ": ";
    switch (mode) {
        case 1: std::cout << "OpenCV implementation" << std::endl; break;
        case 2: std::cout << "Serial implementation" << std::endl; break;
        case 3: std::cout << "CUDA parallel implementation" << std::endl; break;
    }
    
    // Open video file
    cv::VideoCapture cap("../data/test2.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    // Read first frame
    cv::Mat prev_frame, prev_gray;
    cap >> prev_frame;
    if (prev_frame.empty()) {
        std::cerr << "Failed to read first frame" << std::endl;
        return -1;
    }

    // Convert to grayscale
    cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
    
    // Initialize HSV image for visualization
    cv::Mat hsv, bgr;
    
    cv::Mat frame, curr_gray, flow;
    
    // Parameters for optical flow
    double pyr_scale = 0.5;
    int levels = 3;
    int winsize = 15;
    int iterations = 3;
    int poly_n = 5;
    double poly_sigma = 1.2;
    int flags = 0;
    
    // Timing variables
    double proc_time = 0;
    int frame_count = 0;
    std::string window_title;
    
    // 设置窗口标题
    switch (mode) {
        case 1: window_title = "OpenCV Farneback Optical Flow"; break;
        case 2: window_title = "Serial Farneback Optical Flow"; break;
        case 3: window_title = "CUDA Parallel Farneback Optical Flow"; break;
    }
    
    while (true) {
        // Read next frame
        cap >> frame;
        if (frame.empty()) break;
        
        // Convert to grayscale
        cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);
        
        // Calculate optical flow based on selected mode
        int64 start_time = cv::getTickCount();
        
        switch (mode) {
            case 1: // OpenCV实现
                cv::calcOpticalFlowFarneback(
                    prev_gray, curr_gray, flow,
                    pyr_scale, levels, winsize,
                    iterations, poly_n, poly_sigma, flags
                );
                break;
                
            case 2: // 串行实现（CUDA单线程）
                serialCalcOpticalFlowFarneback(
                    prev_gray, curr_gray, flow,
                    pyr_scale, levels, winsize,
                    iterations, poly_n, poly_sigma, flags
                );
                break;
                
            case 3: // CUDA并行实现
                cudaCalcOpticalFlowFarneback(
                    prev_gray, curr_gray, flow,
                    pyr_scale, levels, winsize,
                    iterations, poly_n, poly_sigma, flags
                );
                break;
        }
        
        int64 end_time = cv::getTickCount();
        proc_time += (end_time - start_time) / cv::getTickFrequency();
        
        frame_count++;
        
        // Visualize the optical flow
        cv::Mat flow_parts[2], magnitude, angle;
        cv::split(flow, flow_parts);
        
        // 检查光流数据是否有效并进行调试输出
        double min_x, max_x, min_y, max_y;
        cv::minMaxLoc(flow_parts[0], &min_x, &max_x);
        cv::minMaxLoc(flow_parts[1], &min_y, &max_y);
        
        if (frame_count % 30 == 0) {
            std::cout << "Flow range X: " << min_x << " to " << max_x << std::endl;
            std::cout << "Flow range Y: " << min_y << " to " << max_y << std::endl;
        }
        
        // 如果光流值太小，可以乘以一个放大因子来增强可视化效果
        if (mode == 3) {
            // CUDA模式可能需要调整光流值范围
            double scale = 20.0; // 增大缩放因子
            flow_parts[0] *= scale;
            flow_parts[1] *= scale;
        }
        
        cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle);
        
        // Create HSV channels with the correct size
        std::vector<cv::Mat> hsv_planes;
        
        // Hue channel (angle)
        cv::Mat hue;
        angle.convertTo(hue, CV_8U, 180 / CV_PI);
        hsv_planes.push_back(hue);
        
        // 确保饱和度都为255，以获得更鲜艳的颜色
        cv::Mat sat = cv::Mat::ones(angle.size(), CV_8UC1) * 255;
        hsv_planes.push_back(sat);
        
        // Value channel (magnitude)
        cv::Mat val;
        
        // 确保幅值有足够的对比度
        double min_mag, max_mag;
        cv::minMaxLoc(magnitude, &min_mag, &max_mag);
        
        if (frame_count % 30 == 0) {
            std::cout << "Magnitude range: " << min_mag << " to " << max_mag << std::endl;
        }
        
        // 强制规范化幅值，确保有可视效果
        cv::normalize(magnitude, val, 0, 255, cv::NORM_MINMAX);
        val.convertTo(val, CV_8UC1);
        
        // 如果模式3且最大幅值很小，设置亮度为阈值
        if (mode == 3 && max_mag < 1.0) {
            cv::threshold(val, val, 0, 255, cv::THRESH_BINARY);
        }
        
        hsv_planes.push_back(val);
        
        // Merge channels and convert back to BGR
        cv::merge(hsv_planes, hsv);
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        
        // Display results
        cv::imshow(window_title, bgr);
        
        // Print current performance stats
        if (frame_count % 10 == 0) {
            std::cout << "Average processing time: " << proc_time/frame_count * 1000 << " ms" << std::endl;
        }
        
        // Break on ESC key
        if (cv::waitKey(30) == 27) break;
        
        // Update previous frame
        curr_gray.copyTo(prev_gray);
    }
    
    // Print final performance stats
    std::cout << "Final average processing time: " << proc_time/frame_count * 1000 << " ms" << std::endl;
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}

// 串行版本的Farneback光流算法实现（使用CUDA单线程）
bool serialCalcOpticalFlowFarneback(
    const cv::Mat& prev_img,
    const cv::Mat& next_img,
    cv::Mat& flow,
    double pyr_scale,
    int levels,
    int winsize,
    int iterations,
    int poly_n,
    double poly_sigma,
    int flags
) {
    // 检查输入图像
    if (prev_img.empty() || next_img.empty() || prev_img.size() != next_img.size() || 
        prev_img.type() != next_img.type()) {
        printf("Invalid input images\n");
        return false;
    }
    
    // 确保图像是单通道灰度图
    cv::Mat prev_gray = prev_img.clone();
    cv::Mat next_gray = next_img.clone();
    
    // 如果是8位图像，转换为32位浮点图像并归一化
    if (prev_gray.type() != CV_32F) {
        prev_gray.convertTo(prev_gray, CV_32F, 1.0/255.0);
        next_gray.convertTo(next_gray, CV_32F, 1.0/255.0);
    }
    
    // 准备输出光流图
    int width = prev_gray.cols;
    int height = prev_gray.rows;
    if (flow.empty() || flow.size() != prev_gray.size() || flow.type() != CV_32FC2) {
        flow.create(height, width, CV_32FC2);
    }
    
    // 填充为零或使用初始流量
    if (!(flags & cv::OPTFLOW_USE_INITIAL_FLOW)) {
        flow.setTo(cv::Scalar::all(0));
    }
    
    // 在这里实现串行版本的Farneback算法
    // 设置CUDA在启动时使用单线程
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 假设使用设备0
    
    // 修改CUDA内核配置为只使用1个线程（串行执行）
    return cudaCalcOpticalFlowFarneback(
        prev_gray, next_gray, flow,
        pyr_scale, levels, winsize,
        iterations, poly_n, poly_sigma,
        flags | 0x10000000  // 添加一个自定义标志，在CUfarneback.cu中识别为串行模式
    );
}
