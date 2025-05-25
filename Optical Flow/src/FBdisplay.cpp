#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "CUfarneback.h"

int main(int argc, char** argv) {
    // 参数解析
    int mode = 2; // 默认使用CUDA版本
    std::string video_path = "H:/Project/Parallel_Computing_HW/Optical Flow/data/increase_constrast.mp4"; // 默认视频路径
    
    // 解析命令行参数
    if (argc > 1) {
        mode = std::stoi(argv[1]);
    }
    
    if (argc > 2) {
        video_path = argv[2];
    }
    
    std::cout << "使用模式: " << (mode == 1 ? "CPU" : "CUDA") << std::endl;
    std::cout << "视频路径: " << video_path << std::endl;
    
    // 初始化视频捕获
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }
    
    cv::Mat prev_frame, frame;
    cap >> prev_frame;
    if (prev_frame.empty()) {
        std::cerr << "Error: Could not read first frame." << std::endl;
        return -1;
    }

    cv::Mat prev_gray;
    cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
    
    // 修复HSV图像初始化 - 确保正确的大小和类型匹配
    std::vector<cv::Mat> hsv_planes(3);
    for (int i = 0; i < 3; i++) {
        hsv_planes[i] = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
    }
    hsv_planes[1] = cv::Mat::ones(prev_frame.size(), CV_8UC1) * 255; // S通道固定为255
    
    cv::Mat hsv;
    cv::merge(hsv_planes, hsv);

    // 可视化模式：0=HSV, 1=箭头, 2=轨迹, 3=掩码
    int display_mode = 0;
    bool pause = false;
    cv::Mat flow, bgr_flow;
    cv::Mat magnitude, angle;
    std::string title;
    
    // 性能监控
    int frame_count = 0;
    double total_time = 0.0;
    int memory_check_interval = 30; // 每30帧检查一次内存

    while (true) {
        cv::Mat vis;
          if (!pause) {
            cap >> frame;
            if (frame.empty()) {
                break;  // 视频结束
            }

            cv::Mat curr_gray;
            cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);
            
            // 记录光流计算开始时间
            double start_time = static_cast<double>(cv::getTickCount());
            
            // 光流算法参数
            double pyr_scale = 0.5;
            int levels = 3;
            int winsize = 15;
            int iterations = 3;
            int poly_n = 5;
            double poly_sigma = 1.2;
            int flags = 0;
    
            // 清除之前的流场内存
            flow.release();

            // 根据mode选择使用CPU或GPU版本
            if (mode == 1) {
                // 使用OpenCV CPU版本
                cv::calcOpticalFlowFarneback(
                    prev_gray, curr_gray, flow,
                    pyr_scale, levels, winsize,
                    iterations, poly_n, poly_sigma, flags
                );
            } else {
                // 使用CUDA GPU版本
                cudaCalcOpticalFlowFarneback(
                    prev_gray, curr_gray, flow,
                    pyr_scale, levels, winsize,
                    iterations, poly_n, poly_sigma, flags
                );
            }
            
            // 计算处理时间
            double end_time = static_cast<double>(cv::getTickCount());
            double proc_time = (end_time - start_time) / cv::getTickFrequency() * 1000; // 毫秒
            total_time += proc_time;
            frame_count++;
            
            // 每隔一定帧数输出性能信息
            if (frame_count % memory_check_interval == 0) {
                size_t free_memory, total_memory;
                cudaMemGetInfo(&free_memory, &total_memory);
                
                double avg_time = total_time / frame_count;
                std::cout << "Frame: " << frame_count 
                          << ", Avg Time: " << avg_time << " ms"
                          << ", GPU Memory - Free: " << (free_memory/1024/1024) << " MB"
                          << ", Total: " << (total_memory/1024/1024) << " MB" << std::endl;
                
                // 检测内存泄漏
                static size_t last_free_memory = free_memory;
                if (last_free_memory > free_memory && 
                    (last_free_memory - free_memory) > 10*1024*1024) { // 超过10MB的变化
                    std::cout << "警告：检测到内存减少 " 
                              << (last_free_memory - free_memory)/1024/1024 
                              << " MB" << std::endl;
                }
                last_free_memory = free_memory;
            }
            
            // 更新参考帧 - 注意使用clone避免共享数据
            prev_gray = curr_gray.clone();
            curr_gray.release(); // 显式释放不再需要的内存

            // 转换为极坐标（幅值和角度）
            cv::Mat flow_parts[2];
            cv::split(flow, flow_parts);
            cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle);
            
            // 确保HSV通道与图像大小匹配
            if (hsv.size() != frame.size()) {
                // 重新初始化HSV通道
                for (int i = 0; i < 3; i++) {
                    hsv_planes[i] = cv::Mat::zeros(frame.size(), CV_8UC1);
                }
                hsv_planes[1] = cv::Mat::ones(frame.size(), CV_8UC1) * 255;
                cv::merge(hsv_planes, hsv);
            }
            
            // 更新HSV通道 - 确保类型匹配
            // 色调表示方向
            angle.convertTo(hsv_planes[0], CV_8UC1, 180 / (2 * CV_PI));
            
            // 亮度表示速度 - 确保结果是CV_8UC1
            cv::Mat norm_magnitude;
            cv::normalize(magnitude, norm_magnitude, 0, 255, cv::NORM_MINMAX);
            norm_magnitude.convertTo(hsv_planes[2], CV_8UC1);
            
            // 检查所有通道的尺寸和类型是否匹配
            bool channels_ok = true;
            cv::Size size = hsv_planes[0].size();
            int type = hsv_planes[0].type();
            
            for (int i = 0; i < 3; i++) {
                if (hsv_planes[i].size() != size || hsv_planes[i].type() != type) {
                    channels_ok = false;
                    break;
                }
            }
            
            if (channels_ok) {
                cv::merge(hsv_planes, hsv);
                cv::cvtColor(hsv, bgr_flow, cv::COLOR_HSV2BGR);
            } else {
                std::cerr << "channel not ok" << std::endl;
                break;
            }
        }

        // 根据显示模式生成不同可视化结果
        switch (display_mode) {
            case 0: {  // HSV颜色编码
                vis = bgr_flow.clone();
                title = "HSV Color Encoding (Press 1/2/3/4 to switch)";
                break;
            }
                
            case 1: {  // 箭头矢量场
                vis = frame.clone();
                // 绘制箭头网格
                int step = 16;  // 箭头网格步长
                for (int y = step/2; y < frame.rows; y += step) {
                    for (int x = step/2; x < frame.cols; x += step) {
                        // 获取光流向量
                        const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                        
                        // 画箭头
                        cv::line(
                            vis,
                            cv::Point(x, y),
                            cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
                            cv::Scalar(0, 255, 0),
                            1, cv::LINE_AA
                        );
                        cv::circle(vis, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), -1);
                    }
                }
                title = "Arrow Vector Field (Press 1/2/3/4 to switch)";
                break;
            }
                
            case 2: {  // 运动轨迹叠加
                vis = frame.clone();
                // 修复轨迹掩码创建
                std::vector<cv::Mat> mask_planes(3);
                for (int i = 0; i < 3; i++) {
                    mask_planes[i] = cv::Mat::zeros(frame.size(), CV_8UC1);
                }
                
                // 绿色轨迹
                mask_planes[1] = cv::Mat::ones(frame.size(), CV_8UC1) * 255;
                
                // 确保幅值是CV_8UC1
                cv::Mat flow_mag;
                cv::normalize(magnitude, flow_mag, 0, 255, cv::NORM_MINMAX);
                flow_mag.convertTo(mask_planes[0], CV_8UC1);
                
                // 合并轨迹掩码
                cv::Mat mask;
                cv::merge(mask_planes, mask);
                cv::cvtColor(mask, mask, cv::COLOR_HSV2BGR);
                cv::addWeighted(vis, 0.7, mask, 0.3, 0, vis);
                
                title = "Motion Trails (Press 1/2/3/4 to switch)";
                break;
            }
                
            case 3: {  // 二值运动掩码
                // 速度阈值=5
                cv::Mat mask;
                cv::threshold(magnitude, mask, 5, 255, cv::THRESH_BINARY);
                mask.convertTo(mask, CV_8UC1);
                vis = cv::Mat::zeros(frame.size(), frame.type());
                frame.copyTo(vis, mask);
                title = "Binary Motion Mask (Press 1/2/3/4 to switch)";
                break;
            }
        }

        // 显示结果和说明文字
        if (!vis.empty()) {
            cv::putText(vis, title, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            cv::imshow("Optical Flow Visualization", vis);
        }

        // 键盘控制
        int key = cv::waitKey(pause ? 0 : 30);
        if (key == 27)  // ESC退出
            break;
        else if (key == '1')  // 切换HSV模式
            display_mode = 0;
        else if (key == '2')  // 切换箭头模式
            display_mode = 1;
        else if (key == '3')  // 切换轨迹模式
            display_mode = 2;
        else if (key == '4')  // 切换掩码模式
            display_mode = 3;
        else if (key == ' ')  // 暂停/继续
            pause = !pause;
    }    // 释放资源
    cap.release();
    cv::destroyAllWindows();
    
    // 最终输出性能统计
    if (frame_count > 0) {
        std::cout << "处理完成。总帧数: " << frame_count 
                  << ", 平均每帧处理时间: " << (total_time / frame_count) << " ms" << std::endl;
    }
    
    // 清理CUDA环境
    cudaDeviceReset();
    
    return 0;
}
