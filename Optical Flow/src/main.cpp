#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "CUfarneback.h"
#include "SEfarneback.h"

// 帮助信息
void showHelp() {
    std::cout << "Farneback Optical Flow Implementation Comparison" << std::endl;
    std::cout << "Usage: program [mode] [video_path]" << std::endl;
    std::cout << "  mode: 1 - OpenCV implementation (default)" << std::endl;
    std::cout << "        2 - CUDA parallel implementation" << std::endl;
    std::cout << "        3 - Compare both implementations and calculate speedup" << std::endl;
    std::cout << "        4 - Sequential implementation (SEfarneback)" << std::endl;
    std::cout << "        5 - Compare SEfarneback and CUfarneback implementations" << std::endl;
    std::cout << "  video_path: Path to the input video file" << std::endl;
}

int main(int argc, char** argv) {
    // 处理命令行参数
    int mode = 1; // 默认使用OpenCV实现
    std::string video_path = "H:/Project/Parallel_Computing_HW/Optical Flow/data/testspeedup.mp4"; // 默认视频路径
    if (argc > 1) {
        mode = std::atoi(argv[1]);
        if (mode < 1 || mode > 5) {
            showHelp();
            return -1;
        }
    }
    
    if (argc > 2) {
        video_path = argv[2];
    }
    
    std::cout << "Running in mode " << mode << ": ";
    switch (mode) {
        case 1: std::cout << "OpenCV implementation" << std::endl; break;
        case 2: std::cout << "CUDA parallel implementation" << std::endl; break;
        case 3: std::cout << "Comparing implementations and calculating speedup" << std::endl; break;
        case 4: std::cout << "Sequential implementation (SEfarneback)" << std::endl; break;
        case 5: std::cout << "Comparing SEfarneback and CUfarneback implementations" << std::endl; break;
    }
    
    // Open video file
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return -1;
    }

    // Read first frame
    cv::Mat prev_frame, prev_gray;
    cap >> prev_frame;
    if (prev_frame.empty()) {
        std::cerr << "Failed to read first frame" << std::endl;
        return -1;
    }

    // 直接转换为灰度图，不做任何预处理
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
        case 2: window_title = "CUDA Parallel Farneback Optical Flow"; break;
        case 3: window_title = "Optical Flow Comparison"; break;
        case 4: window_title = "Sequential Farneback Optical Flow"; break;
        case 5: window_title = "SEfarneback vs CUfarneback Comparison"; break;
    }
    
    if (mode == 3 || mode == 5) {
        // 比较模式下，先用一种方法处理所有帧，再用另一种方法处理
        std::vector<double> times_method1;
        std::vector<double> times_method2;
        double total_time_method1 = 0.0;
        double total_time_method2 = 0.0;
        std::string method1_name, method2_name;
        
        if (mode == 3) {
            method1_name = "OpenCV";
            method2_name = "CUDA";
        } else { // mode == 5
            method1_name = "SEfarneback";
            method2_name = "CUfarneback";
        }
        
        // 处理第一种方法
        std::cout << "Processing with " << method1_name << "..." << std::endl;
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // 重置视频到第一帧
        cap >> prev_frame;
        cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
        
        frame_count = 0;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);
            
            int64 start_time = cv::getTickCount();
            if (mode == 3) {
                cv::calcOpticalFlowFarneback(
                    prev_gray, curr_gray, flow,
                    pyr_scale, levels, winsize,
                    iterations, poly_n, poly_sigma, flags
                );
            } else { // mode == 5
                serialCalcOpticalFlowFarneback(
                    prev_gray, curr_gray, flow,
                    pyr_scale, levels, winsize,
                    iterations, poly_n, poly_sigma, flags
                );
            }
            int64 end_time = cv::getTickCount();
            
            double frame_time = (end_time - start_time) / cv::getTickFrequency();
            times_method1.push_back(frame_time);
            total_time_method1 += frame_time;
            
            curr_gray.copyTo(prev_gray);
            frame_count++;
            
            // 可选：显示进度
            if (frame_count % 10 == 0) {
                std::cout << "Processed " << frame_count << " frames with " << method1_name << std::endl;
            }
        }
        
        // 处理第二种方法
        std::cout << "Processing with " << method2_name << "..." << std::endl;
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // 重置视频到第一帧
        cap >> prev_frame;
        cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
        
        frame_count = 0;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);
            
            int64 start_time = cv::getTickCount();
            cudaCalcOpticalFlowFarneback(
                prev_gray, curr_gray, flow,
                pyr_scale, levels, winsize,
                iterations, poly_n, poly_sigma, flags
            );
            int64 end_time = cv::getTickCount();
            
            double frame_time = (end_time - start_time) / cv::getTickFrequency();
            times_method2.push_back(frame_time);
            total_time_method2 += frame_time;
            
            curr_gray.copyTo(prev_gray);
            frame_count++;
            
            // 可选：显示进度
            if (frame_count % 10 == 0) {
                std::cout << "Processed " << frame_count << " frames with " << method2_name << std::endl;
            }
        }
        
        // 去除最大和最小时间值
        if (times_method1.size() > 2 && times_method2.size() > 2) {
            // 对时间数组排序
            std::sort(times_method1.begin(), times_method1.end());
            std::sort(times_method2.begin(), times_method2.end());
            
            // 去除最大和最小值后重新计算总时间
            double adjusted_total_time1 = 0.0;
            double adjusted_total_time2 = 0.0;
            
            for (size_t i = 1; i < times_method1.size() - 1; i++) {
                adjusted_total_time1 += times_method1[i];
            }
            
            for (size_t i = 1; i < times_method2.size() - 1; i++) {
                adjusted_total_time2 += times_method2[i];
            }
            
            // 计算调整后的平均时间
            double avg_time_method1 = adjusted_total_time1 / (times_method1.size() - 2);
            double avg_time_method2 = adjusted_total_time2 / (times_method2.size() - 2);
            
            // 计算加速比
            double speedup = avg_time_method1 / avg_time_method2;
            
            // 输出结果到文件和控制台
            std::cout << "===== Performance Comparison =====" << std::endl;
            std::cout << method1_name << " average processing time: " << avg_time_method1 * 1000 << " ms" << std::endl;
            std::cout << method2_name << " average processing time: " << avg_time_method2 * 1000 << " ms" << std::endl;
            std::cout << "Speedup: " << speedup << "x" << std::endl;
            std::cout << "=================================" << std::endl;
            
            // 将结果输出到文件
            std::ofstream result_file("performance_results.txt", std::ios::app);
            if (result_file.is_open()) {
                result_file << "===== Performance Comparison =====" << std::endl;
                result_file << "Video: " << video_path << std::endl;
                result_file << "Parameters: levels=" << levels << ", winsize=" << winsize 
                            << ", iterations=" << iterations << ", poly_n=" << poly_n << std::endl;
                result_file << method1_name << " average processing time: " << avg_time_method1 * 1000 << " ms" << std::endl;
                result_file << method2_name << " average processing time: " << avg_time_method2 * 1000 << " ms" << std::endl;
                result_file << "Speedup: " << speedup << "x" << std::endl;
                result_file << "Frame count: " << frame_count << std::endl;
                result_file << "=================================" << std::endl;
                result_file.close();
                std::cout << "Results saved to performance_results.txt" << std::endl;
            }
        } else {
            std::cout << "Not enough frames to calculate meaningful statistics." << std::endl;
        }
          // 比较模式下直接退出，不需要可视化
        std::cout << "Performance comparison completed. Exiting program." << std::endl;
        // 释放资源
        cap.release();
        cv::destroyAllWindows();
        return 0;
    }
    
    while (true) {
        // 只有在非比较模式或比较模式完成后才执行此循环
        if (mode == 3 || mode == 5) {
            // 比较模式下，这部分用于可视化
            cap >> frame;
            if (frame.empty()) break;
            
            cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);
            
            // 使用CUDA方法进行可视化
            cudaCalcOpticalFlowFarneback(
                prev_gray, curr_gray, flow,
                pyr_scale, levels, winsize,
                iterations, poly_n, poly_sigma, flags
            );
            
            curr_gray.copyTo(prev_gray);
            frame_count++;
        } else {
            // 非比较模式下的正常处理
            // Read next frame
            cap >> frame;
            if (frame.empty()) break;
            
            // 直接转换为灰度图，不做任何预处理
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
                    
                case 2: // CUDA并行实现
                    cudaCalcOpticalFlowFarneback(
                        prev_gray, curr_gray, flow,
                        pyr_scale, levels, winsize,
                        iterations, poly_n, poly_sigma, flags
                    );
                    break;
                    
                case 4: // 串行实现(SEfarneback)
                    serialCalcOpticalFlowFarneback(
                        prev_gray, curr_gray, flow,
                        pyr_scale, levels, winsize,
                        iterations, poly_n, poly_sigma, flags
                    );
                    break;
            }
            int64 end_time = cv::getTickCount();
            proc_time += (end_time - start_time) / cv::getTickFrequency();
            
            frame_count++;
            
            // 更新上一帧
            curr_gray.copyTo(prev_gray);
        }
        
        // 可视化光流
        cv::Mat flow_parts[2], magnitude, angle;
        cv::split(flow, flow_parts);
        
        // 检查光流数据
        double min_x, max_x, min_y, max_y;
        cv::minMaxLoc(flow_parts[0], &min_x, &max_x);
        cv::minMaxLoc(flow_parts[1], &min_y, &max_y);
        
        if (frame_count % 30 == 0) {
            std::cout << "Flow range X: " << min_x << " to " << max_x << std::endl;
            std::cout << "Flow range Y: " << min_y << " to " << max_y << std::endl;
        }
        
        // 计算幅值和角度
        cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle);
        
        // 创建HSV图像
        cv::Mat hsv_image(flow.size(), CV_8UC3);
        std::vector<cv::Mat> hsv_planes;
        cv::split(hsv_image, hsv_planes);
        
        // 色调通道 - 角度
        // hsv[..., 0] = ang * 180 / np.pi / 2  # 色调表示方向
        cv::Mat hue;
        angle.convertTo(hue, CV_32F, 180 / (2 * CV_PI));  // 角度转换为0-90度
        hue.convertTo(hsv_planes[0], CV_8U);              // 转为8位
        
        // 饱和度通道 - 全设置为255
        hsv_planes[1] = cv::Mat::ones(angle.size(), CV_8UC1) * 255;
        
        // 亮度通道 - 使用归一化的幅度
        // hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        cv::normalize(magnitude, hsv_planes[2], 0, 255, cv::NORM_MINMAX);
        hsv_planes[2].convertTo(hsv_planes[2], CV_8UC1);
        
        // 合并通道并转换回BGR
        cv::merge(hsv_planes, hsv_image);
        cv::cvtColor(hsv_image, bgr, cv::COLOR_HSV2BGR);
        
        // 显示结果
        cv::imshow(window_title, bgr);
        
        // Print current performance stats
        if (frame_count % 10 == 0) {
            std::cout << "Average processing time: " << proc_time/frame_count * 1000 << " ms" << std::endl;
        }
        
        // Break on ESC key
        if (cv::waitKey(30) == 27) break;
    }
    
    // Print final performance stats (only for non-comparison modes)
    if (mode != 3 && mode != 5) {
        std::cout << "Final average processing time: " << proc_time/frame_count * 1000 << " ms" << std::endl;
    }
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}
