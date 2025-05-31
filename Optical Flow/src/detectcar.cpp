#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "CUfarneback.h"

// 跟踪车辆信息的结构体
struct Vehicle
{
    int id;                         // 车辆ID
    cv::Point2f center;             // 当前位置
    std::vector<cv::Point2f> trace; // 轨迹
    bool counted;                   // 是否已计数
    int framesExisted;              // 存在的帧数

    Vehicle(int _id, cv::Point2f _center) : id(_id), center(_center), counted(false), framesExisted(1)
    {
        trace.push_back(_center);
    }
};

// 定义检测线的位置（百分比）
const double COUNT_LINE_POSITION = 0.5; // 在ROI区域的中间位置

int main()
{
    // 打开视频文件
    cv::VideoCapture cap("/root/autodl-tmp/Parallel_Computing_HW/Optical Flow/data/testspeedup.mp4");
    if (!cap.isOpened())
    {
        std::cerr << "can not open video" << std::endl;
        return -1;
    }

    // 获取视频帧的尺寸信息
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 创建ROI掩码 - 使用提供的多边形顶点
    cv::Mat roi = cv::Mat::zeros(cv::Size(frame_width, frame_height), CV_8UC1);
    std::vector<cv::Point> roi_points;
    roi_points.push_back(cv::Point(8, 182));
    roi_points.push_back(cv::Point(1899, 225));
    roi_points.push_back(cv::Point(1904, 471));
    roi_points.push_back(cv::Point(9, 423));
    roi_points.push_back(cv::Point(8, 180));

    // 使用多边形填充创建ROI区域
    std::vector<std::vector<cv::Point>> roi_contours;
    roi_contours.push_back(roi_points);
    cv::fillPoly(roi, roi_contours, cv::Scalar(255));

    // 创建视频写入器 - 修改编码器和输出路径
    cv::VideoWriter writer;
    // 使用H.264编码代替MJPG
    writer.open("/root/autodl-tmp/Parallel_Computing_HW/Optical Flow/data/detection_result.mp4",
                cv::VideoWriter::fourcc('H', '2', '6', '4'), fps,
                cv::Size(frame_width, frame_height));

    // 如果H.264不支持，尝试使用XVID或MP4V
    if (!writer.isOpened())
    {
        writer.open("/root/autodl-tmp/Parallel_Computing_HW/Optical Flow/data/detection_result.mp4",
                    cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps,
                    cv::Size(frame_width, frame_height));
    }

    // 如果仍然无法打开，尝试使用AVI格式+XVID编码
    if (!writer.isOpened())
    {
        writer.open("/root/autodl-tmp/Parallel_Computing_HW/Optical Flow/data/detection_result.avi",
                    cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps,
                    cv::Size(frame_width, frame_height));
        std::cout << "尝试使用AVI格式+XVID编码..." << std::endl;
    }

    if (!writer.isOpened())
    {
        std::cerr << "无法创建输出视频文件！" << std::endl;
        return -1;
    }

    // 变量初始化
    cv::Mat prev_frame, prev_gray;
    cv::Mat frame, gray;
    cv::Mat flow, flow_visual;
    std::vector<Vehicle> vehicles;
    int vehicle_counter = 0;
    int next_vehicle_id = 1;
    int frame_count = 0;

    // 确定计数线的y坐标
    int count_line_y = static_cast<int>(frame_height * COUNT_LINE_POSITION);

    // 读取第一帧
    cap >> prev_frame;
    if (prev_frame.empty())
    {
        std::cerr << "无法读取第一帧！" << std::endl;
        return -1;
    }
    cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);

    // 设置光流参数
    double pyr_scale = 0.5;
    int levels = 3;
    int winsize = 15;
    int iterations = 3;
    int poly_n = 5;
    double poly_sigma = 1.2;
    int flags = 0;

    std::cout << "开始处理视频..." << std::endl;

    // 主循环：处理每一帧
    while (true)
    {
        // 读取当前帧
        cap >> frame;
        if (frame.empty())
            break;
        frame_count++;

        // 转为灰度图
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // 使用CUDA Farneback计算光流
        cudaCalcOpticalFlowFarneback(prev_gray, gray, flow,
                                     pyr_scale, levels, winsize,
                                     iterations, poly_n, poly_sigma, flags);

        // 创建光流的可视化表示
        std::vector<cv::Mat> flow_channels(2);
        cv::split(flow, flow_channels);
        cv::Mat magnitude, angle;
        cv::cartToPolar(flow_channels[0], flow_channels[1], magnitude, angle);

        // 在ROI内部处理
        cv::Mat mag_roi;
        magnitude.copyTo(mag_roi, roi);

        // 阈值处理以检测移动物体
        cv::Mat motion_mask;
        double motion_threshold = 2.0; // 移动阈值，可根据实际情况调整
        cv::threshold(mag_roi, motion_mask, motion_threshold, 255, cv::THRESH_BINARY);
        motion_mask.convertTo(motion_mask, CV_8U);

        // 应用形态学操作以减少噪声并连接相邻区域
        int morph_size = 5;
        cv::Mat element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
            cv::Point(morph_size, morph_size));
        cv::morphologyEx(motion_mask, motion_mask, cv::MORPH_CLOSE, element);
        cv::morphologyEx(motion_mask, motion_mask, cv::MORPH_OPEN, element);

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(motion_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 分析轮廓以跟踪车辆
        std::vector<cv::RotatedRect> vehicle_boxes;
        std::vector<cv::Point2f> vehicle_centers;

        for (const auto &contour : contours)
        {
            double area = cv::contourArea(contour);
            if (area > 200)
            { // 过滤掉小面积的噪声
                cv::RotatedRect box = cv::minAreaRect(contour);
                vehicle_boxes.push_back(box);
                vehicle_centers.push_back(box.center);
            }
        }

        // 匹配新检测到的车辆与之前跟踪的车辆
        std::vector<bool> detected_vehicles(vehicle_centers.size(), false);

        // 更新现有车辆位置
        for (auto it = vehicles.begin(); it != vehicles.end();)
        {
            bool found_match = false;
            for (size_t i = 0; i < vehicle_centers.size(); i++)
            {
                if (detected_vehicles[i])
                    continue;

                float distance = cv::norm(it->center - vehicle_centers[i]);
                if (distance < 50)
                { // 距离阈值，可根据实际情况调整
                    // 更新车辆位置
                    it->center = vehicle_centers[i];
                    it->trace.push_back(vehicle_centers[i]);
                    it->framesExisted++;

                    // 检查是否穿过计数线
                    if (!it->counted)
                    {
                        int trace_size = it->trace.size();
                        if (trace_size >= 2)
                        {
                            cv::Point2f prev_pos = it->trace[trace_size - 2];
                            cv::Point2f curr_pos = it->trace[trace_size - 1];

                            // 检查是否穿过计数线（从上到下）
                            if (prev_pos.y < count_line_y && curr_pos.y >= count_line_y)
                            {
                                it->counted = true;
                                vehicle_counter++;
                                std::cout << "车辆计数: " << vehicle_counter << std::endl;
                            }
                        }
                    }

                    detected_vehicles[i] = true;
                    found_match = true;
                    break;
                }
            }

            // 如果当前帧没有匹配到或超过最大帧数则删除跟踪器
            if (!found_match)
            {
                if (it->framesExisted > 5)
                { // 至少跟踪5帧以避免误检
                    it = vehicles.erase(it);
                    continue;
                }
            }
            ++it;
        }

        // 添加新的未匹配车辆
        for (size_t i = 0; i < vehicle_centers.size(); i++)
        {
            if (!detected_vehicles[i])
            {
                vehicles.push_back(Vehicle(next_vehicle_id++, vehicle_centers[i]));
            }
        }

        // 绘制结果
        cv::Mat display_frame = frame.clone();

        // 绘制ROI区域
        cv::Mat roi_overlay;
        cv::cvtColor(roi, roi_overlay, cv::COLOR_GRAY2BGR);
        cv::addWeighted(display_frame, 1.0, roi_overlay, 0.2, 0, display_frame);

        // 绘制车辆边界框和ID
        for (const auto &vehicle_box : vehicle_boxes)
        {
            cv::Point2f vertices[4];
            vehicle_box.points(vertices);
            for (int i = 0; i < 4; i++)
            {
                cv::line(display_frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        }

        // 绘制每个车辆的轨迹
        for (const auto &vehicle : vehicles)
        {
            if (vehicle.trace.size() > 1)
            {
                for (size_t i = 1; i < vehicle.trace.size(); i++)
                {
                    cv::line(display_frame, vehicle.trace[i - 1], vehicle.trace[i],
                             cv::Scalar(0, 0, 255), 2);
                }
            }

            // 添加车辆ID
            cv::putText(display_frame, "ID:" + std::to_string(vehicle.id),
                        vehicle.center + cv::Point2f(10, 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

            // 检查是否应该计入车辆总数
            if (vehicle.counted && vehicle.id > vehicle_counter)
            {
                vehicle_counter = vehicle.id;
            }
        }

        // 更新通过线的车辆数量（确保不为0）
        if (next_vehicle_id > 1 && vehicle_counter == 0)
        {
            vehicle_counter = next_vehicle_id - 1; // 下一个ID减1等于已经分配的ID数量
        }

        // 绘制计数线
        cv::line(display_frame, cv::Point(0, count_line_y),
                 cv::Point(frame_width, count_line_y),
                 cv::Scalar(255, 0, 255), 2);

        // 显示计数结果
        std::string counter_text = "Vehicle Count: " + std::to_string(vehicle_counter);
        cv::putText(display_frame, counter_text, cv::Point(30, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

        // 显示和保存结果
        // cv::imshow("Vehicle Detection", display_frame);
        writer.write(display_frame);

        // 更新前一帧
        gray.copyTo(prev_gray);

        // 不需要等待键盘事件，但可以定期显示进度
        if (frame_count % 100 == 0)
        {
            std::cout << "已处理 " << frame_count << " 帧..." << std::endl;
        }
    }

    // 释放资源
    cap.release();
    writer.release();
    // cv::destroyAllWindows(); // 不需要此行，因为没有显示窗口

    std::cout << "处理完成，共检测到 " << vehicle_counter << " 辆车。" << std::endl;
    if (writer.get(cv::CAP_PROP_FOURCC) == cv::VideoWriter::fourcc('H', '2', '6', '4'))
    {
        std::cout << "输出视频已保存到: /root/autodl-tmp/Parallel_Computing_HW/Optical Flow/data/detection_result.mp4 (H264编码)" << std::endl;
    }
    else if (writer.get(cv::CAP_PROP_FOURCC) == cv::VideoWriter::fourcc('X', 'V', 'I', 'D'))
    {
        std::cout << "输出视频已保存到: /root/autodl-tmp/Parallel_Computing_HW/Optical Flow/data/detection_result.avi (XVID编码)" << std::endl;
    }
    else
    {
        std::cout << "输出视频已保存" << std::endl;
    }

    return 0;
}
