#pragma once

#include <opencv2/opencv.hpp>

/**
 * @brief CUDA实现的Farneback稠密光流算法
 * 
 * @param prev_img 前一帧灰度图像（单通道8位或浮点型）
 * @param next_img 当前帧灰度图像（必须与prev_img尺寸和类型相同）
 * @param flow 输出的光流场（CV_32FC2格式，每个像素包含x/y方向位移向量）
 * @param pyr_scale 金字塔缩放因子（0-1之间，建议0.5表示每层缩小一半）
 * @param levels 金字塔层数（1表示单层处理，数值越大能捕获更大位移但耗时增加）
 * @param winsize 多项式展开的滑动窗口尺寸（建议奇数如13/15/21）
 * @param iterations 每层金字塔的迭代优化次数（通常3-10次，影响精度和耗时）
 * @param poly_n 多项式展开的邻域大小（典型值5或7，越大对噪声越鲁棒但计算量增加）
 * @param poly_sigma 高斯平滑的标准差（通常1.1-1.5，控制邻域权重分布）
 * @param flags 算法标志位（可选以下组合）：
 *              - 0：默认处理模式
 *              - OPTFLOW_USE_INITIAL_FLOW：使用输入flow作为初始估计
 *              - OPTFLOW_FARNEBACK_GAUSSIAN：使用高斯加权窗口
 * @return true 计算成功，false 表示失败（如图像尺寸不匹配等）
 * 
 * 算法工作流程：
 * 1. 构建图像高斯金字塔（层级由levels参数控制）
 * 2. 在每层金字塔上，通过多项式展开近似局部图像结构
 * 3. 假设相邻帧间多项式系数存在线性关系，推导位移方程
 * 4. 从顶层（最粗糙）开始逐层向下优化光流场
 * 5. 将上层结果作为下层初始值，实现由粗到精的估计
 * 
 * 性能提示：
 * - 对640x480视频，GTX 1080Ti上典型处理时间约5-15ms/帧
 * - 增大winsize/poly_n会显著增加显存占用
 * - 多金字塔层级对大幅运动更有效，但会增加边缘模糊
 */

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
