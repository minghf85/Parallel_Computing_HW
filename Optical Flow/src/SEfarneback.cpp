#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// 声明函数接口
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
    int flags);

// PI常量
#define M_PI 3.14159265358979323846

/******************************************************************************
 * 辅助函数
 ******************************************************************************/
// 创建高斯核系数
void createGaussianKernel(float sigma, int n, std::vector<float>& kernel) {
    kernel.resize(n * n);
    int radius = n / 2;
    float sum = 0.0f;
    
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            int idx = (y + radius) * n + (x + radius);
            kernel[idx] = value;
            sum += value;
        }
    }
    
    // 归一化
    if (sum > 0) {
        for (int i = 0; i < n * n; i++) {
            kernel[i] /= sum;
        }
    }
}

// 将字节图像转换为浮点格式
void convertToFloat(const cv::Mat& src, cv::Mat& dst) {
    if (src.type() == CV_8UC1) {
        src.convertTo(dst, CV_32F, 1.0/255.0);
    } else if (src.type() == CV_32FC1) {
        src.copyTo(dst);
    }
}

// 高斯平滑
void gaussianSmooth(const cv::Mat& src, cv::Mat& dst, const std::vector<float>& kernel, int ksize) {
    dst.create(src.size(), src.type());
    int radius = ksize / 2;
    int width = src.cols;
    int height = src.rows;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;
            
            for (int j = -radius; j <= radius; j++) {
                int ny = y + j;
                if (ny < 0 || ny >= height) continue;
                
                for (int i = -radius; i <= radius; i++) {
                    int nx = x + i;
                    if (nx < 0 || nx >= width) continue;
                    
                    float kernel_val = kernel[(j+radius) * ksize + (i+radius)];
                    sum += src.at<float>(ny, nx) * kernel_val;
                    weight_sum += kernel_val;
                }
            }
            
            // 归一化
            if (weight_sum > 1e-6f) {
                dst.at<float>(y, x) = sum / weight_sum;
            } else {
                dst.at<float>(y, x) = src.at<float>(y, x);
            }
        }
    }
}

// 增强对比度
void enhanceContrast(const cv::Mat& src, cv::Mat& dst) {
    dst.create(src.size(), src.type());
    int width = src.cols;
    int height = src.rows;
    
    float alpha = 1.3f; // 对比度因子
    float beta = -0.1f; // 亮度调整
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float pixel = src.at<float>(y, x);
            float enhanced = alpha * pixel + beta;
            dst.at<float>(y, x) = std::max(0.0f, std::min(1.0f, enhanced));
        }
    }
}

// 各向异性扩散滤波
void anisotropicDiffusion(const cv::Mat& src, cv::Mat& dst, float k) {
    dst.create(src.size(), src.type());
    int width = src.cols;
    int height = src.rows;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x == 0 || x == width-1 || y == 0 || y == height-1) {
                // 边界情况直接复制
                dst.at<float>(y, x) = src.at<float>(y, x);
                continue;
            }
            
            float center = src.at<float>(y, x);
            
            // 计算四个方向的梯度
            float north = src.at<float>(y-1, x) - center;
            float south = src.at<float>(y+1, x) - center;
            float west = src.at<float>(y, x-1) - center;
            float east = src.at<float>(y, x+1) - center;
            
            // 计算扩散系数
            float cn = expf(-(north*north) / (k*k));
            float cs = expf(-(south*south) / (k*k));
            float cw = expf(-(west*west) / (k*k));
            float ce = expf(-(east*east) / (k*k));
            
            // 更新中心像素值
            dst.at<float>(y, x) = center + 0.25f * (cn*north + cs*south + cw*west + ce*east);
        }
    }
}

// 图像金字塔下采样
void pyrDown(const cv::Mat& src, cv::Mat& dst, double scale) {
    int dstWidth = cvRound(src.cols * scale);
    int dstHeight = cvRound(src.rows * scale);
    dst.create(dstHeight, dstWidth, src.type());
    
    for (int y = 0; y < dstHeight; y++) {
        for (int x = 0; x < dstWidth; x++) {
            int srcX = x / scale;
            int srcY = y / scale;
            
            // 简单下采样 - 可以改进为双线性插值或其他方法
            dst.at<float>(y, x) = src.at<float>(srcY, srcX);
        }
    }
}

// 流场上采样
void pyrUpFlow(const cv::Mat& src, cv::Mat& dst, double scale) {
    int dstWidth = cvRound(src.cols / scale);
    int dstHeight = cvRound(src.rows / scale);
    dst.create(dstHeight, dstWidth, src.type());
    
    for (int y = 0; y < dstHeight; y++) {
        for (int x = 0; x < dstWidth; x++) {
            float srcX = x * scale;
            float srcY = y * scale;
            
            int x0 = floorf(srcX);
            int y0 = floorf(srcY);
            int x1 = std::min(x0 + 1, src.cols - 1);
            int y1 = std::min(y0 + 1, src.rows - 1);
            
            float wx = srcX - x0;
            float wy = srcY - y0;
            
            // 双线性插值，分别处理x和y分量
            cv::Vec2f& dstFlow = dst.at<cv::Vec2f>(y, x);
            const cv::Vec2f& src00 = src.at<cv::Vec2f>(y0, x0);
            const cv::Vec2f& src01 = src.at<cv::Vec2f>(y0, x1);
            const cv::Vec2f& src10 = src.at<cv::Vec2f>(y1, x0);
            const cv::Vec2f& src11 = src.at<cv::Vec2f>(y1, x1);
            
            dstFlow[0] = (1.0f/scale) * ((1.0f - wx) * (1.0f - wy) * src00[0] +
                           wx * (1.0f - wy) * src01[0] +
                           (1.0f - wx) * wy * src10[0] +
                           wx * wy * src11[0]);
            
            dstFlow[1] = (1.0f/scale) * ((1.0f - wx) * (1.0f - wy) * src00[1] +
                           wx * (1.0f - wy) * src01[1] +
                           (1.0f - wx) * wy * src10[1] +
                           wx * wy * src11[1]);
        }
    }
}

// 多项式展开
void polyExpansion(const cv::Mat& src, std::vector<cv::Mat>& dst, 
                  int poly_n, const std::vector<float>& kernel) {
    int width = src.cols;
    int height = src.rows;
    int radius = poly_n / 2;
    
    // 确保输出矩阵大小正确
    dst.resize(6);
    for (int i = 0; i < 6; i++) {
        dst[i].create(height, width, CV_32F);
    }
    
    // 针对每个像素计算多项式系数
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 多项式系数
            float sum_w = 0.0f;
            float sum_wx = 0.0f, sum_wy = 0.0f;
            float sum_wxx = 0.0f, sum_wxy = 0.0f, sum_wyy = 0.0f;
            float sum_wI = 0.0f, sum_wxI = 0.0f, sum_wyI = 0.0f;
            
            // 遍历窗口
            for (int j = -radius; j <= radius; j++) {
                int ny = y + j;
                if (ny < 0 || ny >= height) continue;
                
                for (int i = -radius; i <= radius; i++) {
                    int nx = x + i;
                    if (nx < 0 || nx >= width) continue;
                    
                    // 获取窗口内像素和权重
                    float pixel_val = src.at<float>(ny, nx);
                    int kidx = (j + radius) * poly_n + (i + radius);
                    float weight = kernel[kidx];
                    
                    // 累加加权项
                    sum_w += weight;
                    sum_wx += weight * i;
                    sum_wy += weight * j;
                    sum_wxx += weight * i * i;
                    sum_wxy += weight * i * j;
                    sum_wyy += weight * j * j;
                    sum_wI += weight * pixel_val;
                    sum_wxI += weight * i * pixel_val;
                    sum_wyI += weight * j * pixel_val;
                }
            }
            
            // 防止除零
            if (sum_w < 1e-6f) sum_w = 1e-6f;
            
            // 存储多项式系数 [1, x, y, x^2, xy, y^2]
            dst[0].at<float>(y, x) = sum_wI / sum_w;                               // 常数项
            dst[1].at<float>(y, x) = sum_wxI / sum_w;                              // x系数
            dst[2].at<float>(y, x) = sum_wyI / sum_w;                              // y系数
            dst[3].at<float>(y, x) = (sum_wxx - sum_wx * sum_wx / sum_w) / sum_w;  // x^2系数
            dst[4].at<float>(y, x) = (sum_wxy - sum_wx * sum_wy / sum_w) / sum_w;  // xy系数
            dst[5].at<float>(y, x) = (sum_wyy - sum_wy * sum_wy / sum_w) / sum_w;  // y^2系数
        }
    }
}

// 计算光流
void computeFlow(const std::vector<cv::Mat>& poly1, const std::vector<cv::Mat>& poly2, 
                cv::Mat& flow, int winsize, const std::vector<float>& kernel) {
    int width = poly1[0].cols;
    int height = poly1[0].rows;
    int radius = winsize / 2;
    
    // 确保流场矩阵大小正确
    if (flow.empty() || flow.size() != cv::Size(width, height) || flow.type() != CV_32FC2) {
        flow.create(height, width, CV_32FC2);
    }
    
    // 针对每个像素计算光流
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 线性系统 A*x = b 的系数
            float A11 = 0, A12 = 0, A22 = 0;
            float b1 = 0, b2 = 0;
            
            // 遍历窗口
            for (int j = -radius; j <= radius; j++) {
                int ny = y + j;
                if (ny < 0 || ny >= height) continue;
                
                for (int i = -radius; i <= radius; i++) {
                    int nx = x + i;
                    if (nx < 0 || nx >= width) continue;
                    
                    int kidx = (j + radius) * winsize + (i + radius);
                    float weight = kernel[kidx];
                    
                    // 获取第一帧多项式系数
                    float c0 = poly1[0].at<float>(ny, nx);  // 常数项
                    float c1 = poly1[1].at<float>(ny, nx);  // x系数
                    float c2 = poly1[2].at<float>(ny, nx);  // y系数
                    
                    // 获取第二帧对应位置多项式系数
                    float c0_2 = poly2[0].at<float>(ny, nx);
                    
                    // 光流约束方程: c1*u + c2*v = c0_2 - c0
                    float diff = c0_2 - c0;
                    
                    // 构建加权最小二乘系统
                    A11 += weight * c1 * c1;
                    A12 += weight * c1 * c2;
                    A22 += weight * c2 * c2;
                    b1 += weight * c1 * diff;
                    b2 += weight * c2 * diff;
                }
            }
            
            // 求解2x2线性方程组
            float det = A11 * A22 - A12 * A12;
            cv::Vec2f& flowVec = flow.at<cv::Vec2f>(y, x);
            
            if (fabs(det) > 1e-6) {
                // 克莱默法则求解
                float u = (A22 * b1 - A12 * b2) / det;
                float v = (A11 * b2 - A12 * b1) / det;
                
                // 限制极端值
                if (isnan(u) || isinf(u)) u = 0;
                if (isnan(v) || isinf(v)) v = 0;
                
                const float MAX_FLOW = 10.0f;
                u = std::max(-MAX_FLOW, std::min(MAX_FLOW, u));
                v = std::max(-MAX_FLOW, std::min(MAX_FLOW, v));
                
                flowVec[0] = u;
                flowVec[1] = v;
            } else {
                // 奇异矩阵情况
                flowVec[0] = 0;
                flowVec[1] = 0;
            }
        }
    }
}

// 图像变形
void warpImage(const cv::Mat& src, cv::Mat& dst, const cv::Mat& flow) {
    int width = src.cols;
    int height = src.rows;
    dst.create(height, width, src.type());
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 计算反向映射坐标
            const cv::Vec2f& flowVec = flow.at<cv::Vec2f>(y, x);
            float srcX = x - flowVec[0];
            float srcY = y - flowVec[1];
            
            // 边界检查和双线性插值
            if (srcX >= 0 && srcX < width - 1 && srcY >= 0 && srcY < height - 1) {
                int x0 = floorf(srcX);
                int y0 = floorf(srcY);
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                
                float wx = srcX - x0;
                float wy = srcY - y0;
                
                // 双线性插值
                dst.at<float>(y, x) = (1 - wx) * (1 - wy) * src.at<float>(y0, x0) +
                                      wx * (1 - wy) * src.at<float>(y0, x1) +
                                      (1 - wx) * wy * src.at<float>(y1, x0) +
                                      wx * wy * src.at<float>(y1, x1);
            } else {
                // 超出边界，使用最近值
                int sx = std::min(std::max(0, (int)srcX), width - 1);
                int sy = std::min(std::max(0, (int)srcY), height - 1);
                dst.at<float>(y, x) = src.at<float>(sy, sx);
            }
        }
    }
}

// 更新流场
void updateFlow(cv::Mat& flow, const cv::Mat& update) {
    CV_Assert(flow.size() == update.size() && flow.type() == update.type());
    
    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {
            cv::Vec2f& flowVec = flow.at<cv::Vec2f>(y, x);
            const cv::Vec2f& updateVec = update.at<cv::Vec2f>(y, x);
            
            flowVec[0] += updateVec[0];
            flowVec[1] += updateVec[1];
        }
    }
}

// 光流场后处理 - 基于阈值的过滤
void thresholdFlow(cv::Mat& flow, float max_magnitude) {
    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {
            cv::Vec2f& flowVec = flow.at<cv::Vec2f>(y, x);
            float fx = flowVec[0];
            float fy = flowVec[1];
            float magnitude = sqrt(fx*fx + fy*fy);
            
            // 如果光流向量大小超过阈值，则缩放
            if (magnitude > max_magnitude) {
                float scale = max_magnitude / magnitude;
                flowVec[0] *= scale;
                flowVec[1] *= scale;
            }
        }
    }
}

// 光流场后处理 - 中值滤波
void medianFilterFlow(const cv::Mat& src, cv::Mat& dst, int kernel_size) {
    dst.create(src.size(), src.type());
    int radius = kernel_size / 2;
    
    // 分别对x和y分量进行中值滤波
    for (int c = 0; c < 2; c++) {
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                std::vector<float> values;
                values.reserve(kernel_size * kernel_size);
                
                // 收集邻域值
                for (int j = -radius; j <= radius; j++) {
                    int ny = y + j;
                    if (ny < 0 || ny >= src.rows) continue;
                    
                    for (int i = -radius; i <= radius; i++) {
                        int nx = x + i;
                        if (nx < 0 || nx >= src.cols) continue;
                        
                        values.push_back(src.at<cv::Vec2f>(ny, nx)[c]);
                    }
                }
                
                // 对收集的值进行排序
                std::sort(values.begin(), values.end());
                
                // 使用中值
                if (!values.empty()) {
                    dst.at<cv::Vec2f>(y, x)[c] = values[values.size() / 2];
                } else {
                    dst.at<cv::Vec2f>(y, x)[c] = src.at<cv::Vec2f>(y, x)[c];
                }
            }
        }
    }
}

/******************************************************************************
 * 主函数实现
 ******************************************************************************/
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
    int flags)
{
    // 检查输入参数
    if (prev_img.empty() || next_img.empty() || 
        prev_img.size() != next_img.size() || 
        prev_img.type() != next_img.type()) {
        std::cout << "Invalid input images" << std::endl;
        return false;
    }
    
    int width = prev_img.cols;
    int height = prev_img.rows;
    
    // 初始化流场
    if (flow.empty() || flow.size() != prev_img.size() || flow.type() != CV_32FC2) {
        flow.create(prev_img.size(), CV_32FC2);
    }
    
    if (!(flags & cv::OPTFLOW_USE_INITIAL_FLOW)) {
        flow.setTo(cv::Scalar::all(0));
    }
    
    // 转换输入图像到浮点格式
    cv::Mat prev_float, next_float;
    convertToFloat(prev_img, prev_float);
    convertToFloat(next_img, next_float);
    
    // 预处理输入图像
    cv::Mat prev_processed = prev_float.clone();
    cv::Mat next_processed = next_float.clone();
    
    // 步骤1: 高斯平滑
    std::vector<float> smooth_kernel;
    createGaussianKernel(1.0f, 5, smooth_kernel);
    gaussianSmooth(prev_processed, prev_processed, smooth_kernel, 5);
    gaussianSmooth(next_processed, next_processed, smooth_kernel, 5);
    
    // 步骤2: 增强对比度
    enhanceContrast(prev_processed, prev_processed);
    enhanceContrast(next_processed, next_processed);
    
    // 步骤3: 各向异性扩散
    const float k_param = 0.02f;
    anisotropicDiffusion(prev_processed, prev_processed, k_param);
    anisotropicDiffusion(next_processed, next_processed, k_param);
    
    // 构建图像金字塔
    std::vector<cv::Mat> prev_pyr(levels), next_pyr(levels);
    std::vector<cv::Mat> flow_pyr(levels);
    std::vector<int> pyr_widths(levels), pyr_heights(levels);
    
    // 初始化金字塔第0层
    prev_pyr[0] = prev_processed.clone();
    next_pyr[0] = next_processed.clone();
    pyr_widths[0] = width;
    pyr_heights[0] = height;
    
    // 构建图像金字塔
    for (int i = 1; i < levels; i++) {
        pyr_widths[i] = cvRound(pyr_widths[i-1] * pyr_scale);
        pyr_heights[i] = cvRound(pyr_heights[i-1] * pyr_scale);
        
        // 下采样
        pyrDown(prev_pyr[i-1], prev_pyr[i], pyr_scale);
        pyrDown(next_pyr[i-1], next_pyr[i], pyr_scale);
    }
    
    // 创建高斯核
    std::vector<float> poly_kernel, win_kernel;
    createGaussianKernel(poly_sigma, poly_n, poly_kernel);
    createGaussianKernel(0.15 * winsize, winsize, win_kernel);
    
    // 从最顶层向下优化光流场
    for (int level = levels - 1; level >= 0; level--) {
        int w = pyr_widths[level];
        int h = pyr_heights[level];
        std::cout << "Processing level " << level << ": " << w << "x" << h << std::endl;
        
        // 计算当前层多项式系数
        std::vector<cv::Mat> poly_prev, poly_next;
        polyExpansion(prev_pyr[level], poly_prev, poly_n, poly_kernel);
        
        // 初始化或传播流场
        if (level == levels - 1) {
            // 顶层初始化为0
            flow_pyr[level].create(h, w, CV_32FC2);
            flow_pyr[level].setTo(cv::Scalar::all(0));
        } else {
            // 上采样上一层的流场
            pyrUpFlow(flow_pyr[level+1], flow_pyr[level], pyr_scale);
        }
        
        // 迭代优化当前层流场
        for (int iter = 0; iter < iterations; iter++) {
            // 使用当前流场变形第二帧
            cv::Mat warped_next;
            warpImage(next_pyr[level], warped_next, flow_pyr[level]);
            
            // 计算变形后图像的多项式展开
            polyExpansion(warped_next, poly_next, poly_n, poly_kernel);
            
            // 计算流场更新
            cv::Mat flow_update;
            computeFlow(poly_prev, poly_next, flow_update, winsize, win_kernel);
            
            // 更新流场
            updateFlow(flow_pyr[level], flow_update);
            
            // 输出调试信息
            if (level == 0) {
                std::cout << "  Level 0, Iteration " << iter << " completed" << std::endl;
            }
        }
    }
    
    // 复制最终结果
    flow_pyr[0].copyTo(flow);
    
    // 后处理
    thresholdFlow(flow, 20.0f);  // 限制最大光流大小
    
    cv::Mat filtered_flow;
    medianFilterFlow(flow, filtered_flow, 3);  // 中值滤波平滑
    filtered_flow.copyTo(flow);
    
    return true;
}
