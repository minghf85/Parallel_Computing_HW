#include "CUfarneback.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// CUDA错误检查宏
#define CUDA_CHECK(call) { const cudaError_t error = call; if (error != cudaSuccess) { \
    printf("CUDA Error: %s:%d, ", __FILE__, __LINE__); \
    printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
    return false; }}

// 通用块大小
#define BLOCK_SIZE 16

/******************************************************************************
 * 内存管理辅助函数
 ******************************************************************************/
template <typename T>
bool allocAndUpload(const cv::Mat& src, T*& dst) {
    size_t size = src.cols * src.rows * src.elemSize();
    CUDA_CHECK(cudaMalloc(&dst, size));
    CUDA_CHECK(cudaMemcpy(dst, src.ptr(), size, cudaMemcpyHostToDevice));
    return true;
}

template <typename T>
bool downloadAndFree(T* src, cv::Mat& dst) {
    size_t size = dst.cols * dst.rows * dst.elemSize();
    CUDA_CHECK(cudaMemcpy(dst.ptr(), src, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(src));
    return true;
}

/******************************************************************************
 * 基础图像处理核函数
 ******************************************************************************/
// 转换字节图像到浮点格式
__global__ void convertToFloatKernel(const unsigned char* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = static_cast<float>(input[idx]) / 255.0f;
    }
}

// 创建高斯核系数
void createGaussianKernel(float sigma, int n, float* kernel) {
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

/******************************************************************************
 * 多尺度金字塔核函数
 ******************************************************************************/
// 金字塔下采样核函数
__global__ void pyrDownKernel(const float* src, float* dst, int srcWidth, int srcHeight, int dstWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int dstHeight = srcHeight / 2;
    
    if (x < dstWidth && y < dstHeight) {
        int srcX = x * 2;
        int srcY = y * 2;
        int srcIdx = srcY * srcWidth + srcX;
        int dstIdx = y * dstWidth + x;
        
        // 简单2x2平均下采样
        float sum = src[srcIdx] + 
                   src[srcIdx + 1] + 
                   src[srcIdx + srcWidth] + 
                   src[srcIdx + srcWidth + 1];
        dst[dstIdx] = sum * 0.25f;
    }
}

// 流场上采样核函数 - 双线性插值
__global__ void pyrUpFlowKernel(const float* src, float* dst, int srcWidth, int srcHeight, int dstWidth, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int dstHeight = srcHeight * 2;
    
    if (x < dstWidth && y < dstHeight) {
        float srcX = x / 2.0f;
        float srcY = y / 2.0f;
        
        int x0 = floorf(srcX);
        int y0 = floorf(srcY);
        int x1 = min(x0 + 1, srcWidth - 1);
        int y1 = min(y0 + 1, srcHeight - 1);
        
        float wx = srcX - x0;
        float wy = srcY - y0;
        
        // 双线性插值，分别处理x和y分量
        for (int c = 0; c < 2; c++) {
            int srcIdx00 = (y0 * srcWidth + x0) * 2 + c;
            int srcIdx01 = (y0 * srcWidth + x1) * 2 + c;
            int srcIdx10 = (y1 * srcWidth + x0) * 2 + c;
            int srcIdx11 = (y1 * srcWidth + x1) * 2 + c;
            int dstIdx = (y * dstWidth + x) * 2 + c;
            
            dst[dstIdx] = scale * ((1.0f - wx) * (1.0f - wy) * src[srcIdx00] +
                              wx * (1.0f - wy) * src[srcIdx01] +
                              (1.0f - wx) * wy * src[srcIdx10] +
                              wx * wy * src[srcIdx11]);
        }
    }
}

/******************************************************************************
 * Farneback算法核函数
 ******************************************************************************/
// 多项式展开核函数
__global__ void polyExpansionKernel(const float* src, float* dst, int width, int height, 
                                  int poly_n, const float* g_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int radius = poly_n / 2;
        int coeffStride = width * height;
        float center_val = src[y * width + x];
        
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
                float pixel_val = src[ny * width + nx];
                int kidx = (j + radius) * poly_n + (i + radius);
                float weight = g_kernel[kidx];
                
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
        dst[0 * coeffStride + y * width + x] = sum_wI / sum_w;                             // 常数项
        dst[1 * coeffStride + y * width + x] = sum_wxI / sum_w;                            // x系数
        dst[2 * coeffStride + y * width + x] = sum_wyI / sum_w;                            // y系数
        dst[3 * coeffStride + y * width + x] = (sum_wxx - sum_wx * sum_wx / sum_w) / sum_w; // x^2系数
        dst[4 * coeffStride + y * width + x] = (sum_wxy - sum_wx * sum_wy / sum_w) / sum_w; // xy系数
        dst[5 * coeffStride + y * width + x] = (sum_wyy - sum_wy * sum_wy / sum_w) / sum_w; // y^2系数
    }
}

// 更新光流核函数
__global__ void updateFlowKernel(float* flow, const float* update, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 2;
        flow[idx] += update[idx];        // x分量
        flow[idx + 1] += update[idx + 1]; // y分量
    }
}

// 计算光流核函数
__global__ void computeFlowKernel(const float* poly1, const float* poly2, float* flow, 
                                int width, int height, int winsize, const float* g_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int radius = winsize / 2;
        int coeffStride = width * height;
        
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
                
                int nidx = ny * width + nx;
                int kidx = (j + radius) * winsize + (i + radius);
                float weight = g_kernel[kidx];
                
                // 获取第一帧多项式系数
                float c0 = poly1[0 * coeffStride + nidx]; // 常数项
                float c1 = poly1[1 * coeffStride + nidx]; // x系数
                float c2 = poly1[2 * coeffStride + nidx]; // y系数
                
                // 获取第二帧对应位置多项式系数
                float c0_2 = poly2[0 * coeffStride + nidx];
                
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
        int fidx = (y * width + x) * 2;
        
        if (fabs(det) > 1e-6) {
            // 克莱默法则求解
            float u = (A22 * b1 - A12 * b2) / det;
            float v = (A11 * b2 - A12 * b1) / det;
            
            // 限制极端值
            if (isnan(u) || isinf(u)) u = 0;
            if (isnan(v) || isinf(v)) v = 0;
            
            const float MAX_FLOW = 10.0f;
            u = fmaxf(-MAX_FLOW, fminf(MAX_FLOW, u));
            v = fmaxf(-MAX_FLOW, fminf(MAX_FLOW, v));
            
            flow[fidx] = u;
            flow[fidx + 1] = v;
        } else {
            // 奇异矩阵情况
            flow[fidx] = 0;
            flow[fidx + 1] = 0;
        }
    }
}

// 图像变形核函数 - 用于迭代中的图像变形
__global__ void warpImageKernel(const float* src, float* dst, const float* flow, 
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // 计算反向映射坐标
        int idx = y * width + x;
        float srcX = x - flow[idx * 2];
        float srcY = y - flow[idx * 2 + 1];
        
        // 边界检查和双线性插值
        if (srcX >= 0 && srcX < width - 1 && srcY >= 0 && srcY < height - 1) {
            int x0 = floorf(srcX);
            int y0 = floorf(srcY);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float wx = srcX - x0;
            float wy = srcY - y0;
            
            // 双线性插值
            dst[idx] = (1 - wx) * (1 - wy) * src[y0 * width + x0] +
                      wx * (1 - wy) * src[y0 * width + x1] +
                      (1 - wx) * wy * src[y1 * width + x0] +
                      wx * wy * src[y1 * width + x1];
        } else {
            // 超出边界，使用最近值
            int sx = min(max(0, (int)srcX), width - 1);
            int sy = min(max(0, (int)srcY), height - 1);
            dst[idx] = src[sy * width + sx];
        }
    }
}

/******************************************************************************
 * 主函数实现
 ******************************************************************************/
bool cudaCalcOpticalFlowFarneback(
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
        printf("Invalid input images\n");
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
    
    // 设备内存指针
    unsigned char *d_prev_byte = nullptr, *d_next_byte = nullptr;
    float *d_prev = nullptr, *d_next = nullptr;
    float *d_flow = nullptr;
    
    // CUDA流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // 金字塔数据结构
    std::vector<float*> prev_pyr(levels);
    std::vector<float*> next_pyr(levels);
    std::vector<float*> flow_pyr(levels);
    std::vector<float*> poly_prev(levels);
    std::vector<float*> poly_next(levels);
    std::vector<int> pyr_widths(levels);
    std::vector<int> pyr_heights(levels);
    
    // 处理输入图像
    if (prev_img.type() == CV_8UC1) {
        // 上传字节图像
        allocAndUpload(prev_img, d_prev_byte);
        allocAndUpload(next_img, d_next_byte);
        
        // 分配和转换为浮点格式
        CUDA_CHECK(cudaMalloc(&d_prev, width * height * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_next, width * height * sizeof(float)));
        
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        convertToFloatKernel<<<grid, block, 0, stream>>>(d_prev_byte, d_prev, width, height);
        convertToFloatKernel<<<grid, block, 0, stream>>>(d_next_byte, d_next, width, height);
        
        // 释放原始字节数据
        CUDA_CHECK(cudaFree(d_prev_byte));
        CUDA_CHECK(cudaFree(d_next_byte));
    } else if (prev_img.type() == CV_32FC1) {
        // 直接上传浮点图像
        allocAndUpload(prev_img, d_prev);
        allocAndUpload(next_img, d_next);
    } else {
        printf("Unsupported image format\n");
        return false;
    }
    
    // 为流场分配内存
    CUDA_CHECK(cudaMalloc(&d_flow, width * height * 2 * sizeof(float)));
    if (flags & cv::OPTFLOW_USE_INITIAL_FLOW) {
        CUDA_CHECK(cudaMemcpy(d_flow, flow.ptr(), width * height * 2 * sizeof(float), 
                            cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemset(d_flow, 0, width * height * 2 * sizeof(float)));
    }
    
    // 初始化金字塔第0层
    pyr_widths[0] = width;
    pyr_heights[0] = height;
    prev_pyr[0] = d_prev;
    next_pyr[0] = d_next;
    
    // 构建图像金字塔
    for (int i = 1; i < levels; i++) {
        pyr_widths[i] = cvRound(pyr_widths[i-1] * pyr_scale);
        pyr_heights[i] = cvRound(pyr_heights[i-1] * pyr_scale);
        
        // 分配金字塔层内存
        CUDA_CHECK(cudaMalloc(&prev_pyr[i], pyr_widths[i] * pyr_heights[i] * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&next_pyr[i], pyr_widths[i] * pyr_heights[i] * sizeof(float)));
        
        // 下采样
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((pyr_widths[i] + block.x - 1) / block.x, 
                 (pyr_heights[i] + block.y - 1) / block.y);
        
        pyrDownKernel<<<grid, block, 0, stream>>>(
            prev_pyr[i-1], prev_pyr[i], 
            pyr_widths[i-1], pyr_heights[i-1], 
            pyr_widths[i]);
        
        pyrDownKernel<<<grid, block, 0, stream>>>(
            next_pyr[i-1], next_pyr[i], 
            pyr_widths[i-1], pyr_heights[i-1], 
            pyr_widths[i]);
    }
    
    // 创建高斯核
    float *h_poly_kernel = new float[poly_n * poly_n];
    float *d_poly_kernel = nullptr;
    createGaussianKernel(poly_sigma, poly_n, h_poly_kernel);
    CUDA_CHECK(cudaMalloc(&d_poly_kernel, poly_n * poly_n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_poly_kernel, h_poly_kernel, poly_n * poly_n * sizeof(float), 
                       cudaMemcpyHostToDevice));
    
    float *h_win_kernel = new float[winsize * winsize];
    float *d_win_kernel = nullptr;
    createGaussianKernel(0.15 * winsize, winsize, h_win_kernel);
    CUDA_CHECK(cudaMalloc(&d_win_kernel, winsize * winsize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_win_kernel, h_win_kernel, winsize * winsize * sizeof(float), 
                       cudaMemcpyHostToDevice));
    
    // 从最顶层向下优化光流场
    for (int level = levels - 1; level >= 0; level--) {
        int w = pyr_widths[level];
        int h = pyr_heights[level];
        printf("Processing level %d: %dx%d\n", level, w, h);
        
        // 分配当前层多项式系数内存
        CUDA_CHECK(cudaMalloc(&poly_prev[level], w * h * 6 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&poly_next[level], w * h * 6 * sizeof(float)));
        
        // 初始化多项式内存
        CUDA_CHECK(cudaMemset(poly_prev[level], 0, w * h * 6 * sizeof(float)));
        CUDA_CHECK(cudaMemset(poly_next[level], 0, w * h * 6 * sizeof(float)));
        
        // 计算多项式展开
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        
        polyExpansionKernel<<<grid, block, 0, stream>>>(
            prev_pyr[level], poly_prev[level], w, h, poly_n, d_poly_kernel);
        
        // 初始化或传播当前层流场
        if (level == levels - 1) {
            // 顶层流场初始化为0
            CUDA_CHECK(cudaMalloc(&flow_pyr[level], w * h * 2 * sizeof(float)));
            CUDA_CHECK(cudaMemset(flow_pyr[level], 0, w * h * 2 * sizeof(float)));
        } else {
            // 上采样上一层的流场
            CUDA_CHECK(cudaMalloc(&flow_pyr[level], w * h * 2 * sizeof(float)));
            
            dim3 upblock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 upgrid((w + upblock.x - 1) / upblock.x, (h + upblock.y - 1) / upblock.y);
            
            float scale = 1.0f / pyr_scale;
            pyrUpFlowKernel<<<upgrid, upblock, 0, stream>>>(
                flow_pyr[level+1], flow_pyr[level],
                pyr_widths[level+1], pyr_heights[level+1],
                w, scale);
        }
        
        // 迭代优化当前层流场
        float* warped_next;
        CUDA_CHECK(cudaMalloc(&warped_next, w * h * sizeof(float)));
        
        for (int iter = 0; iter < iterations; iter++) {
            // 使用当前流场变形第二帧
            warpImageKernel<<<grid, block, 0, stream>>>(
                next_pyr[level], warped_next, flow_pyr[level], w, h);
            
            // 计算变形后图像的多项式展开
            CUDA_CHECK(cudaMemset(poly_next[level], 0, w * h * 6 * sizeof(float)));
            
            polyExpansionKernel<<<grid, block, 0, stream>>>(
                warped_next, poly_next[level], w, h, poly_n, d_poly_kernel);
            
            // 计算流场更新
            float* flow_update;
            CUDA_CHECK(cudaMalloc(&flow_update, w * h * 2 * sizeof(float)));
            CUDA_CHECK(cudaMemset(flow_update, 0, w * h * 2 * sizeof(float)));
            
            computeFlowKernel<<<grid, block, 0, stream>>>(
                poly_prev[level], poly_next[level],
                flow_update, w, h, winsize, d_win_kernel);
            
            // 更新流场
            updateFlowKernel<<<grid, block, 0, stream>>>(
                flow_pyr[level], flow_update, w, h);
            
            // 释放更新内存
            CUDA_CHECK(cudaFree(flow_update));
            
            // 输出调试信息
            if (level == 0) {
                printf("  Level 0, Iteration %d completed\n", iter);
            }
        }
        
        // 释放临时内存
        CUDA_CHECK(cudaFree(warped_next));
    }
    
    // 复制最终结果
    CUDA_CHECK(cudaMemcpy(d_flow, flow_pyr[0], width * height * 2 * sizeof(float), 
                       cudaMemcpyDeviceToDevice));
    
    // 下载结果到主机
    downloadAndFree(d_flow, flow);
    
    // 检查结果的最大值
    float* h_flow = new float[width * height * 2];
    memcpy(h_flow, flow.ptr(), width * height * 2 * sizeof(float));
    
    float max_flow = 0.0f;
    for (int i = 0; i < width * height * 2; i++) {
        max_flow = std::max(max_flow, std::abs(h_flow[i]));
    }
    
    printf("Final max flow magnitude: %f\n", max_flow);
    delete[] h_flow;
    
    // 释放内存
    for (int i = 0; i < levels; i++) {
        if (i > 0) {
            CUDA_CHECK(cudaFree(prev_pyr[i]));
            CUDA_CHECK(cudaFree(next_pyr[i]));
        }
        CUDA_CHECK(cudaFree(poly_prev[i]));
        CUDA_CHECK(cudaFree(poly_next[i]));
        CUDA_CHECK(cudaFree(flow_pyr[i]));
    }
    
    // 释放其他资源
    CUDA_CHECK(cudaFree(d_poly_kernel));
    CUDA_CHECK(cudaFree(d_win_kernel));
    delete[] h_poly_kernel;
    delete[] h_win_kernel;
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return true;
}
