#include "CUfarneback.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>

// CUDA错误检查宏 - 改进版本，在错误时释放所有资源
#define CUDA_CHECK(call) { const cudaError_t error = call; if (error != cudaSuccess) { \
    printf("CUDA Error: %s:%d, ", __FILE__, __LINE__); \
    printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
    return false; }}

// 用于资源跟踪和清理的全局结构体
struct CudaResources {
    std::vector<void*> allocated_memory;
    cudaStream_t stream;
    bool stream_created;
    
    CudaResources() : stream_created(false) {}
    
    // 添加内存指针到跟踪列表
    void addMemory(void* ptr) {
        if (ptr) {
            allocated_memory.push_back(ptr);
        }
    }
    
    // 安全分配内存并跟踪
    bool safeMalloc(void** ptr, size_t size) {
        cudaError_t error = cudaMalloc(ptr, size);
        if (error != cudaSuccess) {
            printf("CUDA Malloc Error: %s\n", cudaGetErrorString(error));
            return false;
        }
        addMemory(*ptr);
        return true;
    }
    
    // 创建和跟踪流
    bool createStream() {
        cudaError_t error = cudaStreamCreate(&stream);
        if (error != cudaSuccess) {
            printf("CUDA Stream Creation Error: %s\n", cudaGetErrorString(error));
            return false;
        }
        stream_created = true;
        return true;
    }
    
    // 释放所有资源
    void freeAll() {
        // 释放所有分配的内存
        for (void* ptr : allocated_memory) {
            if (ptr) {
                cudaFree(ptr);
            }
        }
        allocated_memory.clear();
        
        // 释放流
        if (stream_created) {
            cudaStreamDestroy(stream);
            stream_created = false;
        }
        
        // 确保所有CUDA操作完成
        cudaDeviceSynchronize();
    }
};

// 通用块大小
#define BLOCK_SIZE 16
#define M_PI       3.14159265358979323846   // pi

/******************************************************************************
 * 内存管理辅助函数
 ******************************************************************************/
template <typename T>
bool allocAndUpload(const cv::Mat& src, T*& dst, CudaResources& resources) {
    size_t size = src.cols * src.rows * src.elemSize();
    if (!resources.safeMalloc((void**)&dst, size)) {
        return false;
    }
    
    cudaError_t error = cudaMemcpy(dst, src.ptr(), size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
        return false;
    }
    return true;
}

template <typename T>
bool downloadAndFree(T* src, cv::Mat& dst) {
    if (!src) return false;
    
    size_t size = dst.cols * dst.rows * dst.elemSize();
    cudaError_t error = cudaMemcpy(dst.ptr(), src, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
        cudaFree(src);
        return false;
    }
    
    error = cudaFree(src);
    if (error != cudaSuccess) {
        printf("CUDA Free Error: %s\n", cudaGetErrorString(error));
        return false;
    }
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
 * 光流场后处理核函数
 ******************************************************************************/
// 基于阈值的过滤 - 去除异常值
__global__ void thresholdFlowKernel(float* flow, int width, int height, float max_magnitude) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 2;
        float fx = flow[idx];
        float fy = flow[idx + 1];
        float magnitude = sqrtf(fx*fx + fy*fy);
        
        // 如果光流向量大小超过阈值，则置为0或缩放
        if (magnitude > max_magnitude) {
            // 方法1: 置为0
            // flow[idx] = 0.0f;
            // flow[idx + 1] = 0.0f;
            
            // 方法2: 等比缩放到阈值
            float scale = max_magnitude / magnitude;
            flow[idx] *= scale;
            flow[idx + 1] *= scale;
        }
    }
}

// 运动方向过滤 - 基于方向一致性
__global__ void directionFilterKernel(float* flow, int width, int height, float angle_threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 2 && x < width-2 && y >= 2 && y < height-2) {
        int idx = (y * width + x) * 2;
        float fx = flow[idx];
        float fy = flow[idx + 1];
        
        if (abs(fx) < 1e-6 && abs(fy) < 1e-6) {
            return; // 静止点不处理
        }
        
        // 当前像素的运动方向角
        float angle = atan2f(fy, fx);
        
        // 计算邻域内角度一致性
        int inconsistent_count = 0;
        const int window = 3; // 3x3窗口
        
        for (int j = -window/2; j <= window/2; j++) {
            for (int i = -window/2; i <= window/2; i++) {
                if (i == 0 && j == 0) continue; // 跳过中心点
                
                int nidx = ((y+j) * width + (x+i)) * 2;
                float nfx = flow[nidx];
                float nfy = flow[nidx + 1];
                
                if (abs(nfx) < 1e-6 && abs(nfy) < 1e-6) {
                    continue; // 跳过静止点
                }
                
                float nangle = atan2f(nfy, nfx);
                float diff = abs(angle - nangle);
                
                // 调整为最小角度差
                while (diff > M_PI) diff = 2.0f * M_PI - diff;
                
                // 如果角度差超过阈值，计数加1
                if (diff > angle_threshold) {
                    inconsistent_count++;
                }
            }
        }
        
        // 如果不一致点较多，认为是异常点，进行平滑或置0
        const int inconsistency_threshold = 4;
        if (inconsistent_count > inconsistency_threshold) {
            // 可以选择置为0或平滑
            flow[idx] = 0.0f;
            flow[idx + 1] = 0.0f;
        }
    }
}

// 中值滤波核函数 - 局部窗口内的中值
__global__ void medianFilterKernel(const float* src_flow, float* dst_flow, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 共享内存用于排序
    extern __shared__ float shared_mem[];
    
    if (x < width && y < height) {
        int radius = kernel_size / 2;
        int idx = (y * width + x) * 2;
        
        // 为x和y分量分别进行中值滤波
        for (int c = 0; c < 2; c++) {
            // 收集邻域值到局部数组
            float values[25]; // 最大支持5x5窗口
            int count = 0;
            
            for (int j = -radius; j <= radius; j++) {
                int ny = y + j;
                if (ny < 0 || ny >= height) continue;
                
                for (int i = -radius; i <= radius; i++) {
                    int nx = x + i;
                    if (nx < 0 || nx >= width) continue;
                    
                    int nidx = (ny * width + nx) * 2 + c;
                    values[count++] = src_flow[nidx];
                }
            }
            
            // 简单冒泡排序找中值
            for (int i = 0; i < count - 1; i++) {
                for (int j = 0; j < count - i - 1; j++) {
                    if (values[j] > values[j+1]) {
                        float temp = values[j];
                        values[j] = values[j+1];
                        values[j+1] = temp;
                    }
                }
            }
            
            // 中值赋值
            dst_flow[idx + c] = values[count / 2];
        }
    }
}

// 均值滤波核函数 - 带权重的局部平均
__global__ void meanFilterKernel(const float* src_flow, float* dst_flow, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int radius = kernel_size / 2;
        int idx = (y * width + x) * 2;
        
        // 为x和y分量分别进行均值滤波
        for (int c = 0; c < 2; c++) {
            float sum = 0.0f;
            int count = 0;
            
            for (int j = -radius; j <= radius; j++) {
                int ny = y + j;
                if (ny < 0 || ny >= height) continue;
                
                for (int i = -radius; i <= radius; i++) {
                    int nx = x + i;
                    if (nx < 0 || nx >= width) continue;
                    
                    int nidx = (ny * width + nx) * 2 + c;
                    sum += src_flow[nidx];
                    count++;
                }
            }
            
            // 防止除零
            if (count > 0) {
                dst_flow[idx + c] = sum / count;
            } else {
                dst_flow[idx + c] = src_flow[idx + c];
            }
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

/******************************************************************************
 * 预处理优化核函数
 ******************************************************************************/
// 图像预处理 - 高斯平滑
__global__ void gaussianSmoothKernel(const float* src, float* dst, int width, int height, 
                                  const float* kernel, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        int radius = ksize / 2;
        
        for (int j = -radius; j <= radius; j++) {
            int ny = y + j;
            if (ny < 0 || ny >= height) continue;
            
            for (int i = -radius; i <= radius; i++) {
                int nx = x + i;
                if (nx < 0 || nx >= width) continue;
                
                float kernel_val = kernel[(j+radius) * ksize + (i+radius)];
                sum += src[ny * width + nx] * kernel_val;
                weight_sum += kernel_val;
            }
        }
        
        // 归一化
        if (weight_sum > 1e-6f) {
            dst[y * width + x] = sum / weight_sum;
        } else {
            dst[y * width + x] = src[y * width + x];
        }
    }
}

// 增强对比度和边缘
__global__ void enhanceContrastKernel(const float* src, float* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float pixel = src[idx];
        
        // 简单的对比度增强，可以根据需要调整参数
        float alpha = 1.3f; // 对比度因子
        float beta = -0.1f; // 亮度调整
        
        float enhanced = alpha * pixel + beta;
        dst[idx] = fmaxf(0.0f, fminf(1.0f, enhanced)); // 截断在[0,1]范围内
    }
}

// 直方图均衡化 - 局部区域
__global__ void localHistogramEqualizationKernel(const float* src, float* dst, int width, int height, int window_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 外部共享内存声明
    extern __shared__ float shared_data[];
    
    if (x < width && y < height) {
        int radius = window_size / 2;
        int idx = y * width + x;
        
        // 收集局部窗口内的像素值
        float min_val = 1.0f;
        float max_val = 0.0f;
        int count = 0;
        float hist[10] = {0}; // 简单的10个桶的直方图
        
        for (int j = -radius; j <= radius; j++) {
            int ny = y + j;
            if (ny < 0 || ny >= height) continue;
            
            for (int i = -radius; i <= radius; i++) {
                int nx = x + i;
                if (nx < 0 || nx >= width) continue;
                
                float val = src[ny * width + nx];
                min_val = fminf(min_val, val);
                max_val = fmaxf(max_val, val);
                
                // 添加到直方图
                int bin = min(9, int(val * 10));
                hist[bin]++;
                count++;
            }
        }
        
        // 计算累积分布函数
        float cdf[10] = {0};
        cdf[0] = hist[0];
        for (int i = 1; i < 10; i++) {
            cdf[i] = cdf[i-1] + hist[i];
        }
        
        // 归一化累积分布函数
        for (int i = 0; i < 10; i++) {
            cdf[i] /= count;
        }
        
        // 应用直方图均衡化
        float pixel = src[idx];
        int bin = min(9, int(pixel * 10));
        float new_val = cdf[bin];
        
        // 写入结果
        dst[idx] = new_val;
    }
}

// 各向异性扩散滤波 - 保留边缘的平滑
__global__ void anisotropicDiffusionKernel(const float* src, float* dst, int width, int height, float k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        if (x == 0 || x == width-1 || y == 0 || y == height-1) {
            // 边界情况直接复制
            dst[y * width + x] = src[y * width + x];
            return;
        }
        
        int idx = y * width + x;
        float center = src[idx];
        
        // 计算四个方向的梯度
        float north = src[(y-1) * width + x] - center;
        float south = src[(y+1) * width + x] - center;
        float west = src[y * width + (x-1)] - center;
        float east = src[y * width + (x+1)] - center;
        
        // 计算扩散系数
        float cn = expf(-(north*north) / (k*k));
        float cs = expf(-(south*south) / (k*k));
        float cw = expf(-(west*west) / (k*k));
        float ce = expf(-(east*east) / (k*k));
        
        // 更新中心像素值
        dst[idx] = center + 0.25f * (cn*north + cs*south + cw*west + ce*east);
    }
}

// 区域分割与运动聚类 - 简化版本
__global__ void segmentAndClusterFlowKernel(float* flow, int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 2 && x < width-2 && y >= 2 && y < height-2) {
        int idx = (y * width + x) * 2;
        float fx = flow[idx];
        float fy = flow[idx+1];
        
        // 如果当前点是静止点，不处理
        if (fabs(fx) < 1e-3 && fabs(fy) < 1e-3) {
            return;
        }
        
        float mag = sqrtf(fx*fx + fy*fy);
        float angle = atan2f(fy, fx);
        
        // 统计邻域内相似的流向
        int count_similar = 0;
        float sum_fx = 0.0f, sum_fy = 0.0f;
        
        for (int j = -2; j <= 2; j++) {
            for (int i = -2; i <= 2; i++) {
                if (i == 0 && j == 0) continue;
                
                int nidx = ((y+j) * width + (x+i)) * 2;
                float nfx = flow[nidx];
                float nfy = flow[nidx+1];
                
                float nmag = sqrtf(nfx*nfx + nfy*nfy);
                float nangle = atan2f(nfy, nfx);
                
                // 计算方向和大小的差异
                float angle_diff = fabsf(angle - nangle);
                while (angle_diff > M_PI) angle_diff = 2.0f * M_PI - angle_diff;
                
                float mag_ratio = (mag > nmag) ? (nmag / mag) : (mag / nmag);
                
                // 如果方向和大小都相似
                if (angle_diff < threshold && mag_ratio > 0.7f) {
                    count_similar++;
                    sum_fx += nfx;
                    sum_fy += nfy;
                }
            }
        }
        
        // 如果有足够多相似的邻居，则使用邻居的平均流向
        if (count_similar > 8) { // 相当于至少一半邻居相似
            flow[idx] = (fx + sum_fx) / (count_similar + 1);
            flow[idx+1] = (fy + sum_fy) / (count_similar + 1);
        }
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
    
    // 创建资源管理器
    CudaResources resources;
    
    // 设备内存指针
    unsigned char *d_prev_byte = nullptr, *d_next_byte = nullptr;
    float *d_prev = nullptr, *d_next = nullptr;
    float *d_flow = nullptr;
    
    // 创建CUDA流
    if (!resources.createStream()) {
        resources.freeAll();
        return false;
    }
    
    // 金字塔数据结构
    std::vector<float*> prev_pyr(levels, nullptr);
    std::vector<float*> next_pyr(levels, nullptr);
    std::vector<float*> flow_pyr(levels, nullptr);
    std::vector<float*> poly_prev(levels, nullptr);
    std::vector<float*> poly_next(levels, nullptr);
    std::vector<int> pyr_widths(levels);
    std::vector<int> pyr_heights(levels);
    
    // 处理输入图像
    if (prev_img.type() == CV_8UC1) {
        // 上传字节图像
        if (!allocAndUpload(prev_img, d_prev_byte, resources) ||
            !allocAndUpload(next_img, d_next_byte, resources)) {
            resources.freeAll();
            return false;
        }
        
        // 分配和转换为浮点格式
        if (!resources.safeMalloc((void**)&d_prev, width * height * sizeof(float)) ||
            !resources.safeMalloc((void**)&d_next, width * height * sizeof(float))) {
            resources.freeAll();
            return false;
        }
        
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        convertToFloatKernel<<<grid, block, 0, resources.stream>>>(d_prev_byte, d_prev, width, height);
        convertToFloatKernel<<<grid, block, 0, resources.stream>>>(d_next_byte, d_next, width, height);
        
        // 检查内核执行错误
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        // 释放原始字节数据 - 由资源管理器处理
        cudaFree(d_prev_byte);
        cudaFree(d_next_byte);
        
        // 从资源跟踪列表中移除这些指针
        resources.allocated_memory.erase(
            std::remove(resources.allocated_memory.begin(), resources.allocated_memory.end(), d_prev_byte),
            resources.allocated_memory.end());
        resources.allocated_memory.erase(
            std::remove(resources.allocated_memory.begin(), resources.allocated_memory.end(), d_next_byte),
            resources.allocated_memory.end());
        
        d_prev_byte = nullptr;
        d_next_byte = nullptr;
        
        // 预处理输入图像以增强质量
        float* d_prev_temp = nullptr;
        float* d_next_temp = nullptr;
        if (!resources.safeMalloc((void**)&d_prev_temp, width * height * sizeof(float)) ||
            !resources.safeMalloc((void**)&d_next_temp, width * height * sizeof(float))) {
            resources.freeAll();
            return false;
        }
        
        // 步骤1: 高斯平滑降噪
        float* h_smooth_kernel = new float[5 * 5];
        float* d_smooth_kernel = nullptr;
        createGaussianKernel(1.0f, 5, h_smooth_kernel);
        
        if (!resources.safeMalloc((void**)&d_smooth_kernel, 5 * 5 * sizeof(float))) {
            delete[] h_smooth_kernel;
            resources.freeAll();
            return false;
        }
        
        error = cudaMemcpy(d_smooth_kernel, h_smooth_kernel, 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_smooth_kernel;
        
        if (error != cudaSuccess) {
            printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        gaussianSmoothKernel<<<grid, block, 0, resources.stream>>>(d_prev, d_prev_temp, width, height, d_smooth_kernel, 5);
        gaussianSmoothKernel<<<grid, block, 0, resources.stream>>>(d_next, d_next_temp, width, height, d_smooth_kernel, 5);
        
        // 检查内核执行错误
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        // 更新指针
        error = cudaMemcpy(d_prev, d_prev_temp, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess) {
            printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        error = cudaMemcpy(d_next, d_next_temp, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess) {
            printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        // 步骤2: 增强对比度
        enhanceContrastKernel<<<grid, block, 0, resources.stream>>>(d_prev, d_prev_temp, width, height);
        enhanceContrastKernel<<<grid, block, 0, resources.stream>>>(d_next, d_next_temp, width, height);
        
        // 检查内核执行错误
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        // 更新指针
        error = cudaMemcpy(d_prev, d_prev_temp, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess) {
            printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        error = cudaMemcpy(d_next, d_next_temp, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess) {
            printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        // 步骤3: 各向异性扩散滤波 - 保留边缘
        const float k_param = 0.02f; // 可调整的扩散参数
        anisotropicDiffusionKernel<<<grid, block, 0, resources.stream>>>(d_prev, d_prev_temp, width, height, k_param);
        anisotropicDiffusionKernel<<<grid, block, 0, resources.stream>>>(d_next, d_next_temp, width, height, k_param);
        
        // 检查内核执行错误
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        // 更新最终预处理结果
        error = cudaMemcpy(d_prev, d_prev_temp, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess) {
            printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        error = cudaMemcpy(d_next, d_next_temp, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess) {
            printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        // 释放临时内存 - 由资源管理器处理
        cudaFree(d_prev_temp);
        cudaFree(d_next_temp);
        cudaFree(d_smooth_kernel);
        
        // 从资源跟踪列表中移除这些指针
        resources.allocated_memory.erase(
            std::remove(resources.allocated_memory.begin(), resources.allocated_memory.end(), d_prev_temp),
            resources.allocated_memory.end());
        resources.allocated_memory.erase(
            std::remove(resources.allocated_memory.begin(), resources.allocated_memory.end(), d_next_temp),
            resources.allocated_memory.end());
        resources.allocated_memory.erase(
            std::remove(resources.allocated_memory.begin(), resources.allocated_memory.end(), d_smooth_kernel),
            resources.allocated_memory.end());
        
        d_prev_temp = nullptr;
        d_next_temp = nullptr;
        d_smooth_kernel = nullptr;
    } else if (prev_img.type() == CV_32FC1) {
        // 直接上传浮点图像
        if (!allocAndUpload(prev_img, d_prev, resources) ||
            !allocAndUpload(next_img, d_next, resources)) {
            resources.freeAll();
            return false;
        }
    } else {
        printf("Unsupported image format\n");
        resources.freeAll();
        return false;
    }
    
    // 为流场分配内存
    if (!resources.safeMalloc((void**)&d_flow, width * height * 2 * sizeof(float))) {
        resources.freeAll();
        return false;
    }
    
    if (flags & cv::OPTFLOW_USE_INITIAL_FLOW) {
        cudaError_t error = cudaMemcpy(d_flow, flow.ptr(), width * height * 2 * sizeof(float), 
                            cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
    } else {
        cudaError_t error = cudaMemset(d_flow, 0, width * height * 2 * sizeof(float));
        if (error != cudaSuccess) {
            printf("CUDA Memset Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
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
        if (!resources.safeMalloc((void**)&prev_pyr[i], pyr_widths[i] * pyr_heights[i] * sizeof(float)) ||
            !resources.safeMalloc((void**)&next_pyr[i], pyr_widths[i] * pyr_heights[i] * sizeof(float))) {
            resources.freeAll();
            return false;
        }
        
        // 下采样
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((pyr_widths[i] + block.x - 1) / block.x, 
                 (pyr_heights[i] + block.y - 1) / block.y);
        
        pyrDownKernel<<<grid, block, 0, resources.stream>>>(
            prev_pyr[i-1], prev_pyr[i], 
            pyr_widths[i-1], pyr_heights[i-1], 
            pyr_widths[i]);
        
        pyrDownKernel<<<grid, block, 0, resources.stream>>>(
            next_pyr[i-1], next_pyr[i], 
            pyr_widths[i-1], pyr_heights[i-1], 
            pyr_widths[i]);
        
        // 检查内核执行错误
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
    }
    
    // 创建高斯核
    float *h_poly_kernel = new float[poly_n * poly_n];
    float *d_poly_kernel = nullptr;
    createGaussianKernel(poly_sigma, poly_n, h_poly_kernel);
    
    if (!resources.safeMalloc((void**)&d_poly_kernel, poly_n * poly_n * sizeof(float))) {
        delete[] h_poly_kernel;
        resources.freeAll();
        return false;
    }
    
    cudaError_t error = cudaMemcpy(d_poly_kernel, h_poly_kernel, poly_n * poly_n * sizeof(float), 
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        delete[] h_poly_kernel;
        printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
        resources.freeAll();
        return false;
    }
    
    float *h_win_kernel = new float[winsize * winsize];
    float *d_win_kernel = nullptr;
    createGaussianKernel(0.15 * winsize, winsize, h_win_kernel);
    
    if (!resources.safeMalloc((void**)&d_win_kernel, winsize * winsize * sizeof(float))) {
        delete[] h_poly_kernel;
        delete[] h_win_kernel;
        resources.freeAll();
        return false;
    }
    
    error = cudaMemcpy(d_win_kernel, h_win_kernel, winsize * winsize * sizeof(float), 
                    cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        delete[] h_poly_kernel;
        delete[] h_win_kernel;
        printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
        resources.freeAll();
        return false;
    }
    
    delete[] h_poly_kernel;
    delete[] h_win_kernel;
    
    // 从最顶层向下优化光流场
    for (int level = levels - 1; level >= 0; level--) {
        int w = pyr_widths[level];
        int h = pyr_heights[level];
        printf("Processing level %d: %dx%d\n", level, w, h);
        
        // 分配当前层多项式系数内存
        if (!resources.safeMalloc((void**)&poly_prev[level], w * h * 6 * sizeof(float)) ||
            !resources.safeMalloc((void**)&poly_next[level], w * h * 6 * sizeof(float))) {
            resources.freeAll();
            return false;
        }
        
        // 初始化多项式内存
        error = cudaMemset(poly_prev[level], 0, w * h * 6 * sizeof(float));
        if (error != cudaSuccess) {
            printf("CUDA Memset Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        error = cudaMemset(poly_next[level], 0, w * h * 6 * sizeof(float));
        if (error != cudaSuccess) {
            printf("CUDA Memset Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        // 计算多项式展开
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        
        polyExpansionKernel<<<grid, block, 0, resources.stream>>>(
            prev_pyr[level], poly_prev[level], w, h, poly_n, d_poly_kernel);
        
        // 检查内核执行错误
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
            resources.freeAll();
            return false;
        }
        
        // 初始化或传播当前层流场
        if (level == levels - 1) {
            // 顶层流场初始化为0
            if (!resources.safeMalloc((void**)&flow_pyr[level], w * h * 2 * sizeof(float))) {
                resources.freeAll();
                return false;
            }
            
            error = cudaMemset(flow_pyr[level], 0, w * h * 2 * sizeof(float));
            if (error != cudaSuccess) {
                printf("CUDA Memset Error: %s\n", cudaGetErrorString(error));
                resources.freeAll();
                return false;
            }
        } else {
            // 上采样上一层的流场
            if (!resources.safeMalloc((void**)&flow_pyr[level], w * h * 2 * sizeof(float))) {
                resources.freeAll();
                return false;
            }
            
            dim3 upblock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 upgrid((w + upblock.x - 1) / upblock.x, (h + upblock.y - 1) / upblock.y);
            
            float scale = 1.0f / pyr_scale;
            pyrUpFlowKernel<<<upgrid, upblock, 0, resources.stream>>>(
                flow_pyr[level+1], flow_pyr[level],
                pyr_widths[level+1], pyr_heights[level+1],
                w, scale);
            
            // 检查内核执行错误
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
                resources.freeAll();
                return false;
            }
        }
        
        // 迭代优化当前层流场
        float* warped_next = nullptr;
        if (!resources.safeMalloc((void**)&warped_next, w * h * sizeof(float))) {
            resources.freeAll();
            return false;
        }
        
        for (int iter = 0; iter < iterations; iter++) {
            // 使用当前流场变形第二帧
            warpImageKernel<<<grid, block, 0, resources.stream>>>(
                next_pyr[level], warped_next, flow_pyr[level], w, h);
            
            // 检查内核执行错误
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
                resources.freeAll();
                return false;
            }
            
            // 计算变形后图像的多项式展开
            error = cudaMemset(poly_next[level], 0, w * h * 6 * sizeof(float));
            if (error != cudaSuccess) {
                printf("CUDA Memset Error: %s\n", cudaGetErrorString(error));
                resources.freeAll();
                return false;
            }
            
            polyExpansionKernel<<<grid, block, 0, resources.stream>>>(
                warped_next, poly_next[level], w, h, poly_n, d_poly_kernel);
            
            // 检查内核执行错误
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
                resources.freeAll();
                return false;
            }
            
            // 计算流场更新
            float* flow_update = nullptr;
            if (!resources.safeMalloc((void**)&flow_update, w * h * 2 * sizeof(float))) {
                resources.freeAll();
                return false;
            }
            
            error = cudaMemset(flow_update, 0, w * h * 2 * sizeof(float));
            if (error != cudaSuccess) {
                printf("CUDA Memset Error: %s\n", cudaGetErrorString(error));
                resources.freeAll();
                return false;
            }
            
            computeFlowKernel<<<grid, block, 0, resources.stream>>>(
                poly_prev[level], poly_next[level],
                flow_update, w, h, winsize, d_win_kernel);
            
            // 检查内核执行错误
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
                resources.freeAll();
                return false;
            }
            
            // 更新流场
            updateFlowKernel<<<grid, block, 0, resources.stream>>>(
                flow_pyr[level], flow_update, w, h);
            
            // 检查内核执行错误
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
                resources.freeAll();
                return false;
            }
            
            // 释放更新内存 - 由资源管理器处理
            cudaFree(flow_update);
            
            // 从资源跟踪列表中移除指针
            resources.allocated_memory.erase(
                std::remove(resources.allocated_memory.begin(), resources.allocated_memory.end(), flow_update),
                resources.allocated_memory.end());
            flow_update = nullptr;
            
            // 输出调试信息
            if (level == 0) {
                printf("  Level 0, Iteration %d completed\n", iter);
            }
        }
        
        // 释放临时内存 - 由资源管理器处理
        cudaFree(warped_next);
        
        // 从资源跟踪列表中移除指针
        resources.allocated_memory.erase(
            std::remove(resources.allocated_memory.begin(), resources.allocated_memory.end(), warped_next),
            resources.allocated_memory.end());
        warped_next = nullptr;
    }
    
    // 复制最终结果
    error = cudaMemcpy(d_flow, flow_pyr[0], width * height * 2 * sizeof(float), 
                    cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
        printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
        resources.freeAll();
        return false;
    }
    
    // 下载结果到主机
    error = cudaMemcpy(flow.ptr(), d_flow, width * height * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
        resources.freeAll();
        return false;
    }
    
    // 检查结果的最大值
    float* h_flow = new float[width * height * 2];
    memcpy(h_flow, flow.ptr(), width * height * 2 * sizeof(float));
    
    float max_flow = 0.0f;
    for (int i = 0; i < width * height * 2; i++) {
        max_flow = std::max(max_flow, std::abs(h_flow[i]));
    }
    
    printf("Final max flow magnitude: %f\n", max_flow);
    delete[] h_flow;
    
    // 下载处理后的结果
    error = cudaMemcpy(flow.ptr(), d_flow, width * height * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(error));
        resources.freeAll();
        return false;
    }
    
    // 同步设备确保所有操作完成
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("CUDA Synchronize Error: %s\n", cudaGetErrorString(error));
        resources.freeAll();
        return false;
    }
    
    // 释放所有CUDA资源
    resources.freeAll();
    
    // 打印完成信息
    printf("Optical flow computation completed successfully.\n");
    
    return true;
}
