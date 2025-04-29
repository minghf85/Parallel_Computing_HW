#include "CuImage.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void convertToGrayKernel(uchar *d_image, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;
        uchar r = d_image[idx];
        uchar g = d_image[idx + 1];
        uchar b = d_image[idx + 2];
        uchar gray = static_cast<uchar>(0.299f * r + 0.587f * g + 0.114f * b);
        d_image[idx] = gray;
        d_image[idx + 1] = gray;
        d_image[idx + 2] = gray;
    }
}

// Define the kernel function for horizontal Gaussian blur
__global__ void gaussianBlurHorizontalKernel(const uchar *input, uchar *output, int width, int height, int channels, 
                                           float *kernel, int kernelRadius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;
            for (int k = -kernelRadius; k <= kernelRadius; k++)
            {
                int sampleX = min(max(x + k, 0), width - 1);
                sum += input[(y * width + sampleX) * channels + c] * kernel[k + kernelRadius];
            }
            output[(y * width + x) * channels + c] = static_cast<uchar>(sum);
        }
    }
}

// Define the kernel function for vertical Gaussian blur
__global__ void gaussianBlurVerticalKernel(const uchar *input, uchar *output, int width, int height, int channels, 
                                         float *kernel, int kernelRadius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;
            for (int k = -kernelRadius; k <= kernelRadius; k++)
            {
                int sampleY = min(max(y + k, 0), height - 1);
                sum += input[(sampleY * width + x) * channels + c] * kernel[k + kernelRadius];
            }
            output[(y * width + x) * channels + c] = static_cast<uchar>(sum);
        }
    }
}

// Kernel to compute histogram in parallel
__global__ void computeHistogramKernel(const uchar* image, int width, int height, int channels, 
                                     unsigned int* histogram, int numBins) 
{
    __shared__ unsigned int localHist[256];
    
    // Initialize shared memory histogram
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockSize = blockDim.x * blockDim.y;
    
    // Each thread initializes some bins
    for (int i = tid; i < numBins; i += blockSize) {
        localHist[i] = 0;
    }
    __syncthreads();
    
    // Each thread processes pixels from its assigned region
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        // For grayscale, we just need the first channel
        uchar pixelValue = image[idx];
        atomicAdd(&localHist[pixelValue], 1);
    }
    __syncthreads();
    
    // Each thread updates some bins in the global histogram
    for (int i = tid; i < numBins; i += blockSize) {
        atomicAdd(&histogram[i], localHist[i]);
    }
}

// Kernel to find the optimal threshold
__global__ void findOtsuThresholdKernel(const unsigned int* histogram, int totalPixels, int numBins, int* threshold) 
{
    __shared__ float maxVariance;
    __shared__ int optThreshold;
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        maxVariance = 0.0f;
        optThreshold = 0;
    }
    __syncthreads();
    
    // Each thread handles one potential threshold value
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (t < numBins) {
        unsigned int sum = 0;
        unsigned int sumB = 0;
        int wB = 0;
        int wF = 0;
        
        // Sum histogram up to threshold t
        for (int i = 0; i <= t; i++) {
            wB += histogram[i];
            sumB += i * histogram[i];
        }
        
        wF = totalPixels - wB;
        sum = 0;
        for (int i = 0; i < numBins; i++) {
            sum += i * histogram[i];
        }
        
        float mB = (wB > 0) ? (float)sumB / wB : 0;
        float mF = (wF > 0) ? (float)(sum - sumB) / wF : 0;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
        
        // Update the maximum variance and threshold atomically
        if (varBetween > 0) {
            atomicMax((int*)&maxVariance, __float_as_int(varBetween));
            // Need a second synchronization to ensure maxVariance is updated before comparison
            __syncthreads();
            
            if (__float_as_int(varBetween) == __float_as_int(maxVariance)) {
                atomicExch(threshold, t);
            }
        }
    }
}

// Kernel to apply threshold
__global__ void applyThresholdKernel(uchar* image, int width, int height, int channels, int threshold) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            int idx = (y * width + x) * channels + c;
            image[idx] = (image[idx] > threshold) ? 255 : 0;
        }
    }
}

// Sobel kernel for gradient computation
__global__ void sobelKernel(const uchar* input, float* gradientX, float* gradientY, float* magnitude, uchar* direction, 
                          int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // 3x3 Sobel kernels
        const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Sobel operator
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int sampleY = min(max(y + ky, 0), height - 1);
                int sampleX = min(max(x + kx, 0), width - 1);
                
                float pixel = static_cast<float>(input[(sampleY * width + sampleX) * channels]);
                
                gx += pixel * sobel_x[ky+1][kx+1];
                gy += pixel * sobel_y[ky+1][kx+1];
            }
        }
        
        // Store gradient values
        int idx = y * width + x;
        gradientX[idx] = gx;
        gradientY[idx] = gy;
        
        // Compute gradient magnitude
        magnitude[idx] = sqrtf(gx * gx + gy * gy);
        
        // Compute gradient direction (quantized to 4 directions: 0, 45, 90, 135 degrees)
        float angle = atan2f(abs(gy), abs(gx)) * 180.0f / M_PI;
        
        // Quantize direction: 0, 45, 90, 135
        uchar dir;
        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180))
            dir = 0;      // 0 degrees (horizontal)
        else if (angle >= 22.5 && angle < 67.5)
            dir = 45;     // 45 degrees (diagonal)
        else if (angle >= 67.5 && angle < 112.5)
            dir = 90;     // 90 degrees (vertical)
        else
            dir = 135;    // 135 degrees (diagonal)
            
        direction[idx] = dir;
    }
}

// Non-maximum suppression kernel with bilinear interpolation
__global__ void nonMaxSuppressionKernel(const float* magnitude, const uchar* direction, uchar* output, 
                                      int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float mag = magnitude[idx];
        uchar dir = direction[idx];
        
        // 计算基于梯度方向的插值坐标
        float gx = 0.0f, gy = 0.0f;
        
        switch(dir) {
            case 0: // 水平边缘 (0度)
                gx = 1.0f; gy = 0.0f;
                break;
            case 45: // 对角线边缘 (45度)
                gx = 1.0f; gy = -1.0f;
                break;
            case 90: // 垂直边缘 (90度)
                gx = 0.0f; gy = -1.0f;
                break;
            case 135: // 对角线边缘 (135度)
                gx = -1.0f; gy = -1.0f;
                break;
        }
        
        // 检查梯度方向上的插值
        float mag1 = 0.0f, mag2 = 0.0f;
        
        // 正方向插值
        float xp = x + gx;
        float yp = y + gy;
        
        // 第一点的双线性插值
        if (xp >= 0 && xp < width-1 && yp >= 0 && yp < height-1) {
            int x0 = floor(xp);
            int y0 = floor(yp);
            float wx = xp - x0;
            float wy = yp - y0;
            
            // 获取四个相邻像素
            float f00 = magnitude[y0 * width + x0];
            float f01 = magnitude[y0 * width + (x0+1)];
            float f10 = magnitude[(y0+1) * width + x0];
            float f11 = magnitude[(y0+1) * width + (x0+1)];
            
            // 双线性插值公式
            mag1 = f00 * (1-wx) * (1-wy) + f01 * wx * (1-wy) + 
                   f10 * (1-wx) * wy + f11 * wx * wy;
        } else if (xp >= 0 && xp < width && yp >= 0 && yp < height) {
            // 如果在边缘，使用最近邻插值
            mag1 = magnitude[int(yp) * width + int(xp)];
        }
        
        // 负方向插值
        float xn = x - gx;
        float yn = y - gy;
        
        // 第二点的双线性插值
        if (xn >= 0 && xn < width-1 && yn >= 0 && yn < height-1) {
            int x0 = floor(xn);
            int y0 = floor(yn);
            float wx = xn - x0;
            float wy = yn - y0;
            
            // 获取四个相邻像素
            float f00 = magnitude[y0 * width + x0];
            float f01 = magnitude[y0 * width + (x0+1)];
            float f10 = magnitude[(y0+1) * width + x0];
            float f11 = magnitude[(y0+1) * width + (x0+1)];
            
            // 双线性插值公式
            mag2 = f00 * (1-wx) * (1-wy) + f01 * wx * (1-wy) + 
                   f10 * (1-wx) * wy + f11 * wx * wy;
        } else if (xn >= 0 && xn < width && yn >= 0 && yn < height) {
            // 如果在边缘，使用最近邻插值
            mag2 = magnitude[int(yn) * width + int(xn)];
        }
        
        // 如果当前像素不是局部最大值，抑制它
        output[idx] = (mag >= mag1 && mag >= mag2) ? static_cast<uchar>(min(mag, 255.0f)) : 0;
    }
}

// 改进的滞后阈值处理核函数 - 使用更细致的分类
__global__ void improvedHysteresisThresholdingKernel(const uchar* input, uchar* output, int width, int height, 
                                                   int lowThreshold, int highThreshold) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        uchar val = input[idx];
        
        if (val >= highThreshold) {
            output[idx] = 255;  // 强边缘
        } else if (val >= lowThreshold) {
            // 对于弱边缘，我们进一步将它们分级
            // 这样在边缘跟踪时可以优先处理更强的弱边缘
            int strength = ((val - lowThreshold) * 127) / (highThreshold - lowThreshold);
            output[idx] = 128 + strength; // 128-254之间的值表示弱边缘的强度
        } else {
            output[idx] = 0;    // 非边缘
        }
    }
}

// 改进的边缘跟踪核函数 - 使用更精细的边缘连接逻辑
__global__ void improvedEdgeTrackingKernel(uchar* edges, int width, int height, bool* changed) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // 只处理弱边缘 (128-254)
        if (edges[idx] >= 128 && edges[idx] < 255) {
            // 计算连接强度 - 弱边缘的强度和周围强边缘的数量和距离有关
            float connectionStrength = 0.0f;
            bool hasStrongNeighbor = false;
            
            // 检查3x3邻域
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        // 如果是强边缘，增加连接强度
                        if (edges[ny * width + nx] == 255) {
                            // 直接相邻的强边缘贡献更大
                            float weight = (abs(dx) + abs(dy) == 1) ? 2.0f : 1.0f;
                            connectionStrength += weight;
                            hasStrongNeighbor = true;
                        }
                        // 如果是较强的弱边缘，也增加一些连接强度
                        else if (edges[ny * width + nx] > 128 + 64) { // 较强的弱边缘
                            float weight = (abs(dx) + abs(dy) == 1) ? 0.5f : 0.25f;
                            connectionStrength += weight * (float)(edges[ny * width + nx] - 128) / 127.0f;
                        }
                    }
                }
            }
            
            // 检查5x5邻域中的强边缘，但权重较小
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    // 跳过已处理的3x3邻域
                    if (abs(dx) <= 1 && abs(dy) <= 1) continue;
                    
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        // 只检查强边缘
                        if (edges[ny * width + nx] == 255) {
                            // 距离越远权重越小
                            float distance = sqrtf(dx*dx + dy*dy);
                            float weight = 1.0f / distance;
                            connectionStrength += weight * 0.5f; // 乘以0.5来降低5x5部分的整体贡献
                            hasStrongNeighbor = true;
                        }
                    }
                }
            }
            
            // 基于连接强度和弱边缘自身的强度决定是否保留
            float selfStrength = (float)(edges[idx] - 128) / 127.0f; // 归一化的弱边缘强度
            
            // 决策逻辑: 强连接、或者较强的弱边缘且有一定连接
            if (connectionStrength >= 2.0f || 
                (connectionStrength >= 1.0f && selfStrength >= 0.5f) ||
                (hasStrongNeighbor && selfStrength >= 0.75f)) {
                edges[idx] = 255;  // 提升为强边缘
                *changed = true;   // 标记发生了变化
            }
        }
    }
}

// 改进的清理核函数 - 删除未连接的弱边缘，保留强边缘
__global__ void improvedCleanupKernel(uchar* edges, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        if (edges[idx] >= 128 && edges[idx] < 255) {
            edges[idx] = 0;
        }
    }
}

CuImage::CuImage() : d_image(nullptr), width(0), height(0), channels(0) {}

CuImage::~CuImage()
{
    freeDeviceMemory();
}

void CuImage::allocateDeviceMemory()
{
    if (d_image)
        cudaFree(d_image);

    size_t size = width * height * channels * sizeof(uchar);
    cudaMalloc(&d_image, size);
}

void CuImage::freeDeviceMemory()
{
    if (d_image)
    {
        cudaFree(d_image);
        d_image = nullptr;
    }
}

void CuImage::loadImage(const std::string &filename)
{
    image = cv::imread(filename);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return;
    }

    width = image.cols;
    height = image.rows;
    channels = image.channels();

    allocateDeviceMemory();
    cudaMemcpy(d_image, image.data, width * height * channels * sizeof(uchar), cudaMemcpyHostToDevice);
}

void CuImage::showImage(const std::string &windowName)
{
    cv::imshow(windowName, image);
    cv::waitKey(0);
}

void CuImage::saveImage(const std::string &filename)
{
    cudaMemcpy(image.data, d_image, width * height * channels * sizeof(uchar), cudaMemcpyDeviceToHost);
    cv::imwrite(filename, image);
}

void CuImage::convertToGray()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    convertToGrayKernel<<<gridSize, blockSize>>>(d_image, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(image.data, d_image, width * height * channels * sizeof(uchar), cudaMemcpyDeviceToHost);
}


void CuImage::GaussianBlur(int kernelSize)
{
    if (kernelSize <= 0 || kernelSize % 2 == 0)
    {
        std::cerr << "Kernel size must be a positive odd number" << std::endl;
        return;
    }

    // Calculate kernel radius from kernel size
    int kernelRadius = kernelSize / 2;
    
    // Create Gaussian kernel
    float sigma = kernelSize / 6.0f; // Rule of thumb: sigma = kernelSize/6
    float *h_kernel = new float[kernelSize];
    float sum = 0.0f;
    
    // Calculate Gaussian function values
    for (int i = 0; i < kernelSize; i++)
    {
        int x = i - kernelRadius;
        h_kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
        sum += h_kernel[i];
    }
    
    // Normalize the kernel
    for (int i = 0; i < kernelSize; i++)
    {
        h_kernel[i] /= sum;
    }
    
    // Allocate device memory for kernel
    float *d_kernel;
    cudaMalloc(&d_kernel, kernelSize * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate memory for temporary image data
    uchar *d_temp;
    cudaMalloc(&d_temp, width * height * channels * sizeof(uchar));
    
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Apply horizontal blur
    gaussianBlurHorizontalKernel<<<gridSize, blockSize>>>(d_image, d_temp, width, height, channels, d_kernel, kernelRadius);
    
    // Apply vertical blur
    gaussianBlurVerticalKernel<<<gridSize, blockSize>>>(d_temp, d_image, width, height, channels, d_kernel, kernelRadius);
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(image.data, d_image, width * height * channels * sizeof(uchar), cudaMemcpyDeviceToHost);
    
    // Free allocated memory
    cudaFree(d_temp);
    cudaFree(d_kernel);
    delete[] h_kernel;
}

void CuImage::OtsuThreshold()
{
    // First convert to grayscale if needed (3 channels)
    if (channels == 3) {
        convertToGray();
    }
    
    const int numBins = 256; // For 8-bit grayscale image
    
    // Allocate memory for histogram on device
    unsigned int* d_histogram;
    cudaMalloc(&d_histogram, numBins * sizeof(unsigned int));
    cudaMemset(d_histogram, 0, numBins * sizeof(unsigned int));
    
    // Calculate optimal block size
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Compute histogram
    computeHistogramKernel<<<gridSize, blockSize>>>(d_image, width, height, channels, d_histogram, numBins);
    
    // Allocate memory for threshold value
    int* d_threshold;
    cudaMalloc(&d_threshold, sizeof(int));
    cudaMemset(d_threshold, 0, sizeof(int));
    
    // Find optimal threshold
    int threadsPerBlock = 256;
    int blocksPerGrid = (numBins + threadsPerBlock - 1) / threadsPerBlock;
    findOtsuThresholdKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_histogram, width * height, numBins, d_threshold);
    
    // Copy threshold back to host
    int threshold;
    cudaMemcpy(&threshold, d_threshold, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Apply threshold
    applyThresholdKernel<<<gridSize, blockSize>>>(d_image, width, height, channels, threshold);
    cudaDeviceSynchronize();
    
    // Copy data back to host
    cudaMemcpy(image.data, d_image, width * height * channels * sizeof(uchar), cudaMemcpyDeviceToHost);
    
    // Free allocated memory
    cudaFree(d_histogram);
    cudaFree(d_threshold);
    
    std::cout << "Otsu threshold value: " << threshold << std::endl;
}

void CuImage::Canny(int lowThreshold, int highThreshold)
{
    // Convert to grayscale if needed
    if (channels == 3) {
        convertToGray();
    }
    
    // Gaussian blur to reduce noise
    GaussianBlur(3);
    
    // Allocate device memory for gradient computation
    float *d_gradientX, *d_gradientY, *d_magnitude;
    uchar *d_direction, *d_nms, *d_edges;
    
    size_t imageSize = width * height * sizeof(float);
    size_t byteSize = width * height * sizeof(uchar);
    
    cudaMalloc(&d_gradientX, imageSize);
    cudaMalloc(&d_gradientY, imageSize);
    cudaMalloc(&d_magnitude, imageSize);
    cudaMalloc(&d_direction, byteSize);
    cudaMalloc(&d_nms, byteSize);
    cudaMalloc(&d_edges, byteSize);
    
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Step 1: Compute gradients using Sobel operator
    sobelKernel<<<gridSize, blockSize>>>(d_image, d_gradientX, d_gradientY, 
                                      d_magnitude, d_direction, width, height, channels);
    
    // Step 2: Non-maximum suppression
    nonMaxSuppressionKernel<<<gridSize, blockSize>>>(d_magnitude, d_direction, d_nms, width, height);
    
    // Step 3: Hysteresis thresholding (initial marking)
    improvedHysteresisThresholdingKernel<<<gridSize, blockSize>>>(d_nms, d_edges, width, height, lowThreshold, highThreshold);
    
    // Step 4: Edge tracking by hysteresis (multiple passes)
    bool *d_changed;
    bool h_changed;
    cudaMalloc(&d_changed, sizeof(bool));
    
    // Perform edge tracking until no more changes
    for (int i = 0; i < 10; i++) {  // Limit iterations to avoid infinite loop
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
        
        improvedEdgeTrackingKernel<<<gridSize, blockSize>>>(d_edges, width, height, d_changed);
        
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        if (!h_changed) break;  // No changes were made, we're done
    }
    
    // Step 5: Final cleanup - remove weak edges
    improvedCleanupKernel<<<gridSize, blockSize>>>(d_edges, width, height);
    
    // Copy result back to original image (for all channels)
    if (channels == 1) {
        cudaMemcpy(d_image, d_edges, byteSize, cudaMemcpyDeviceToDevice);
    } else if (channels == 3) {
        // For 3-channel image, set all channels to the edge value
        // Define a special kernel for this if performance is critical
        uchar* h_edges = new uchar[byteSize];
        cudaMemcpy(h_edges, d_edges, byteSize, cudaMemcpyDeviceToHost);
        
        uchar* h_result = new uchar[width * height * channels];
        for (int i = 0; i < width * height; i++) {
            for (int c = 0; c < channels; c++) {
                h_result[i * channels + c] = h_edges[i];
            }
        }
        
        cudaMemcpy(d_image, h_result, width * height * channels * sizeof(uchar), cudaMemcpyHostToDevice);
        
        delete[] h_edges;
        delete[] h_result;
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(image.data, d_image, width * height * channels * sizeof(uchar), cudaMemcpyDeviceToHost);
    
    // Free allocated memory
    cudaFree(d_gradientX);
    cudaFree(d_gradientY);
    cudaFree(d_magnitude);
    cudaFree(d_direction);
    cudaFree(d_nms);
    cudaFree(d_edges);
    cudaFree(d_changed);
}