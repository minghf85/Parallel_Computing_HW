# CUDA图像处理库实现报告

完成以下内容
- 灰度图
- 直方图均衡化
- 高斯滤波,中值滤波
- 边缘检测(sobel)

## 1. 并行计算基础

在实现CuImage库的过程中，我们利用了NVIDIA的CUDA平台进行并行计算。并行计算的核心思想是将大型计算任务分解为多个小任务，这些小任务可以同时在多个处理单元上执行。在CUDA编程模型中，这些处理单元被组织为线程，线程又被组织为块（Block），多个块组成一个网格（Grid）。

### 基本并行模式

在我们的实现中，每个像素的处理通常由一个独立的线程完成，这种并行策略称为数据并行。我们采用以下固定模式组织线程：

```cuda
dim3 blockSize(16, 16);  // 每个块包含16×16=256个线程
dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
              (height + blockSize.y - 1) / blockSize.y);
```

这种组织方式可以确保对于任意大小的图像，都有足够的线程来处理每一个像素，同时保持了良好的硬件利用率。

## 2. 灰度图转换

### 算法原理

灰度转换是将彩色图像（通常是RGB三通道）转换为灰度图像的过程。我们使用标准的加权公式进行转换：

Gray = 0.299×R + 0.587×G + 0.114×B

这个公式考虑了人眼对不同颜色的敏感度，绿色通道的权重最高，因为人眼对绿色最为敏感。

### 并行实现

```cuda
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
```

在这个实现中：
- 每个线程负责处理一个像素
- 线程通过其在块中的位置以及块在网格中的位置计算出它要处理的像素坐标
- 边界检查确保不会访问图像范围外的内存
- 对于输入的三通道图像，将计算出的灰度值赋给所有三个通道，保持图像格式不变

并行优势：这种实现方式使得成千上万个像素可以同时进行灰度转换，大大加速了处理速度。

## 3. 高斯滤波

### 算法原理

高斯滤波是一种常用的图像平滑技术，它使用高斯函数作为卷积核对图像进行模糊处理。高斯函数是：

G(x) = (1/√(2πσ²))·e^(-(x²)/(2σ²))

二维高斯函数是分离的，这意味着二维高斯卷积可以分解为两个一维卷积，先水平方向再垂直方向，这大大提高了计算效率。

### 并行实现

我们的实现利用了高斯核的可分离性，将二维卷积分为两步一维卷积：

1. 首先创建一维高斯核:
```cuda
// 计算高斯函数值
for (int i = 0; i < kernelSize; i++)
{
    int x = i - kernelRadius;
    h_kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
    sum += h_kernel[i];
}
// 归一化核
for (int i = 0; i < kernelSize; i++)
{
    h_kernel[i] /= sum;
}
```

2. 水平方向卷积:
```cuda
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
```

3. 垂直方向卷积:
```cuda
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
```

并行优势：
- 分离卷积将计算复杂度从O(k²)降低到O(2k)，其中k是核大小
- 每个像素的处理完全独立，非常适合GPU的并行处理
- 在处理边界时使用了镜像填充（clamp-to-edge），避免了额外的边界检查

## 4. Otsu阈值处理

### 算法原理

Otsu方法是一种自动确定二值化阈值的算法，它通过最大化类间方差来找到最佳阈值。其步骤为：
1. 计算图像的直方图
2. 对每个可能的阈值，计算在该阈值下前景和背景的类间方差
3. 选择使类间方差最大的阈值作为最优阈值

### 并行实现

我们的实现分为三个步骤：

1. 并行计算直方图:
```cuda
__global__ void computeHistogramKernel(const uchar* image, int width, int height, int channels, 
                                     unsigned int* histogram, int numBins) 
{
    __shared__ unsigned int localHist[256];
    
    // 初始化共享内存中的直方图
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockSize = blockDim.x * blockDim.y;
    
    // 每个线程初始化一些直方图bins
    for (int i = tid; i < numBins; i += blockSize) {
        localHist[i] = 0;
    }
    __syncthreads();
    
    // 每个线程处理其负责的像素区域
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        // 对于灰度图，我们只需要第一个通道
        uchar pixelValue = image[idx];
        atomicAdd(&localHist[pixelValue], 1);
    }
    __syncthreads();
    
    // 每个线程更新全局直方图中的一些bins
    for (int i = tid; i < numBins; i += blockSize) {
        atomicAdd(&histogram[i], localHist[i]);
    }
}
```

2. 并行寻找最佳阈值:
```cuda
__global__ void findOtsuThresholdKernel(const unsigned int* histogram, int totalPixels, int numBins, int* threshold) 
{
    __shared__ float maxVariance;
    __shared__ int optThreshold;
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        maxVariance = 0.0f;
        optThreshold = 0;
    }
    __syncthreads();
    
    // 每个线程处理一个潜在的阈值
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (t < numBins) {
        unsigned int sum = 0;
        unsigned int sumB = 0;
        int wB = 0;
        int wF = 0;
        
        // 计算阈值t以下的直方图总和
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
        
        // 原子操作更新最大方差和阈值
        if (varBetween > 0) {
            atomicMax((int*)&maxVariance, __float_as_int(varBetween));
            // 第二次同步确保maxVariance在比较前已更新
            __syncthreads();
            
            if (__float_as_int(varBetween) == __float_as_int(maxVariance)) {
                atomicExch(threshold, t);
            }
        }
    }
}
```

3. 并行应用阈值:
```cuda
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
```

并行优势：
- 直方图计算使用共享内存减少全局内存访问冲突
- 在寻找最佳阈值时，每个线程并行计算不同阈值下的类间方差
- 使用原子操作安全地更新共享的最大方差和最佳阈值
- 最后的二值化处理实现了全并行，每个像素独立处理

## 5. Canny边缘检测

### 算法原理

Canny边缘检测是一种经典的边缘检测算法，包含以下步骤：
1. 高斯滤波去噪
2. 计算梯度幅值和方向（使用Sobel算子）
3. 非极大值抑制
4. 双阈值处理和边缘跟踪

### 并行实现

#### 5.1 Sobel梯度计算

```cuda
__global__ void sobelKernel(const uchar* input, float* gradientX, float* gradientY, float* magnitude, uchar* direction, 
                          int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // 3x3 Sobel核
        const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        
        float gx = 0.0f, gy = 0.0f;
        
        // 应用Sobel算子
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int sampleY = min(max(y + ky, 0), height - 1);
                int sampleX = min(max(x + kx, 0), width - 1);
                
                float pixel = static_cast<float>(input[(sampleY * width + sampleX) * channels]);
                
                gx += pixel * sobel_x[ky+1][kx+1];
                gy += pixel * sobel_y[ky+1][kx+1];
            }
        }
        
        // 存储梯度值
        int idx = y * width + x;
        gradientX[idx] = gx;
        gradientY[idx] = gy;
        
        // 计算梯度幅值
        magnitude[idx] = sqrtf(gx * gx + gy * gy);
        
        // 计算梯度方向（量化为4个方向：0°, 45°, 90°, 135°）
        float angle = atan2f(abs(gy), abs(gx)) * 180.0f / M_PI;
        
        // 方向量化
        uchar dir;
        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180))
            dir = 0;      // 0度（水平）
        else if (angle >= 22.5 && angle < 67.5)
            dir = 45;     // 45度（对角线）
        else if (angle >= 67.5 && angle < 112.5)
            dir = 90;     // 90度（垂直）
        else
            dir = 135;    // 135度（对角线）
            
        direction[idx] = dir;
    }
}
```

#### 5.2 非极大值抑制（使用双线性插值）

```cuda
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
        
        // 在梯度方向上检查插值点
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
```

#### 5.3 改进的滞后阈值处理

```cuda
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
            // 对于弱边缘，进一步将它们分级
            // 这样在边缘跟踪时可以优先处理更强的弱边缘
            int strength = ((val - lowThreshold) * 127) / (highThreshold - lowThreshold);
            output[idx] = 128 + strength; // 128-254之间的值表示弱边缘的强度
        } else {
            output[idx] = 0;    // 非边缘
        }
    }
}
```

#### 5.4 改进的边缘跟踪

```cuda
__global__ void improvedEdgeTrackingKernel(uchar* edges, int width, int height, bool* changed) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // 只处理弱边缘 (128-254)
        if (edges[idx] >= 128 && edges[idx] < 255) {
            // 计算连接强度 - 基于周围强边缘的数量和距离
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
                            connectionStrength += weight * 0.5f; // 降低远处边缘的整体贡献
                            hasStrongNeighbor = true;
                        }
                    }
                }
            }
            
            // 基于连接强度和弱边缘自身的强度决定是否保留
            float selfStrength = (float)(edges[idx] - 128) / 127.0f; // 归一化的弱边缘强度
            
            // 决策逻辑
            if (connectionStrength >= 2.0f || 
                (connectionStrength >= 1.0f && selfStrength >= 0.5f) ||
                (hasStrongNeighbor && selfStrength >= 0.75f)) {
                edges[idx] = 255;  // 提升为强边缘
                *changed = true;   // 标记发生了变化
            }
        }
    }
}
```

#### 5.5 最终清理

```cuda
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
```

并行优势：
- Sobel计算在各个像素上并行进行，每个线程负责一个像素的梯度计算
- 非极大值抑制使用了更精确的双线性插值方法，提高了边缘定位精度
- 边缘跟踪算法使用了迭代方法，每次迭代都并行处理所有像素，直到没有变化为止
- 将弱边缘分级（128-254）而不是简单二值化，这使得边缘跟踪能够优先处理更可能是真实边缘的像素
- 边缘连接使用了更大的5x5邻域和更复杂的连接强度计算，提高了连接质量

## 6. 性能优化技术

在实现这些图像处理算法时，我们应用了多种CUDA优化技术：

### 6.1 内存访问优化

- **合并访问**：尽可能使相邻的线程访问相邻的内存位置，提高内存访问效率
- **共享内存**：在直方图计算中使用共享内存来减少全局内存访问
- **合理的线程块大小**：使用16×16的线程块大小，是在性能和灵活性之间的良好平衡

### 6.2 计算优化

- **内核分离**：将复杂算法分解为多个步骤，每个步骤使用独立的核函数
- **分离卷积**：在高斯滤波中，利用卷积核的可分离性，将二维卷积分为两步一维卷积
- **原子操作**：在需要多个线程更新同一内存位置的场景（如直方图计算、寻找最佳阈值）中使用原子操作
- **边界处理**：通过镜像填充（clamp-to-edge）处理图像边界，减少分支判断

### 6.3 工作流优化

- **异步操作**：使用cudaDeviceSynchronize()确保GPU操作完成
- **内存管理**：明确的内存分配和释放，避免内存泄漏
- **流水线处理**：在Canny算法中，将处理分为多个阶段，每个阶段的输出作为下一阶段的输入

## 7. 结论与未来工作

通过CUDA并行计算，我们实现了高效的图像处理库，包括灰度转换、高斯滤波、Otsu阈值处理和Canny边缘检测等功能。这些算法充分利用了GPU的并行计算能力，相比CPU实现有显著的性能提升。

未来可能的改进方向包括：
1. 实现更多图像处理功能，如霍夫变换、SIFT特征提取等
2. 进一步优化内存访问模式，减少全局内存访问
3. 利用CUDA动态并行性处理复杂的递归算法
4. 使用CUDA流实现并行数据传输和计算
5. 开发更通用的接口，使库更易于使用

![alt text](img/canny.jpg)