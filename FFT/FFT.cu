#include <stdio.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//nvcc FFT.cu -o fft
#define PI 3.14159265358979323846
#define N 65536  
#define THREADS_PER_BLOCK 256  // 每个线程块的线程数
#define TEST_SIZE 16  // 用于验证的2的幂次数据点
#define ERROR_THRESHOLD 1e-3  // 允许0.1%的相对误差

// 复数结构体及其运算
struct Complex {
    float x, y;  // 实部和虚部

    // 构造函数
    __host__ __device__ Complex() : x(0), y(0) {}
    __host__ __device__ Complex(float x, float y) : x(x), y(y) {}

    // 基本运算符重载
    __host__ __device__ Complex operator+(const Complex& other) const {
        return Complex(x + other.x, y + other.y);
    }
    __host__ __device__ Complex operator-(const Complex& other) const {
        return Complex(x - other.x, y - other.y);
    }
    __host__ __device__ Complex operator*(const Complex& other) const {
        return Complex(x*other.x - y*other.y, x*other.y + y*other.x);
    }
    __host__ __device__ Complex operator/(const Complex& other) const {
        float denominator = other.x * other.x + other.y * other.y;
        return Complex(
            (x * other.x + y * other.y) / denominator,
            (y * other.x - x * other.y) / denominator
        );
    }

    // 标量运算
    __host__ __device__ Complex operator+(float scalar) const {
        return Complex(x + scalar, y);
    }
    friend __host__ __device__ Complex operator*(float scalar, const Complex& c) {
        return Complex(scalar * c.x, scalar * c.y);
    }

    // 复数函数
    __host__ __device__ Complex conj() const { return Complex(x, -y); }
    __host__ __device__ float abs() const { return sqrtf(x * x + y * y); }

    // 常量
    static __host__ __device__ Complex i() { return Complex(0, 1); }
    static __host__ __device__ Complex zero() { return Complex(0, 0); }

    // 输出
    friend std::ostream& operator<<(std::ostream& os, const Complex& c) {
        os << "(" << c.x << " + " << c.y << "i)";
        return os;
    }
};

// 位反转排列（预处理）
__global__ void bitReverseKernel(Complex* data, int n) {
    // 计算当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 计算位反转后的索引
    int bits = (int)log2f(n);
    int rev = 0;
    int idx_copy = idx;
    
    // 使用位运算计算反转索引
    for (int i = 0; i < bits; i++) {
        rev = (rev << 1) | (idx_copy & 1);
        idx_copy >>= 1;
    }

    if (rev > idx) {
        Complex temp = data[idx];
        data[idx] = data[rev];
        data[rev] = temp;
    }
}

__global__ void parallelFFTKernel(Complex* data, int n) {
    extern __shared__ Complex sdata[];  // 共享内存，用于缓存数据
    int tid = threadIdx.x;  // 线程块内的线程ID
    int bid = blockIdx.x;   // 线程块ID
    for (int step = 1; step < n; step <<= 1) {
        // 计算旋转因子的基本角度
        float angle = -PI / step;
        Complex w(cosf(angle), sinf(angle));
        
        // 并行处理蝶形运算
        // 每个线程处理多个蝶形运算对
        for (int i = bid * blockDim.x + tid; i < n/2; i += gridDim.x * blockDim.x) {
            // 计算蝶形运算的索引
            int even_idx = 2 * i - (i & (step - 1));// 计算组内的偏移
            int odd_idx = even_idx + step;
            
            if (odd_idx < n && even_idx < n) {
                // 计算当前蝶形运算的旋转因子
                Complex w_k(1.0f, 0.0f);
                for (int k = 0; k < (i & (step - 1)); k++) {
                    w_k = w_k * w;
                }
                
                // 执行蝶形运算
                Complex even = data[even_idx];
                Complex odd = data[odd_idx] * w_k;
                
                data[even_idx] = even + odd;
                data[odd_idx] = even - odd;
            }
        }
        __syncthreads();  // 确保所有线程完成当前阶段
    }
}

// 串行FFT实现（用于对比）
__global__ void serialFFTKernel(Complex* data, int n) {
    // 位反转排序
    for (int i = 0; i < n; i++) {
        int bits = (int)log2f(n);
        int rev = 0;
        int i_copy = i;
        
        for (int j = 0; j < bits; j++) {
            rev = (rev << 1) | (i_copy & 1);
            i_copy >>= 1;
        }
        
        if (rev > i) {
            Complex temp = data[i];
            data[i] = data[rev];
            data[rev] = temp;
        }
    }

    // FFT计算
    for (int step = 1; step < n; step <<= 1) {
        float angle = -PI / step;
        Complex w(cosf(angle), sinf(angle));
        
        for (int i = 0; i < n; i += step * 2) {
            Complex w_k(1.0f, 0.0f);
            for (int j = 0; j < step; j++) {
                Complex even = data[i + j];
                Complex odd = data[i + j + step] * w_k;
                
                data[i + j] = even + odd;
                data[i + j + step] = even - odd;
                
                w_k = w_k * w;
            }
        }
    }
}

void verifyFFTResults() {
    printf("\n=== FFT Validation (Using %d data points) ===\n", TEST_SIZE);
    
    // 分配主机内存
    Complex* h_input = new Complex[TEST_SIZE];
    Complex* h_parallel_output = new Complex[TEST_SIZE];
    Complex* h_serial_output = new Complex[TEST_SIZE];
    
    printf("\nInput data:\n");
    for (int i = 0; i < TEST_SIZE; i++) {
        h_input[i] = Complex((float)(i + 1), 0.0f);
        printf("x[%d] = (%.6f + %.6fi)\n", i, h_input[i].x, h_input[i].y);
    }
    
    // 分配设备内存
    Complex *d_parallel_data, *d_serial_data;
    cudaMalloc(&d_parallel_data, TEST_SIZE * sizeof(Complex));
    cudaMalloc(&d_serial_data, TEST_SIZE * sizeof(Complex));
    
    // 将数据复制到设备
    cudaMemcpy(d_parallel_data, h_input, TEST_SIZE * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_serial_data, h_input, TEST_SIZE * sizeof(Complex), cudaMemcpyHostToDevice);
    
    printf("\nExecuting parallel FFT...\n");
    bitReverseKernel<<<(TEST_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_parallel_data, TEST_SIZE);
    parallelFFTKernel<<<TEST_SIZE/(2*THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK*sizeof(Complex)>>>(d_parallel_data, TEST_SIZE);
    cudaDeviceSynchronize();
    
    printf("Executing serial FFT...\n");
    serialFFTKernel<<<1, 1>>>(d_serial_data, TEST_SIZE);
    cudaDeviceSynchronize();
    
    // 将结果复制回主机
    cudaMemcpy(h_parallel_output, d_parallel_data, TEST_SIZE * sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_serial_output, d_serial_data, TEST_SIZE * sizeof(Complex), cudaMemcpyDeviceToHost);
    
    printf("\nResults comparison:\n");
    printf("%-6s %-30s %-30s %-15s\n", "Index", "Parallel FFT", "Serial FFT", "Relative Error");
    
    float max_error = 0.0f;
    float max_relative_error = 0.0f;
    
    for (int i = 0; i < TEST_SIZE; i++) {
        // 计算绝对误差
        float error = sqrtf(powf(h_parallel_output[i].x - h_serial_output[i].x, 2) + 
                          powf(h_parallel_output[i].y - h_serial_output[i].y, 2));
        
        // 计算相对误差
        float serial_magnitude = sqrtf(h_serial_output[i].x * h_serial_output[i].x + 
                                     h_serial_output[i].y * h_serial_output[i].y);
        float relative_error = (serial_magnitude > 1e-10) ? error / serial_magnitude : error;
        
        max_error = fmaxf(max_error, error);
        max_relative_error = fmaxf(max_relative_error, relative_error);
        
        printf("%-6d (%.6f + %.6fi)    (%.6f + %.6fi)    %.6f%%\n", 
            i, h_parallel_output[i].x, h_parallel_output[i].y, 
            h_serial_output[i].x, h_serial_output[i].y,
            relative_error * 100);
    }
    
    printf("\nMaximum absolute error: %.6f\n", max_error);
    printf("Maximum relative error: %.6f%%\n", max_relative_error * 100);
    printf("Error threshold: %.6f%%\n", ERROR_THRESHOLD * 100);
    printf("Validation result: %s\n", max_relative_error < ERROR_THRESHOLD ? "PASSED" : "FAILED");
    
    // 清理内存
    delete[] h_input;
    delete[] h_parallel_output;
    delete[] h_serial_output;
    cudaFree(d_parallel_data);
    cudaFree(d_serial_data);
}

int main() {
    // 首先进行小规模验证
    verifyFFTResults();
    
    printf("\n=== Large-scale Performance Test ===\n");
    
    // 分配主机内存并初始化数据
    Complex *h_data = new Complex[N];
    Complex *d_data;
    cudaEvent_t start, stop;
    float parallelTime, serialTime;

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_data[i] = Complex((float)rand()/RAND_MAX, (float)rand()/RAND_MAX);
    }
    
    // 分配设备内存
    cudaMalloc(&d_data, N * sizeof(Complex));
    
    // 创建计时器
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 执行并行FFT并计时
    cudaMemcpy(d_data, h_data, N * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    bitReverseKernel<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data, N);
    parallelFFTKernel<<<N/(2*THREADS_PER_BLOCK), THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK*sizeof(Complex)>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&parallelTime, start, stop);
    printf("Parallel FFT Time: %.3f ms\n", parallelTime);

    // 执行串行FFT并计时
    cudaMemcpy(d_data, h_data, N * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    serialFFTKernel<<<1, 1>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&serialTime, start, stop);
    printf("Serial FFT Time: %.3f ms\n", serialTime);

    // 计算加速比
    float speedup = serialTime / parallelTime;
    int num_threads = N/(2*THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
    
    printf("Speedup Ratio: %.2f\n", speedup);
    printf("Number of threads used: %d\n", num_threads);

    // 清理资源
    cudaFree(d_data);
    delete[] h_data;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}