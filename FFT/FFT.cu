#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI 3.14159265358979323846
#define N 1048576  // 2^20 点（根据显存调整）
#define THREADS_PER_BLOCK 256

// 复数结构体
struct Complex {
    float x, y;
    __host__ __device__ Complex() : x(0), y(0) {}
    __host__ __device__ Complex(float x, float y) : x(x), y(y) {}
    __host__ __device__ Complex operator+(const Complex& other) const {
        return Complex(x + other.x, y + other.y);
    }
    __host__ __device__ Complex operator-(const Complex& other) const {
        return Complex(x - other.x, y - other.y);
    }
    __host__ __device__ Complex operator*(const Complex& other) const {
        return Complex(x*other.x - y*other.y, x*other.y + y*other.x);
    }
};

// 位反转排列（预处理）
__global__ void bitReverseKernel(Complex* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int rev = 0;
    for (int i = 0; i < (int)log2f(N); i++) {
        rev |= ((idx >> i) & 1) << ((int)log2f(N) - 1 - i);
    }

    if (rev > idx) {
        Complex temp = data[idx];
        data[idx] = data[rev];
        data[rev] = temp;
    }
}

// 并行FFT核函数
__global__ void parallelFFTKernel(Complex* data) {
    extern __shared__ Complex sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    for (int s = 1; s <= log2f(N); s++) {
        int m = 1 << s;
        int half_m = m >> 1;
        float angle = -2 * PI / m;

        for (int k = bid * THREADS_PER_BLOCK + tid; k < N/2; k += gridDim.x * THREADS_PER_BLOCK) {
            int index = (k % (N/m)) * m + (k / (N/m));
            int i = index % N;
            int j = (index + half_m) % N;
            
            Complex twiddle = Complex(cosf(angle * (k % half_m)), sinf(angle * (k % half_m)));
            Complex even = data[i];
            Complex odd = data[j] * twiddle;

            sdata[tid] = even + odd;
            sdata[tid + THREADS_PER_BLOCK] = even - odd;
            __syncthreads();

            data[i] = sdata[tid];
            data[j] = sdata[tid + THREADS_PER_BLOCK];
        }
        __syncthreads();
    }
}

// 串行FFT核函数（单线程）
__global__ void serialFFTKernel(Complex* data) {
    // 位反转
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int j = 0; j < (int)log2f(N); j++) {
            rev |= ((i >> j) & 1) << ((int)log2f(N) - 1 - j);
        }
        if (rev > i) {
            Complex temp = data[i];
            data[i] = data[rev];
            data[rev] = temp;
        }
    }

    // 蝶形运算
    for (int s = 1; s <= log2f(N); s++) {
        int m = 1 << s;
        int half_m = m >> 1;
        for (int k = 0; k < N; k += m) {
            for (int j = 0; j < half_m; j++) {
                float angle = -2 * PI * j / m;
                Complex twiddle = Complex(cosf(angle), sinf(angle));
                Complex even = data[k + j];
                Complex odd = data[k + j + half_m] * twiddle;
                data[k + j] = even + odd;
                data[k + j + half_m] = even - odd;
            }
        }
    }
}

int main() {
    Complex *h_data = new Complex[N];
    Complex *d_data;
    cudaEvent_t start, stop;
    float elapsedTime;

    // 初始化数据（示例用随机数）
    for (int i = 0; i < N; i++) {
        h_data[i] = Complex((float)rand()/RAND_MAX, (float)rand()/RAND_MAX);
    }

    // 分配设备内存
    cudaMalloc(&d_data, N * sizeof(Complex));
    cudaMemcpy(d_data, h_data, N * sizeof(Complex), cudaMemcpyHostToDevice);

    // 创建事件计时器
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 执行并行FFT
    cudaEventRecord(start);
    bitReverseKernel<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data);
    parallelFFTKernel<<<N/(2*THREADS_PER_BLOCK), THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK*sizeof(Complex)>>>(d_data);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Parallel FFT Time: %.3f ms\n", elapsedTime);

    // 执行串行FFT
    cudaEventRecord(start);
    serialFFTKernel<<<1, 1>>>(d_data);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Serial FFT Time: %.3f ms\n", elapsedTime);

    // 计算加速比和效率
    float speedup = elapsedTime / 0.001f; // 防止除零
    if (speedup < 1) speedup = 1.0f;
    float efficiency = speedup / (N/(2*THREADS_PER_BLOCK) * THREADS_PER_BLOCK);
    printf("Speedup Ratio: %.2f\n", speedup);
    printf("Efficiency: %.2f%%\n", efficiency * 100);

    // 清理资源
    cudaFree(d_data);
    delete[] h_data;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}