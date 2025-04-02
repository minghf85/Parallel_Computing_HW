#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

class Matrix {
private:
    std::vector<double> data;
    int size;

public:
    // 构造函数
    Matrix(int n) : size(n), data(n * n) {}

    // 获取矩阵大小
    int getSize() const { return size; }

    // 访问元素
    double& operator()(int i, int j) { return data[i * size + j]; }
    const double& operator()(int i, int j) const { return data[i * size + j]; }

    // 随机初始化矩阵
    void randomInit() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 9);
        
        for (int i = 0; i < size * size; i++) {
            data[i] = dis(gen);
        }
    }

    // 串行版本的矩阵乘法
    static Matrix multiply_serial(const Matrix& A, const Matrix& B) {
        int n = A.getSize();
        Matrix C(n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }

    // 并行版本的矩阵乘法
    static Matrix multiply_parallel(const Matrix& A, const Matrix& B) {
        int n = A.getSize();
        Matrix C(n);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A(i, k) * B(j, k);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }
    // 并行版本的矩阵乘法+三层循环并行
    static Matrix multiply_parallel_1(const Matrix& A, const Matrix& B) {
        int n = A.getSize();
        Matrix C(n);
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A(i, k) * B(j, k);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }
    // 并行版本的矩阵乘法+内存访问连续性
    static Matrix multiply_parallel_2(const Matrix& A, const Matrix& B) {
        int n = A.getSize();
        Matrix C(n);
        Matrix B_T(n);
        // 将B转置，使内存访问连续性提高
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                B_T(i, j) = B(j, i);
            }
        }
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A(i, k) * B_T(j, k);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }
    // 并行版本的矩阵乘法+内存访问连续性+SIMD:让编译器自动向量化内层循环
    static Matrix multiply_parallel_3(const Matrix& A, const Matrix& B) {
        int n = A.getSize();
        Matrix C(n);
        Matrix B_T(n);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                B_T(i, j) = B(j, i);
            }
        }
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < n; k++) {
                    sum += A(i, k) * B_T(j, k);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }
};

int main() {
    int n = 2000; // 矩阵大小
    
    // 创建并初始化矩阵
    Matrix A(n);
    Matrix B(n);
    
    A.randomInit();
    B.randomInit();
    
    // 设置OpenMP线程数
    omp_set_num_threads(5);
    
    // 串行计算
    double serial_start = omp_get_wtime();
    Matrix C_serial = Matrix::multiply_serial(A, B);
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;
    
    // 并行计算
    double parallel_start = omp_get_wtime();
    Matrix C_parallel = Matrix::multiply_parallel(A, B);
    double parallel_end = omp_get_wtime();
    double parallel_time = parallel_end - parallel_start;
    
    // 计算并输出结果
    double speedup = serial_time / parallel_time;
    
    std::cout << "Matrix size: " << n << " x " << n << std::endl;
    std::cout << "Serial computation time: " << serial_time << " seconds" << std::endl;
    std::cout << "Parallel computation time: " << parallel_time << " seconds" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    std::cout << "Parallel efficiency: " << (speedup / 5) * 100 << "%" << std::endl;
    
    return 0;
} 