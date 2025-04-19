#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
// g++ -fopenmp .\Matrix_openmp.cpp -o .\Matrix_openmp.exe
// .\Matrix_openmp.exe
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
                C(i, j) = 0.0;
                for (int k = 0; k < n; k++) {
                    C(i, j) += A(i, k) * B(k, j);
                }
            }
        }
        return C;
    }

    // 并行规约内层循环版本的矩阵乘法
    static Matrix multiply_parallel_1(const Matrix& A, const Matrix& B) {
        int n = A.getSize();
        Matrix C(n);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                #pragma omp parallel for reduction(+:sum) // 对 k 并行归约
                for (int k = 0; k < n; k++) {
                    sum += A(i, k) * B(k, j);
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

        // 转置矩阵B以提高内存访问效率
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                B_T(i, j) = B(j, i);
            }
        }

        // 使用转置后的矩阵进行计算
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

    // 并行版本的矩阵乘法+内存访问连续性+SIMD
    static Matrix multiply_parallel_3(const Matrix& A, const Matrix& B) {
        int n = A.getSize();
        Matrix C(n);
        Matrix B_T(n);

        // 转置矩阵B
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                B_T(i, j) = B(j, i);
            }
        }

        // 使用SIMD和OpenMP结合的优化版本
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

    // 添加比较函数
    bool isEqual(const Matrix& other, double tolerance = 1e-10) const {
        if (size != other.size) return false;
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double diff = std::abs((*this)(i,j) - other(i,j));
                if (diff > tolerance) {
                    std::cout << "不匹配在位置 (" << i << "," << j << "): " 
                              << (*this)(i,j) << " != " << other(i,j) 
                              << " (差异: " << diff << ")" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    // 添加打印部分矩阵的函数（用于调试）
    void printSubMatrix(int start_i, int start_j, int size_to_print) const {
        int end_i = std::min(start_i + size_to_print, size);
        int end_j = std::min(start_j + size_to_print, size);
        
        for (int i = start_i; i < end_i; i++) {
            for (int j = start_j; j < end_j; j++) {
                std::cout << (*this)(i,j) << "\t";
            }
            std::cout << std::endl;
        }
    }
};

int main() {
    int n = 2000; // 矩阵大小
    int num_threads = 6;
    // 创建并初始化矩阵
    Matrix A(n);
    Matrix B(n);
    
    A.randomInit();
    B.randomInit();
    
    // 设置OpenMP线程数
    omp_set_num_threads(num_threads);
    
    std::cout << "开始矩阵乘法计算 (矩阵大小: " << n << "x" << n << ", 线程数: " << num_threads << ")" << std::endl;
    
    // 串行计算
    double serial_start = omp_get_wtime();
    Matrix C_serial = Matrix::multiply_serial(A, B);
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;
    
    // 基础并行版本
    double parallel_start = omp_get_wtime();
    Matrix C_parallel = Matrix::multiply_parallel(A, B);
    double parallel_end = omp_get_wtime();
    double parallel_time = parallel_end - parallel_start;

    // 并行规约内层循环版本的矩阵乘法
    double parallel1_start = omp_get_wtime();
    Matrix C_parallel1 = Matrix::multiply_parallel_1(A, B);
    double parallel1_end = omp_get_wtime();
    double parallel1_time = parallel1_end - parallel1_start;
    
    // 内存访问连续性版本
    double parallel2_start = omp_get_wtime();
    Matrix C_parallel2 = Matrix::multiply_parallel_2(A, B);
    double parallel2_end = omp_get_wtime();
    double parallel2_time = parallel2_end - parallel2_start;

    // 内存访问优化+SIMD版本
    double parallel3_start = omp_get_wtime();
    Matrix C_parallel3 = Matrix::multiply_parallel_3(A, B);
    double parallel3_end = omp_get_wtime();
    double parallel3_time = parallel3_end - parallel3_start;
    
    // 输出性能结果
    std::cout << "\n性能结果:" << std::endl;
    std::cout << "矩阵大小: " << n << " x " << n << std::endl;
    std::cout << "\n串行计算时间: " << serial_time << " 秒" << std::endl;
    
    std::cout << "\n基础并行版本:" << std::endl;
    std::cout << "计算时间: " << parallel_time << " 秒" << std::endl;
    std::cout << "加速比: " << serial_time / parallel_time << "x" << std::endl;
    std::cout << "并行效率: " << (serial_time / parallel_time / num_threads) * 100 << "%" << std::endl;

    std::cout << "\n并行规约内层循环版本:" << std::endl;
    std::cout << "计算时间: " << parallel1_time << " 秒" << std::endl;
    std::cout << "加速比: " << serial_time / parallel1_time << "x" << std::endl;
    std::cout << "并行效率: " << (serial_time / parallel1_time / num_threads) * 100 << "%" << std::endl;

    std::cout << "\n内存访问连续性版本:" << std::endl;
    std::cout << "计算时间: " << parallel2_time << " 秒" << std::endl;
    std::cout << "加速比: " << serial_time / parallel2_time << "x" << std::endl;
    std::cout << "并行效率: " << (serial_time / parallel2_time / num_threads) * 100 << "%" << std::endl;
    
    std::cout << "\n内存访问优化+SIMD版本:" << std::endl;
    std::cout << "计算时间: " << parallel3_time << " 秒" << std::endl;
    std::cout << "加速比: " << serial_time / parallel3_time << "x" << std::endl;
    std::cout << "并行效率: " << (serial_time / parallel3_time / num_threads) * 100 << "%" << std::endl;

    // 验证结果正确性
    std::cout << "\n验证计算结果正确性:" << std::endl;
    
    std::cout << "基础并行版本: ";
    if (C_serial.isEqual(C_parallel)) {
        std::cout << "正确" << std::endl;
    } else {
        std::cout << "结果不匹配！" << std::endl;
        std::cout << "串行结果示例 (左上角4x4):" << std::endl;
        C_serial.printSubMatrix(0, 0, 4);
        std::cout << "并行结果示例 (左上角4x4):" << std::endl;
        C_parallel.printSubMatrix(0, 0, 4);
    }
    
    std::cout << "并行规约内层循环版本: ";
    if (C_serial.isEqual(C_parallel1)) {
        std::cout << "正确" << std::endl;
    } else {
        std::cout << "结果不匹配！" << std::endl;
        std::cout << "串行结果示例 (左上角4x4):" << std::endl;
        C_serial.printSubMatrix(0, 0, 4);
        std::cout << "并行结果示例 (左上角4x4):" << std::endl;
        C_parallel1.printSubMatrix(0, 0, 4);
    }
    
    std::cout << "内存访问连续性版本: ";
    if (C_serial.isEqual(C_parallel2)) {
        std::cout << "正确" << std::endl;
    } else {
        std::cout << "结果不匹配！" << std::endl;
        std::cout << "串行结果示例 (左上角4x4):" << std::endl;
        C_serial.printSubMatrix(0, 0, 4);
        std::cout << "并行结果示例 (左上角4x4):" << std::endl;
        C_parallel2.printSubMatrix(0, 0, 4);
    }
    
    std::cout << "内存访问优化+SIMD版本: ";
    if (C_serial.isEqual(C_parallel3)) {
        std::cout << "正确" << std::endl;
    } else {
        std::cout << "结果不匹配！" << std::endl;
        std::cout << "串行结果示例 (左上角4x4):" << std::endl;
        C_serial.printSubMatrix(0, 0, 4);
        std::cout << "并行结果示例 (左上角4x4):" << std::endl;
        C_parallel3.printSubMatrix(0, 0, 4);
    }
    
    return 0;
} 