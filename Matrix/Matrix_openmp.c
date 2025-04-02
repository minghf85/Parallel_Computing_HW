#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_multiply(double* A, double* B, double* C, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// 辅助函数：创建矩阵
double* create_matrix(int n) {
    return (double*)malloc(n * n * sizeof(double));
}

// 辅助函数：初始化矩阵
void init_matrix(double* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = rand() % 10;
    }
}

// 辅助函数：释放矩阵内存
void free_matrix(double* matrix) {
    free(matrix);
}

int main() {
    int n = 2000; // 矩阵大小
    
    // 创建并初始化矩阵
    double* A = create_matrix(n);
    double* B = create_matrix(n);
    double* C = create_matrix(n);
    
    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed!\n");
        return -1;
    }
    
    init_matrix(A, n);
    init_matrix(B, n);
    
    // 设置OpenMP线程数
    omp_set_num_threads(6);
    
    // 记录开始时间
    double start_time = omp_get_wtime();
    
    // 执行矩阵乘法
    matrix_multiply(A, B, C, n);
    
    // 记录结束时间
    double end_time = omp_get_wtime();
    printf("Computation time: %f seconds\n", end_time - start_time);
    
    // 释放内存
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    
    return 0;
}
