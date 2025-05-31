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
    // ���캯��
    Matrix(int n) : size(n), data(n * n) {}

    // ��ȡ�����С
    int getSize() const { return size; }

    // ����Ԫ��
    double& operator()(int i, int j) { return data[i * size + j]; }
    const double& operator()(int i, int j) const { return data[i * size + j]; }

    // �����ʼ������
    void randomInit() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 9);
        
        for (int i = 0; i < size * size; i++) {
            data[i] = dis(gen);
        }
    }

    // ���а汾�ľ���˷�
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

    // ���а汾�ľ���˷�
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

    // ���й�Լ�ڲ�ѭ���汾�ľ���˷�
    static Matrix multiply_parallel_1(const Matrix& A, const Matrix& B) {
        int n = A.getSize();
        Matrix C(n);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                #pragma omp parallel for reduction(+:sum) // �� k ���й�Լ
                for (int k = 0; k < n; k++) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }

    // ���а汾�ľ���˷�+�ڴ����������
    static Matrix multiply_parallel_2(const Matrix& A, const Matrix& B) {
        int n = A.getSize();
        Matrix C(n);
        Matrix B_T(n);

        // ת�þ���B������ڴ����Ч��
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                B_T(i, j) = B(j, i);
            }
        }

        // ʹ��ת�ú�ľ�����м���
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

    // ���а汾�ľ���˷�+�ڴ����������+SIMD
    static Matrix multiply_parallel_3(const Matrix& A, const Matrix& B) {
        int n = A.getSize();
        Matrix C(n);
        Matrix B_T(n);

        // ת�þ���B
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                B_T(i, j) = B(j, i);
            }
        }

        // ʹ��SIMD��OpenMP��ϵ��Ż��汾
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

    // ��ӱȽϺ���
    bool isEqual(const Matrix& other, double tolerance = 1e-10) const {
        if (size != other.size) return false;
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double diff = std::abs((*this)(i,j) - other(i,j));
                if (diff > tolerance) {
                    std::cout << "��ƥ����λ�� (" << i << "," << j << "): " 
                              << (*this)(i,j) << " != " << other(i,j) 
                              << " (����: " << diff << ")" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    // ��Ӵ�ӡ���־���ĺ��������ڵ��ԣ�
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
    int n = 2000; // �����С
    int num_threads = 6;
    // ��������ʼ������
    Matrix A(n);
    Matrix B(n);
    
    A.randomInit();
    B.randomInit();
    
    // ����OpenMP�߳���
    omp_set_num_threads(num_threads);
    
    std::cout << "��ʼ����˷����� (�����С: " << n << "x" << n << ", �߳���: " << num_threads << ")" << std::endl;
    
    // ���м���
    double serial_start = omp_get_wtime();
    Matrix C_serial = Matrix::multiply_serial(A, B);
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;
    
    // �������а汾
    double parallel_start = omp_get_wtime();
    Matrix C_parallel = Matrix::multiply_parallel(A, B);
    double parallel_end = omp_get_wtime();
    double parallel_time = parallel_end - parallel_start;

    // ���й�Լ�ڲ�ѭ���汾�ľ���˷�
    double parallel1_start = omp_get_wtime();
    Matrix C_parallel1 = Matrix::multiply_parallel_1(A, B);
    double parallel1_end = omp_get_wtime();
    double parallel1_time = parallel1_end - parallel1_start;
    
    // �ڴ���������԰汾
    double parallel2_start = omp_get_wtime();
    Matrix C_parallel2 = Matrix::multiply_parallel_2(A, B);
    double parallel2_end = omp_get_wtime();
    double parallel2_time = parallel2_end - parallel2_start;

    // �ڴ�����Ż�+SIMD�汾
    double parallel3_start = omp_get_wtime();
    Matrix C_parallel3 = Matrix::multiply_parallel_3(A, B);
    double parallel3_end = omp_get_wtime();
    double parallel3_time = parallel3_end - parallel3_start;
    
    // ������ܽ��
    std::cout << "\n���ܽ��:" << std::endl;
    std::cout << "�����С: " << n << " x " << n << std::endl;
    std::cout << "\n���м���ʱ��: " << serial_time << " ��" << std::endl;
    
    std::cout << "\n�������а汾:" << std::endl;
    std::cout << "����ʱ��: " << parallel_time << " ��" << std::endl;
    std::cout << "���ٱ�: " << serial_time / parallel_time << "x" << std::endl;
    std::cout << "����Ч��: " << (serial_time / parallel_time / num_threads) * 100 << "%" << std::endl;

    std::cout << "\n���й�Լ�ڲ�ѭ���汾:" << std::endl;
    std::cout << "����ʱ��: " << parallel1_time << " ��" << std::endl;
    std::cout << "���ٱ�: " << serial_time / parallel1_time << "x" << std::endl;
    std::cout << "����Ч��: " << (serial_time / parallel1_time / num_threads) * 100 << "%" << std::endl;

    std::cout << "\n�ڴ���������԰汾:" << std::endl;
    std::cout << "����ʱ��: " << parallel2_time << " ��" << std::endl;
    std::cout << "���ٱ�: " << serial_time / parallel2_time << "x" << std::endl;
    std::cout << "����Ч��: " << (serial_time / parallel2_time / num_threads) * 100 << "%" << std::endl;
    
    std::cout << "\n�ڴ�����Ż�+SIMD�汾:" << std::endl;
    std::cout << "����ʱ��: " << parallel3_time << " ��" << std::endl;
    std::cout << "���ٱ�: " << serial_time / parallel3_time << "x" << std::endl;
    std::cout << "����Ч��: " << (serial_time / parallel3_time / num_threads) * 100 << "%" << std::endl;

    // ��֤�����ȷ��
    std::cout << "\n��֤��������ȷ��:" << std::endl;
    
    std::cout << "�������а汾: ";
    if (C_serial.isEqual(C_parallel)) {
        std::cout << "��ȷ" << std::endl;
    } else {
        std::cout << "�����ƥ�䣡" << std::endl;
        std::cout << "���н��ʾ�� (���Ͻ�4x4):" << std::endl;
        C_serial.printSubMatrix(0, 0, 4);
        std::cout << "���н��ʾ�� (���Ͻ�4x4):" << std::endl;
        C_parallel.printSubMatrix(0, 0, 4);
    }
    
    std::cout << "���й�Լ�ڲ�ѭ���汾: ";
    if (C_serial.isEqual(C_parallel1)) {
        std::cout << "��ȷ" << std::endl;
    } else {
        std::cout << "�����ƥ�䣡" << std::endl;
        std::cout << "���н��ʾ�� (���Ͻ�4x4):" << std::endl;
        C_serial.printSubMatrix(0, 0, 4);
        std::cout << "���н��ʾ�� (���Ͻ�4x4):" << std::endl;
        C_parallel1.printSubMatrix(0, 0, 4);
    }
    
    std::cout << "�ڴ���������԰汾: ";
    if (C_serial.isEqual(C_parallel2)) {
        std::cout << "��ȷ" << std::endl;
    } else {
        std::cout << "�����ƥ�䣡" << std::endl;
        std::cout << "���н��ʾ�� (���Ͻ�4x4):" << std::endl;
        C_serial.printSubMatrix(0, 0, 4);
        std::cout << "���н��ʾ�� (���Ͻ�4x4):" << std::endl;
        C_parallel2.printSubMatrix(0, 0, 4);
    }
    
    std::cout << "�ڴ�����Ż�+SIMD�汾: ";
    if (C_serial.isEqual(C_parallel3)) {
        std::cout << "��ȷ" << std::endl;
    } else {
        std::cout << "�����ƥ�䣡" << std::endl;
        std::cout << "���н��ʾ�� (���Ͻ�4x4):" << std::endl;
        C_serial.printSubMatrix(0, 0, 4);
        std::cout << "���н��ʾ�� (���Ͻ�4x4):" << std::endl;
        C_parallel3.printSubMatrix(0, 0, 4);
    }
    
    return 0;
} 