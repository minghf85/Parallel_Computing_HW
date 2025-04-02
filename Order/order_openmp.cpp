#include <iostream>
#include <vector>
#include <omp.h>
#include <ctime>
#include <random>
#include <algorithm>
#include <windows.h>
void setUTF8Console() {
    SetConsoleOutputCP(CP_UTF8);
}


// 串行快速排序
void serialQuickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);
        
        int pi = i + 1;
        serialQuickSort(arr, low, pi - 1);
        serialQuickSort(arr, pi + 1, high);
    }
}

// 优化的并行快速排序
void parallelQuickSort(std::vector<int>& arr, int low, int high) {
    const int THRESHOLD = 10000;  // 设置串行阈值
    
    if (low < high) {
        if (high - low < THRESHOLD) {
            // 小规模数据使用串行排序
            serialQuickSort(arr, low, high);
            return;
        }

        // 使用三数取中法选择pivot
        int mid = low + (high - low) / 2;
        if (arr[mid] < arr[low]) std::swap(arr[mid], arr[low]);
        if (arr[high] < arr[low]) std::swap(arr[high], arr[low]);
        if (arr[mid] < arr[high]) std::swap(arr[mid], arr[high]);
        
        int pivot = arr[high];
        int i = low - 1;
        
        // 使用并行for来进行分区操作
        #pragma omp parallel
        {
            int local_i = -1;
            std::vector<int> local_swaps;
            
            #pragma omp for nowait
            for (int j = low; j < high; j++) {
                if (arr[j] <= pivot) {
                    local_i++;
                    local_swaps.push_back(j);
                }
            }
            
            #pragma omp critical
            {
                for (int j : local_swaps) {
                    i++;
                    std::swap(arr[i], arr[j]);
                }
            }
        }
        
        std::swap(arr[i + 1], arr[high]);
        int pi = i + 1;

        // 只在较大的子数组上使用并行
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                if (pi - low > THRESHOLD)
                    parallelQuickSort(arr, low, pi - 1);
                else
                    serialQuickSort(arr, low, pi - 1);
            }
            
            #pragma omp section
            {
                if (high - pi > THRESHOLD)
                    parallelQuickSort(arr, pi + 1, high);
                else
                    serialQuickSort(arr, pi + 1, high);
            }
        }
    }
}

// 串行归并排序
void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    for (i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
}

void serialMergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        serialMergeSort(arr, left, mid);
        serialMergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// 优化的并行归并排序
void parallelMergeSort(std::vector<int>& arr, int left, int right) {
    const int THRESHOLD = 10000;  // 设置串行阈值
    
    if (left < right) {
        if (right - left < THRESHOLD) {
            // 小规模数据使用串行排序
            serialMergeSort(arr, left, right);
            return;
        }

        int mid = left + (right - left) / 2;
        
        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, left, mid);
            
            #pragma omp section
            parallelMergeSort(arr, mid + 1, right);
        }

        // 优化的并行合并操作
        std::vector<int> temp(right - left + 1);
        #pragma omp parallel
        {
            #pragma omp single
            {
                int i = left, j = mid + 1, k = 0;
                while (i <= mid && j <= right) {
                    if (arr[i] <= arr[j]) {
                        temp[k++] = arr[i++];
                    } else {
                        temp[k++] = arr[j++];
                    }
                }
                
                while (i <= mid) temp[k++] = arr[i++];
                while (j <= right) temp[k++] = arr[j++];
            }
            
            // 并行复制回原数组
            #pragma omp for
            for (int i = 0; i < temp.size(); i++) {
                arr[left + i] = temp[i];
            }
        }
    }
}

// 串行基数排序
void serialRadixSort(std::vector<int>& arr) {
    int max = *std::max_element(arr.begin(), arr.end());
    
    for (int exp = 1; max/exp > 0; exp *= 10) {
        std::vector<int> output(arr.size());
        std::vector<int> count(10, 0);
        
        for (int i = 0; i < arr.size(); i++)
            count[(arr[i]/exp)%10]++;
        
        for (int i = 1; i < 10; i++)
            count[i] += count[i-1];
        
        for (int i = arr.size()-1; i >= 0; i--) {
            output[count[(arr[i]/exp)%10]-1] = arr[i];
            count[(arr[i]/exp)%10]--;
        }
        
        arr = output;
    }
}

// 并行基数排序
void parallelRadixSort(std::vector<int>& arr) {
    int max = *std::max_element(arr.begin(), arr.end());
    
    for (int exp = 1; max/exp > 0; exp *= 10) {
        std::vector<int> output(arr.size());
        std::vector<int> count(10, 0);
        
        // 并行计算每个数字出现的次数
        #pragma omp parallel
        {
            std::vector<int> local_count(10, 0);
            
            #pragma omp for nowait
            for (int i = 0; i < arr.size(); i++)
                local_count[(arr[i]/exp)%10]++;
            
            #pragma omp critical
            for (int i = 0; i < 10; i++)
                count[i] += local_count[i];
        }
        
        // 计算前缀和
        for (int i = 1; i < 10; i++)
            count[i] += count[i-1];
        
        // 并行构建输出数组
        std::vector<int> pos(arr.size());
        #pragma omp parallel for
        for (int i = 0; i < arr.size(); i++) {
            int digit = (arr[i]/exp)%10;
            #pragma omp atomic capture
            pos[i] = --count[digit];
        }
        
        #pragma omp parallel for
        for (int i = 0; i < arr.size(); i++)
            output[pos[i]] = arr[i];
        
        arr = output;
    }
}

// 串行冒泡排序
void serialBubbleSort(std::vector<int>& arr) {
    for (int i = 0; i < arr.size()-1; i++) {
        for (int j = 0; j < arr.size()-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                std::swap(arr[j], arr[j+1]);
            }
        }
    }
}

// 并行冒泡排序（奇偶交换排序）
void parallelBubbleSort(std::vector<int>& arr) {
    bool sorted = false;
    while (!sorted) {
        sorted = true;
        
        // 奇数阶段
        #pragma omp parallel for shared(sorted)
        for (int i = 1; i < arr.size()-1; i += 2) {
            if (arr[i] > arr[i+1]) {
                std::swap(arr[i], arr[i+1]);
                sorted = false;
            }
        }
        
        // 偶数阶段
        #pragma omp parallel for shared(sorted)
        for (int i = 0; i < arr.size()-1; i += 2) {
            if (arr[i] > arr[i+1]) {
                std::swap(arr[i], arr[i+1]);
                sorted = false;
            }
        }
    }
}

// 性能测试函数
void performanceTest(int size) {
    std::vector<int> arr(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, size);
    
    // 生成随机数组
    for (int i = 0; i < size; i++) {
        arr[i] = dis(gen);
    }
    
    std::vector<int> arr1 = arr;
    std::vector<int> arr2 = arr;
    std::vector<int> arr3 = arr;
    std::vector<int> arr4 = arr;
    std::vector<int> arr5 = arr;
    std::vector<int> arr6 = arr;
    std::vector<int> arr7 = arr;
    std::vector<int> arr8 = arr;
    
    double start, end;
    
    // 测试串行快速排序
    start = omp_get_wtime();
    serialQuickSort(arr1, 0, size - 1);
    end = omp_get_wtime();
    double serialQuickTime = end - start;
    
    // 测试并行快速排序
    start = omp_get_wtime();
    parallelQuickSort(arr2, 0, size - 1);
    end = omp_get_wtime();
    double parallelQuickTime = end - start;
    
    // 测试串行归并排序
    start = omp_get_wtime();
    serialMergeSort(arr3, 0, size - 1);
    end = omp_get_wtime();
    double serialMergeTime = end - start;
    
    // 测试并行归并排序
    start = omp_get_wtime();
    parallelMergeSort(arr4, 0, size - 1);
    end = omp_get_wtime();
    double parallelMergeTime = end - start;
    
    // 测试串行基数排序
    start = omp_get_wtime();
    serialRadixSort(arr5);
    end = omp_get_wtime();
    double serialRadixTime = end - start;
    
    // 测试并行基数排序
    start = omp_get_wtime();
    parallelRadixSort(arr6);
    end = omp_get_wtime();
    double parallelRadixTime = end - start;
    
    // 对于小规模数据才测试冒泡排序
    double serialBubbleTime = 0, parallelBubbleTime = 0;
    if (size <= 100000) {  // 只对小规模数据进行冒泡排序测试
        // 测试串行冒泡排序
        start = omp_get_wtime();
        serialBubbleSort(arr7);
        end = omp_get_wtime();
        serialBubbleTime = end - start;
        
        // 测试并行冒泡排序
        start = omp_get_wtime();
        parallelBubbleSort(arr8);
        end = omp_get_wtime();
        parallelBubbleTime = end - start;
    }
    
    // 计算加速比和并行效率
    double quickSpeedup = serialQuickTime / parallelQuickTime;
    double mergeSpeedup = serialMergeTime / parallelMergeTime;
    double radixSpeedup = serialRadixTime / parallelRadixTime;
    
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    
    double quickEfficiency = quickSpeedup / num_threads;
    double mergeEfficiency = mergeSpeedup / num_threads;
    double radixEfficiency = radixSpeedup / num_threads;
    
    // 输出结果
    std::cout << "数组大小: " << size << std::endl;
    std::cout << "线程数: " << num_threads << std::endl << std::endl;
    
    std::cout << "快速排序性能分析:" << std::endl;
    std::cout << "串行时间: " << serialQuickTime << "秒" << std::endl;
    std::cout << "并行时间: " << parallelQuickTime << "秒" << std::endl;
    std::cout << "加速比: " << quickSpeedup << std::endl;
    std::cout << "并行效率: " << quickEfficiency << std::endl << std::endl;
    
    std::cout << "归并排序性能分析:" << std::endl;
    std::cout << "串行时间: " << serialMergeTime << "秒" << std::endl;
    std::cout << "并行时间: " << parallelMergeTime << "秒" << std::endl;
    std::cout << "加速比: " << mergeSpeedup << std::endl;
    std::cout << "并行效率: " << mergeEfficiency << std::endl;
    
    std::cout << "\n基数排序性能分析:" << std::endl;
    std::cout << "串行时间: " << serialRadixTime << "秒" << std::endl;
    std::cout << "并行时间: " << parallelRadixTime << "秒" << std::endl;
    std::cout << "加速比: " << radixSpeedup << std::endl;
    std::cout << "并行效率: " << radixEfficiency << std::endl;
    
    if (size <= 100000) {
        std::cout << "\n冒泡排序性能分析:" << std::endl;
        std::cout << "串行时间: " << serialBubbleTime << "秒" << std::endl;
        std::cout << "并行时间: " << parallelBubbleTime << "秒" << std::endl;
        std::cout << "加速比: " << (serialBubbleTime / parallelBubbleTime) << std::endl;
        std::cout << "并行效率: " << (radixSpeedup / num_threads) << std::endl;
    }
}

int main() {
    setUTF8Console();
    
    // 设置OpenMP线程数
    omp_set_num_threads(6);
    
    // 禁用动态线程调整
    omp_set_dynamic(0);
    
    // 启用嵌套并行
    omp_set_nested(1);
    
    // 测试不同规模的数组
    std::cout << "开始性能测试..." << std::endl << std::endl;
    performanceTest(10000);
    std::cout << "\n------------------------\n" << std::endl;
    performanceTest(100000);
    std::cout << "\n------------------------\n" << std::endl;
    performanceTest(1000000);
    
    return 0;
} 