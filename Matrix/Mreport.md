# 矩阵乘法
## 运行
``` bash
g++ Matrix_openmp.cpp -fopenmp -o Matrix_openmp.exe
.\Matrix_openmp.exe
```
## 无任何优化-矩阵乘法只使用乘法的两重循环
``` cpp
#pragma omp parallel for collapse(2)
```
### Matrix size: 500 x 500
- Serial computation time: 0.784 seconds
- Parallel computation time: 0.194 seconds
- Speedup: 4.04124x
- Parallel efficiency: 80.8248%

### Matrix size: 1000 x 1000
- Serial computation time: 6.27 seconds
- Parallel computation time: 1.56 seconds
- Speedup: 4.01923x
- Parallel efficiency: 80.3846%

## 使用OpenMP优化-矩阵乘法使用三重循环，将三重循环中的k循环求和并行化
``` cpp
#pragma omp parallel for collapse(3)
```
### Matrix size: 1000 x 1000
- Serial computation time: 6.273 seconds
- Parallel computation time: 2.176 seconds
- Speedup: 2.88281x
- Parallel efficiency: 57.6563%

### Matrix size: 2000 x 2000
- Serial computation time: 49.606 seconds
- Parallel computation time: 8.086 seconds
- Speedup: 6.1348x
- Parallel efficiency: 122.696%

### Matrix size: 4000 x 4000
- Serial computation time: 395.905 seconds
- Parallel computation time: 8.7 seconds
- Speedup: 45.5063x
- Parallel efficiency: 910.126%

---

## **总结**
| 情况 | 建议 |
|------|------|
| **单层循环** | 直接用 `#pragma omp parallel for` |
| **嵌套循环，内层循环较小** | 使用 `collapse(2)` 提高并行效率 |
| **嵌套循环，层数更多** | 可以用 `collapse(3)` 或更高 |
| **循环不满足 `collapse` 条件** | 手动调整循环结构或改用其他并行方式 |

可以看出如果矩阵规模较小，使用并行化反而会降低效率，因为并行化需要额外的开销，实际情况需要根据数据量和可并行循环层数选择合适collapse层数。