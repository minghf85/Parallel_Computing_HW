#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

class CuImage
{
private:
    cv::Mat image; // 存储图像数据
    uchar *d_image; // GPU 上的图像数据指针
    int width, height, channels; // 图像的宽、高、通道数

    void allocateDeviceMemory(); // 分配 GPU 内存
    void freeDeviceMemory(); // 释放 GPU 内存

public:
    CuImage();
    ~CuImage();

    void loadImage(const std::string &filename);
    void showImage(const std::string &windowName);
    void saveImage(const std::string &filename);
    void convertToGray();
    void GaussianBlur(int kernelSize);
    void OtsuThreshold();
    void Canny(int lowThreshold = 50, int highThreshold = 150);//使用3x3核
};
