#include "CuImage.h"
#include <iostream>
//./main /home/ming/PROJECT/Parallel_Computing_HW/Image/img/test.jpg /home/ming/PROJECT/Parallel_Computing_HW/Image/img/gray.jpg gray
//./main /home/ming/PROJECT/Parallel_Computing_HW/Image/img/test.jpg /home/ming/PROJECT/Parallel_Computing_HW/Image/img/blur.jpg blur
//./main /home/ming/PROJECT/Parallel_Computing_HW/Image/img/test.jpg /home/ming/PROJECT/Parallel_Computing_HW/Image/img/otsu.jpg otsu
//./main /home/ming/PROJECT/Parallel_Computing_HW/Image/img/test.jpg /home/ming/PROJECT/Parallel_Computing_HW/Image/img/canny.jpg canny
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return -1;
    }
    // 解析命令行参数
    // argv[1] 是输入图像路径，argv[2] 是输出图像路径
    // argv[3] 是模式（可选）
    // 例如：argv[3] 可以是 "gray"、"blur"、"otsu"、"canny"
    std::string inputImage = argv[1];
    std::string outputImage = argv[2];
    std::string mode = argv[3];

    try
    {
        CuImage cuImage;

        // 加载图像
        cuImage.loadImage(inputImage);
        if(mode=="gray")
        {
            // 转换为灰度图
            cuImage.convertToGray();
        }
        else if(mode=="blur")
        {
            // 高斯模糊
            cuImage.GaussianBlur(5);
        }
        else if(mode=="otsu")
        {
            // Otsu 二值化
            cuImage.OtsuThreshold();
        }
        else if(mode=="canny")
        {
            // Canny 边缘检测，多尺度测试
            cuImage.Canny(20, 150);

        }
        else
        {
            std::cerr << "Invalid mode. Use 'gray', 'blur', 'otsu', or 'canny'." << std::endl;
            return -1;
        }
        // 保存图像
        cuImage.saveImage(outputImage);
        // 显示图像
        cuImage.showImage("Output Image");

        std::cout << "image saved to: " << outputImage << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
