import cv2
import numpy as np
from matplotlib import pyplot as plt

def compare_images(img1_path, img2_path, threshold=30):
    """
    比较两张图片的差异
    :param img1_path: 第一张图片路径
    :param img2_path: 第二张图片路径
    :param threshold: 差异阈值（0-255）
    :return: 差异可视化结果
    """
    # 读取图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("无法读取图片，请检查路径是否正确")
    
    # 确保图片尺寸相同
    if img1.shape != img2.shape:
        # 自动调整第二张图片尺寸
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 转换为灰度图（减少计算量）
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 计算绝对差异
    diff = cv2.absdiff(gray1, gray2)
    
    # 应用阈值得到差异掩膜
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 寻找差异区域轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在原图上绘制差异矩形框
    img1_with_diff = img1.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 忽略小面积差异
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img1_with_diff, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(231), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1'), plt.axis('off')
    
    plt.subplot(232), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2'), plt.axis('off')
    
    plt.subplot(233), plt.imshow(diff, cmap='gray')
    plt.title('Absolute Difference'), plt.axis('off')
    
    plt.subplot(234), plt.imshow(thresh, cmap='gray')
    plt.title('Thresholded Difference'), plt.axis('off')
    
    plt.subplot(235), plt.imshow(cv2.cvtColor(img1_with_diff, cv2.COLOR_BGR2RGB))
    plt.title('Differences Highlighted'), plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    return {
        'difference': diff,
        'thresholded': thresh,
        'highlighted': img1_with_diff
    }

# 使用示例
if __name__ == "__main__":
    # 替换为你的图片路径
    image1 = "./img/gray.jpg"
    image2 = "./img/blur.jpg"
    
    # 比较图片（可调整阈值）
    results = compare_images(image1, image2, threshold=2)
    
    # 保存结果（可选）
    cv2.imwrite("difference.jpg", results['difference'])
    cv2.imwrite("differences_highlighted.jpg", cv2.cvtColor(results['highlighted'], cv2.COLOR_RGB2BGR))