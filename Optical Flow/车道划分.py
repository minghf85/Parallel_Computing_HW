import cv2
import numpy as np

# 全局变量
points = []  # 存储多边形顶点
roi_defined = False  # 标记是否完成绘制

# 鼠标回调函数
def draw_roi(event, x, y, flags, param):
    global points, roi_defined
    
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击添加顶点
        points.append((x, y))
        print(f"添加顶点: ({x}, {y})")
        
    elif event == cv2.EVENT_RBUTTONDOWN:  # 右键闭合多边形
        roi_defined = True

# 读取图片
image_path = "data/first_frame.jpg"  # 替换为你的图片路径
image = cv2.imread(image_path)

if image is None:
    print(f"错误：无法加载图片 {image_path}")
    exit()

# 创建窗口并绑定回调
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", draw_roi)

# 交互式绘制ROI
while True:
    temp_image = image.copy()
    
    # 绘制已有点的连线
    if len(points) > 1:
        cv2.polylines(temp_image, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
    
    # 显示当前点
    for point in points:
        cv2.circle(temp_image, point, 5, (0, 0, 255), -1)
    
    cv2.imshow("Select ROI", temp_image)
    
    # 按ESC退出或右键完成绘制
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or roi_defined:  # ESC或右键
        break

# 生成ROI掩模并提取区域
if len(points) >= 3:
    # 闭合多边形
    pts = np.array(points, np.int32)
    mask = np.zeros_like(image[:, :, 0])  # 单通道掩模
    cv2.fillPoly(mask, [pts], 255)       # 填充多边形
    
    # 应用掩模提取ROI
    roi = cv2.bitwise_and(image, image, mask=mask)
    
    # 可视化：在原图上叠加半透明ROI
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0, 50))  # 半透明绿色
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
    
    # 显示结果
    cv2.imshow("Original with ROI", image)
    cv2.imshow("Extracted ROI", roi)
    
    # 保存ROI区域
    cv2.imwrite("roi_output.jpg", roi)
    print("ROI已保存为 roi_output.jpg")
    
    cv2.waitKey(0)
else:
    print("错误：至少需要3个点定义多边形")

cv2.destroyAllWindows()