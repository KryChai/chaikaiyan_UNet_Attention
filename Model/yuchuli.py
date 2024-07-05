import cv2

# 定义结构元素的大小
kernel_size = (3, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

# 读取图像
input_image = cv2.imread("/tmp/U_Net/XCAD/train/images/00018_33.png", cv2.IMREAD_GRAYSCALE)

# 应用顶帽变换
tophat_img = cv2.morphologyEx(input_image, cv2.MORPH_TOPHAT, kernel)

# 显示原始图像和顶帽变换后的图像
cv2.imshow("原始图像", input_image)
cv2.imshow("顶帽变换后的图像", tophat_img)

# 等待用户按键
cv2.waitKey(0)  # 等待任意键按下，按键后关闭所有窗口

# 清理并退出
cv2.destroyAllWindows()
