import cv2
import numpy as np
import os

# 指定要处理的文件夹路径
input_folder = '/tmp/UNet/XCAD/test/images'
output_folder = '/tmp/UNet/XCAD/test/Top_Hat'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义结构单元
kernel7 = np.ones((15, 15), np.uint8)

# 遍历文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):  # 只处理特定扩展名的文件
        # 读取图像
        image = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

        # 单尺度结构单元为7x7的顶帽变换
        tophat7 = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel7)

        # 保存处理后的图像
        output_filename = os.path.join(output_folder, filename)
        cv2.imwrite(output_filename, tophat7)
        print(f'Processed and saved: {output_filename}')

print('所有图像已处理并保存。')


