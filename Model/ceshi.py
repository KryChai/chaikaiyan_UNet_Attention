# # import os
# #
# # # 指定要搜索的目录路径
# # directory_path = '/tmp/U_Net/XCAD/train/images/'
# #
# # # 定义文件模式
# # file_pattern = '_res'
# #
# # # 删除所有包含指定模式的文件
# # for filename in os.listdir(directory_path):
# #     if file_pattern in filename:
# #         file_path = os.path.join(directory_path, filename)
# #         try:
# #             os.remove(file_path)
# #             print(f'删除文件: {file_path}')
# #         except Exception as e:
# #             print(f'无法删除文件: {file_path}, 原因: {e}')
#
# import cv2
# import numpy as np
#
# # 读取灰度图像
# img = cv2.imread('/tmp/UNet/XCAD/test/res/12241_33_res_Atten.png', cv2.IMREAD_GRAYSCALE)
#
# # 创建结构元素
# kernel = np.ones((3, 3), dtype=np.uint8)
#
# # 进行开运算
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#
# save_res_path= "/tmp/UNet/XCAD/test/res/111.png"
#
# cv2.imwrite(save_res_path, opening)
# # 显示结果
# # cv2.imshow('Opening', opening)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# import cv2
# import numpy as np
#
# # 读取灰度图像
# img = cv2.imread('test.jpg', 0)
#
# # 转换为二值图像
# threshold = 128
# ret, img_binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
#
# # 进行开运算
# kernel = np.ones((3, 3), np.uint8)
# img_opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
#
# save_res_path= "/tmp/UNet/XCAD/test/res/.png"
# save_res_path= "/tmp/UNet/XCAD/test/res/222.png"
# cv2.imwrite(save_res_path, img_opening)
# cv2.imwrite(save_res_path, opening)
# # 显示结果
# cv2.imshow('原图像', img)
# cv2.imshow('二值图像', img_binary)
# cv2.imshow('开运算图像', img_opening)
#
# # 等待按键退出
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# import glob
#
# import torch
# import numpy as np
# import cv2
# from unet_model import UNet
# from sklearn.metrics import accuracy_score, recall_score, jaccard_score
#
# if __name__ == "__main__":
#     # 选择设备，有cuda用cuda，没有就用cpu
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # 加载网络，图片单通道，分类为1。
#     net = UNet(n_channels=1, n_classes=1)
#     # 将网络拷贝到deivce中
#     net.to(device=device)
#     # 加载模型参数
#     net.load_state_dict(torch.load('best_model.pth', map_location=device))
#     # 测试模式
#     net.eval()
#     # 读取所有图片路径
#     tests_path = glob.glob('/tmp/UNet/XCAD/test/images/*.png')
#     # 准备真实标签
#     # 假设真实标签保存在 /tmp/UNet/XCAD/test/labels/ 目录下，文件名与测试图像相同
#     labels_path = [path.replace('/tmp/UNet/XCAD/test/images/', '/tmp/UNet/XCAD/test/labels/')  for path in tests_path]
#     # 遍历所有图片
#     for test_path, label_path in zip(tests_path, labels_path):
#         # 保存结果地址
#         path=test_path.split('/')[-1].split('.')[0]
#         save_res_path = '/tmp/UNet/XCAD/test/res/'+ path + '_res_Atten_200epoch.png'
#         print(save_res_path)
#
#         # 读取图片
#         img = cv2.imread(test_path)
#         # 转为灰度图
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         # 转为batch为1，通道为1，大小为512*512的数组
#         img = img.reshape(1, 1, img.shape[0], img.shape[1])
#         # 转为tensor
#         img_tensor = torch.from_numpy(img)
#         # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
#         img_tensor = img_tensor.to(device=device, dtype=torch.float32)
#         # 预测
#         pred = net(img_tensor)
#         # 提取结果
#         pred = np.array(pred.data.cpu()[0])[0]
#         # 处理结果
#         pred[pred >= 0.5] = 255
#         pred[pred < 0.5] = 0
#         # 保存图片
#         cv2.imwrite(save_res_path, pred)
#         # 评估性能
#         # 读取真实标签
#         label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#         # 计算指标
#         accuracy = accuracy_score(label.flatten(), pred.flatten())
#         recall = recall_score(label.flatten(), pred.flatten())
#         iou = jaccard_score(label.flatten(), pred.flatten())
#         print(f'Accuracy: {accuracy:.2f}, Recall: {recall:.2f}, IoU: {iou:.2f}')

# 注意：上述代码假设真实标签与测试图像具有相同的尺寸和格式。
# 您需要根据实际情况调整真实标签的读取方式。

import glob
import numpy as np
import torch
import os
import cv2
from sklearn.metrics import accuracy_score, recall_score, jaccard_score

from unet_model import UNet

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model_Attention_200epoch.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('E:\\U_Net\\ceshi\\Simple-UNet-RetinaSeg-main\\Model\\tophat7.jpg')
    labels_path = glob.glob('E:\\U_Net\ceshi\Simple-UNet-RetinaSeg-main\XCAD\\test\masks\\12241_33.png')
    accuracy=[]
    recall=[]
    iou=[]
    # 遍历所有图片
    for i in range(len(tests_path)):
        # 保存结果地址
        path=tests_path[i].split('/')[-1].split('.')[0]
        save_res_path = 'E:\\U_Net\\ceshi\\Simple-UNet-RetinaSeg-main\\Model\\Top_Hat.png'
        print(save_res_path)

        # 读取图片
        img = cv2.imread(tests_path[i])
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)

        # 评估性能
        # 读取真实标签
        label = cv2.imread(labels_path[i], cv2.IMREAD_GRAYSCALE)
        # 将标签转换为二进制表示，0为背景，255为前景
        label[label == 255] = 1
        # 将预测结果也转换为二进制表示
        pred_binary = np.where(pred >= 0.5, 1, 0)
        # print(label.max())
        # 计算指标
        accuracy.append(accuracy_score(label.flatten(), pred_binary.flatten()))
        recall.append(recall_score(label.flatten(), pred_binary.flatten(), average='binary'))
        iou.append(jaccard_score(label.flatten(), pred_binary.flatten(), average='binary'))
        break

    print(f'Accuracy: {np.mean(accuracy):.2f}, Recall: {np.mean(recall):.2f}, IoU: {np.mean(iou):.2f}')