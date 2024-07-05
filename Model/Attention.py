import torch
import torch.nn as nn
import torch.nn.functional as F

# 通道注意力机制
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        # 共享的MLP结构，用于通道压缩和扩展
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),  # 降维
            nn.ReLU(),  # ReLU激活
            nn.Conv2d(channel // ratio, channel, 1, bias=False)  # 升维
        )
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))  # 平均池化后的特征通过MLP
        maxout = self.shared_MLP(self.max_pool(x))  # 最大池化后的特征通过MLP
        return self.sigmoid(avgout + maxout)  # 将两个池化结果相加后通过Sigmoid

# 空间注意力机制
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        # 卷积层，用于生成空间注意力图
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)  # 在通道维度上计算平均值
        maxout, _ = torch.max(x, dim=1, keepdim=True)  # 在通道维度上计算最大值
        out = torch.cat([avgout, maxout], dim=1)  # 将平均和最大池化结果拼接
        out = self.sigmoid(self.conv2d(out))  # 通过卷积和Sigmoid生成空间注意力图
        return out

# 双模块缝合
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)  # 初始化通道注意力模块
        self.spatial_attention = SpatialAttentionModule()  # 初始化空间注意力模块

    def forward(self, x):
        out = self.channel_attention(x) * x  # 应用通道注意力
        out = self.spatial_attention(out) * out  # 应用空间注意力
        return out  # 返回最终的注意力调整后的特征图
