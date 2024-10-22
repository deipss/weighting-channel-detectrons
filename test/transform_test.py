import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 定义一系列变换
    transform = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小为 256x256
        transforms.CenterCrop(224),  # 从中心裁剪出 224x224 的图像
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转图像
        transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对图像进行标准化
    ])

    # 加载图像和边界框
    image = Image.open(
        '/Users/deipss/workspace/ai/weighting-channel-detectrons/data/Brackish/dataset/img/2019-02-20_19-01-02to2019-02-20_19-01-13_1-0006.png')
    bbox = np.array([100, 100, 50, 50])

    # 应用自定义变换
    transformed_image = transform(image)
    pass