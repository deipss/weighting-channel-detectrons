import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Backbone, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained).features
        self.feature_layers = [13, 18]  # 选择第13层和第18层的特征图

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class FSSD(nn.Module):
    def __init__(self, num_classes, backbone):
        super(FSSD, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.feat_channels = [1280, 1280]  # 特征图的通道数
        self.out_channels = 256  # 融合后的特征图通道数

        # 特征融合模块
        self.fusion_modules = nn.ModuleList([
            FeatureFusionModule(in_channels, self.out_channels)
            for in_channels in self.feat_channels
        ])

        # 检测头
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(self.out_channels, 4 * 3, kernel_size=3, padding=1)  # 3个默认框
            for _ in range(len(self.feat_channels))
        ])
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(self.out_channels, num_classes * 3, kernel_size=3, padding=1)  # 3个默认框
            for _ in range(len(self.feat_channels))
        ])

    def forward(self, x):
        # 特征提取
        features = self.backbone(x)

        # 特征融合
        fused_features = [fusion_module(feature) for fusion_module, feature in zip(self.fusion_modules, features)]

        # 检测头
        loc_preds = []
        conf_preds = []
        for i, feature in enumerate(fused_features):
            loc_pred = self.loc_layers[i](feature)
            conf_pred = self.conf_layers[i](feature)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            loc_preds.append(loc_pred)
            conf_preds.append(conf_pred)

        loc_preds = torch.cat(loc_preds, 1)
        conf_preds = torch.cat(conf_preds, 1)

        return loc_preds, conf_preds