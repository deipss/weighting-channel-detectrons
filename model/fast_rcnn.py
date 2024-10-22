import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.ops import roi_pool
class FastRCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FastRCNN, self).__init__()

        # 加载预训练的 VGG-16 模型
        self.backbone = vgg16(pretrained=pretrained).features

        # 冻结部分层
        for param in self.backbone[:10].parameters():
            param.requires_grad = False

        # RoI Pooling 层
        self.roi_pool = roi_pool

        # 全连接层
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, num_classes)
        self.bbox_regressor = nn.Linear(4096, 4 * num_classes)

    def forward(self, images, rois):
        # 特征提取
        features = self.backbone(images)

        # RoI Pooling
        pooled_features = self.roi_pool(features, rois, output_size=(7, 7), spatial_scale=1.0 / 16)

        # 全连接层
        x = pooled_features.view(pooled_features.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 分类和回归
        class_scores = self.classifier(x)
        bbox_offsets = self.bbox_regressor(x)

        return class_scores, bbox_offsets