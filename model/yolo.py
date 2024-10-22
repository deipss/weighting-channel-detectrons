import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.layer1 = self._make_layer(3, 32, 1)
        self.layer2 = self._make_layer(32, 64, 2)
        self.layer3 = self._make_layer(64, 128, 2)
        self.layer4 = self._make_layer(128, 256, 8)
        self.layer5 = self._make_layer(256, 512, 8)
        self.layer6 = self._make_layer(512, 1024, 4)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = [ConvBlock(in_channels, out_channels, 3, 2, 1)]
        for _ in range(num_blocks):
            layers.append(ConvBlock(out_channels, out_channels // 2, 1, 1, 0))
            layers.append(ConvBlock(out_channels // 2, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, inputs):
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        return tuple(outs)

class PAN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(PAN, self).__init__()
        self.fpn = FPN(in_channels_list, out_channels)
        self.pan_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, inputs):
        fpn_outs = self.fpn(inputs)
        pan_outs = []
        for i in range(len(fpn_outs)):
            if i < len(fpn_outs) - 1:
                pan_outs.append(self.pan_convs[i](fpn_outs[i] + F.interpolate(fpn_outs[i + 1], size=fpn_outs[i].shape[2:], mode='nearest')))
            else:
                pan_outs.append(self.pan_convs[i](fpn_outs[i]))
        return tuple(pan_outs)
class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super(YOLOHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        return self.conv(x)
class YOLOV8(nn.Module):
    def __init__(self, num_classes, num_anchors=3):
        super(YOLOV8, self).__init__()
        self.backbone = Darknet53()
        self.neck = PAN([256, 512, 1024], 256)
        self.head = YOLOHead(256, num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        outputs = [self.head(feature) for feature in features]
        return outputs