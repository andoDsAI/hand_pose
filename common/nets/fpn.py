import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.cbam import SpatialGate
import nets.extractors as extractors 


class FPN(nn.Module):
    def __init__(
        self, 
        fpn_size: int = 2048,
        lateral_sizes: list = [1024, 512, 256],
        deep_feature_size: int = 256,
        backend: str = "resnet18",
        pretrained: bool = True
    ) -> None:
        super(FPN, self).__init__()
        self.in_planes = 64
        resnet = getattr(extractors, backend)(pretrained=pretrained)

        # Bottom-up layers
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.leakyrelu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
                                                                                                                                                                                  
        # Top layer
        self.toplayer = nn.Conv2d(
            in_channels=fpn_size, out_channels=deep_feature_size, 
            kernel_size=1, stride=1, padding=0
        )  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            in_channels=lateral_sizes[0], out_channels=deep_feature_size,
            kernel_size=1, stride=1, padding=0
        )
        self.latlayer2 = nn.Conv2d(
            in_channels=lateral_sizes[1], out_channels=deep_feature_size, 
            kernel_size=1, stride=1, padding=0
        )
        self.latlayer3 = nn.Conv2d(
            in_channels=lateral_sizes[2], out_channels=deep_feature_size, 
            kernel_size=1, stride=1, padding=0
        )
        
        # Smooth layers
        # self.smooth1 = nn.Conv2d(
        #     in_channels=deep_feature_size, out_channels=deep_feature_size,
        #     kernel_size=3, stride=1, padding=1
        # )
        self.smooth2 = nn.Conv2d(
            in_channels=deep_feature_size, out_channels=deep_feature_size, 
            kernel_size=3, stride=1, padding=1
        )
        self.smooth3 = nn.Conv2d(
            in_channels=deep_feature_size, out_channels=deep_feature_size,
            kernel_size=3, stride=1, padding=1
        )

        # Attention Module
        self.attention_module = SpatialGate()

        # Pooling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        # p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # Attention
        p2 = self.pool(p2)
        primary_feats, secondary_feats = self.attention_module(p2)

        return primary_feats, secondary_feats
