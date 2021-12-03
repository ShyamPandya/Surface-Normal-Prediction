import torch.nn.functional as F
import torch.nn as nn

from backbone import build_backbone
from decoder import build_decoder
from aspp import build_aspp

class SurfaceNormalModel(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, freeze_bn=False):
        super(SurfaceNormalModel, self).__init__()

        if backbone == 'drn':
            output_stride = 8

        batch_norm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, output_stride, batch_norm)
        self.aspp = build_aspp(backbone, output_stride, batch_norm)
        self.decoder = build_decoder(num_classes, backbone, batch_norm)

        if freeze_bn:
            self.freeze_bn()


    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
