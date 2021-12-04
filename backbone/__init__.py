from .drn import drn_d_54
from .mobilenet import MobileNetV2
from .resnet import ResNet101
from .xception import AlignedXception


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
