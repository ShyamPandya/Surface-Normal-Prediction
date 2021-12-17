from .drn import drn_d_54
from .resnet import ResNet101


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return ResNet101(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn_d_54(BatchNorm)
    else:
        raise NotImplementedError
