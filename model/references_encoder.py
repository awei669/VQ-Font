import torch.nn as nn
from functools import partial
from .modules import ConvBlock, ResBlock


class ReferenceEncoder(nn.Module):
    def __init__(self, layers, out_shape, sigmoid=False):
        super(ReferenceEncoder, self).__init__()

        self.layers = nn.Sequential(*layers)
        self.out_shape = out_shape
        self.sigmoid = sigmoid

    def forward(self, x):
        ret_feats = {}

        for layer in self.layers:
            x = layer(x)
        ret_feats["last"] = x
        if self.sigmoid:
            ret_feats = {k: nn.Sigmoid()(v) for k, v in ret_feats.items()}
        return ret_feats


def comp_enc_builder(C_in, C, C_out, norm='none', activ='relu', pad_type='reflect', skip_scale_var=False, sigmoid=True):
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, scale_var=skip_scale_var)

    layers = [
        ConvBlk(C_in, C, 3, 1, 1, norm='in', activ='relu'),  # 128x128x32
        ConvBlk(C * 1, C * 2, 3, 1, 1, downsample=True),  # 64x64x64
        ConvBlk(C * 2, C * 4, 3, 1, 1, downsample=True),  # 32x32x128
        ResBlk(C * 4, C * 4, 3, 1),
        ResBlk(C * 4, C * 4, 3, 1),
        ResBlk(C * 4, C * 8, 3, 1, downsample=True),  # 16x16x256
        ResBlk(C * 8, C_out)
    ]

    final_shape = (C_out, 16, 16)

    return ReferenceEncoder(layers, final_shape, sigmoid)
