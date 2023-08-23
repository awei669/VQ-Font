import torch.nn as nn
from functools import partial
from model.modules.blocks import ConvBlock


class ContentEncoder(nn.Module):
    """
    ContentEncoder
    """

    def __init__(self, layers, sigmoid=False):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.sigmoid = sigmoid

    def forward(self, x):
        out = self.net(x)
        if self.sigmoid:
            out = nn.Sigmoid()(out)
        return out

def content_enc_builder(C_in, C, C_out, norm='none', activ='relu', pad_type='reflect', content_sigmoid=False):
    """
    content_enc_builder
    C = 32; C_out = 256
    """
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)

    layers = [
        ConvBlk(C_in, C, 3, 1, 1, norm='in', activ='relu'),
        ConvBlk(C * 1, C * 2, 3, 2, 1),  # 64x64
        ConvBlk(C * 2, C * 4, 3, 2, 1),  # 32x32
        ConvBlk(C * 4, C * 8, 3, 2, 1),  # 16x16
        ConvBlk(C * 8, C_out, 3, 1, 1)
    ]

    return ContentEncoder(layers, content_sigmoid)
