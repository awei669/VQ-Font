import torch
import torch.nn as nn
from functools import partial
from .modules import ResBlock, ConvBlock, w_norm_dispatch, activ_dispatch


class ProjectionDiscriminator(nn.Module):
    """ Multi-task discriminator """
    def __init__(self, C, n_fonts, n_chars, w_norm='spectral', activ='none'):
        super().__init__()

        self.activ = activ_dispatch(activ)()
        w_norm = w_norm_dispatch(w_norm)
        self.font_emb = w_norm(nn.Embedding(n_fonts, C))
        self.char_emb = w_norm(nn.Embedding(n_chars, C))

    def forward(self, x, font_indice, char_indice):
        x = self.activ(x)
        font_emb = self.font_emb(font_indice)
        char_emb = self.char_emb(char_indice)

        font_out = torch.einsum('bchw,bc->bhw', x, font_emb).unsqueeze(1)
        char_out = torch.einsum('bchw,bc->bhw', x, char_emb).unsqueeze(1)

        return [font_out, char_out]

class CustomDiscriminator(nn.Module):
    """
    spectral norm + ResBlock + Multi-task Discriminator (No patchGAN)
    """
    def __init__(self, feats, gap, projD):
        super().__init__()
        self.feats = feats
        self.gap = gap
        self.projD = projD

    def forward(self, x, font_indice, char_indice):
        for layer in self.feats:
            x = layer(x)

        x = self.gap(x)  # final features
        #ret = [x, x]
        ret = self.projD(x, font_indice, char_indice)
        ret = tuple(map(lambda i: i.cuda(), ret))
        return ret


def disc_builder(C, n_fonts, n_chars, activ='relu', gap_activ='relu', w_norm='spectral',
                 res_scale_var=False):
    """ disc_builder """
    ConvBlk = partial(ConvBlock, w_norm=w_norm, activ=activ)
    ResBlk = partial(ResBlock, w_norm=w_norm, activ=activ, scale_var=res_scale_var)

    feats = [
        ConvBlk(1, C, stride=2, activ='none'),  # 64x64 (stirde==2)
        ResBlk(C * 1, C * 2, downsample=True),  # 32x32
        ResBlk(C * 2, C * 4, downsample=True),  # 16x16
        ResBlk(C * 4, C * 8, downsample=True),  # 8x8
        ResBlk(C * 8, C * 16, downsample=False),  # 8x8
        ResBlk(C * 16, C * 16, downsample=False),  # 8x8
    ]

    gap_activ = activ_dispatch(gap_activ)
    gaps = [
        gap_activ(),
        nn.AdaptiveAvgPool2d(1)
    ]

    projD_C_in = feats[-1].C_out
    feats = nn.ModuleList(feats)
    gap = nn.Sequential(*gaps)
    projD = ProjectionDiscriminator(projD_C_in, n_fonts, n_chars, w_norm=w_norm)

    disc = CustomDiscriminator(feats, gap, projD)

    return disc
