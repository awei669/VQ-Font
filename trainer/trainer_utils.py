import torch
import torch.nn as nn

# 判断模型中使用BN否
def has_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True

    return False


def unflatten_B(t):
    """ Unflatten [B*3, ...] tensor to [B, 3, ...] tensor
    t is flattened tensor from component batch, which is [B, 3, ...] tensor
    """
    shape = t.shape
    return t.view(shape[0]//3, 3, *shape[1:])


def overwrite_weight(model, pre_weight):
    model_dict = model.state_dict()
    pre_weight = {k: v for k, v in pre_weight.items() if k in model_dict}

    model_dict.update(pre_weight)
    model.load_state_dict(model_dict)


def load_checkpoint(path, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler):
    """
    load_ckeckpoint
    """
    ckpt = torch.load(path)

    gen.load_state_dict(ckpt['generator'])
    g_optim.load_state_dict(ckpt['optimizer'])
    g_scheduler.load_state_dict(ckpt['g_scheduler'])

    if disc is not None:
        disc.load_state_dict(ckpt['discriminator'])
        d_optim.load_state_dict(ckpt['d_optimizer'])
        d_scheduler.load_state_dict(ckpt['d_scheduler'])

    st_epoch = ckpt['epoch'] + 1
    loss = ckpt['loss']

    return st_epoch, loss