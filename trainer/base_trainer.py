import copy
import random
import torch.nn.functional as F
import utils
from pathlib import Path
from .trainer_utils import *
try:
    from apex import amp
except ImportError:
    print('failed to import apex')


class BaseTrainer:
    def __init__(self, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                 logger, evaluator, cv_loaders, cfg):
        self.gen = gen  # 生成器
        self.gen_ema = copy.deepcopy(self.gen)  # 深度复制这个对象
        self.g_optim = g_optim  # 生成器的优化器
        self.g_scheduler = g_scheduler
        # self.is_bn_gen = has_bn(self.gen) #标志是否使用nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
        self.disc = disc  # 辨别器
        self.d_optim = d_optim  # 辨别器的优化器
        self.d_scheduler = d_scheduler
        self.cfg = cfg  # 配置文件

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.batch_size = cfg.batch_size
        self.num_component_object = cfg.num_embeddings
        self.num_channels = cfg.num_channels
        self.num_postive_samples = cfg.num_positive_samples

        [self.gen, self.gen_ema, self.disc], [self.g_optim, self.d_optim] = self.set_model(
            [self.gen, self.gen_ema, self.disc],
            [self.g_optim, self.d_optim],
        )

        self.logger = logger
        self.evaluator = evaluator
        self.cv_loaders = cv_loaders

        self.step = 1

        self.g_losses = {}
        self.d_losses = {}

        self.projection_style = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128)
        ).cuda()

    def set_model(self, models, opts):
        # if torch.cuda.device_count()>1:
        #     models = nn.DataParallel(models)

        return models, opts

    def clear_losses(self):
        """ Integrate & clear loss json_dict """
        # g losses
        loss_dic = {k: v.item() for k, v in self.g_losses.items()}
        loss_dic['g_total'] = sum(loss_dic.values())
        # d losses
        loss_dic.update({k: v.item() for k, v in self.d_losses.items()})

        self.g_losses = {}
        self.d_losses = {}

        return loss_dic

    def accum_g(self, decay=0.999):
        par1 = dict(self.gen_ema.named_parameters())
        par2 = dict(self.gen.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))

    def train(self):
        return

    def get_codebook_detach(self, component_embeddings):
        component_objects = torch.zeros(self.num_component_object, self.num_channels).cuda()
        component_objects = component_objects + component_embeddings  # [N,C]
        component_objects = component_objects.unsqueeze(0)
        component_objects = component_objects.repeat(self.batch_size, 1, 1)  # [B N C]

        return component_objects.detach()



    def add_pixel_loss(self, out, target, self_infer):
        """
        add_pixel_loss
        """
        # target_variance = torch.var(target)
        # loss1 = F.mse_loss(out, target, reduction="mean") / target_variance * self.cfg['pixel_w']
        # loss2 = F.mse_loss(self_infer, target, reduction="mean") / target_variance * self.cfg['pixel_w']

        loss1 = F.l1_loss(out, target, reduction="mean") * self.cfg['pixel_w']
        loss2 = F.l1_loss(self_infer, target, reduction="mean") * self.cfg['pixel_w']
        self.g_losses['pixel'] = loss1 + loss2
        return loss1 + loss2

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        """
        compute_contrastive_loss
        """
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        # loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))

        return loss

    def take_contrastive_feature(self, input):
        # out = self.enc_style(input)
        out = self.projection_style(input)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def style_contrastive_loss(self, style_components_1, style_components_2, batch_size):
        _, N, _ = style_components_1.shape
        style_contrastive_loss = 0
        if self.cfg['contrastive_w'] == 0:
            return style_contrastive_loss

        style_up = self.take_contrastive_feature(style_components_1)
        style_down = self.take_contrastive_feature(style_components_2)

        # print(style_components_1.shape, style_components_2.shape)

        for s in range(batch_size):
            # 对于每一种风格而言, 正样本是同风格下不同reference对应位置的码本条目
            if s == 0:
                negative_style_up = style_up[1:].transpose(0, 1)
                negative_style_down = style_down[1:].transpose(0, 1)

            if s == batch_size - 1:
                negative_style_up = style_up[:batch_size - 1].transpose(0, 1)
                negative_style_down = style_down[:batch_size - 1].transpose(0, 1)

            else:
                negative_style_up = torch.cat([style_up[0:s], style_up[s + 1:]], 0).transpose(0, 1)
                negative_style_down = torch.cat([style_down[0:s], style_down[s + 1:]], 0).transpose(0, 1)

            # xuanqu gu ding shu liang de fu yang ben
            index_up = torch.LongTensor(random.sample(range(batch_size - 1), 5))
            index_down = torch.LongTensor(random.sample(range(batch_size - 1), 5))

            negative_style_up = torch.index_select(negative_style_up, 1, index_up.cuda())
            negative_style_down = torch.index_select(negative_style_down, 1, index_down.cuda())

            for i in range(N):
                # sample_style = style_up[s][i:i+1] # 正样本
                style_comparisons_up = torch.cat([style_down[s][i:i + 1], negative_style_up[i]], 0)  # 正样本+副样本

                style_contrastive_loss += self.compute_contrastive_loss(style_up[s][i:i + 1],
                                                                        style_comparisons_up, 0.2, 0)

                style_comparisons_down = torch.cat([style_up[s][i:i + 1], negative_style_down[i]], 0)

                style_contrastive_loss += self.compute_contrastive_loss(style_down[s][i:i + 1],
                                                                        style_comparisons_down, 0.2, 0)

        style_contrastive_loss /= N

        style_contrastive_loss *= self.cfg['contrastive_w']
        self.g_losses['contrastive'] = style_contrastive_loss

        return style_contrastive_loss

    def add_gan_g_loss(self, real_font, real_uni, fake_font, fake_uni):
        """
        add_gan_g_loss
        """
        if self.cfg['gan_w'] == 0.:
            return 0.

        g_loss = -(fake_font.mean() + fake_uni.mean())
        g_loss *= self.cfg['gan_w']
        self.g_losses['gen'] = g_loss

        return g_loss

    def add_gan_d_loss(self, real_font, real_uni, fake_font, fake_uni):
        """
        add_gan_d_loss
        """
        if self.cfg['gan_w'] == 0.:
            return 0.

        d_loss = (F.relu(1. - real_font).mean() + F.relu(1. + fake_font).mean()) + \
                 F.relu(1. - real_uni).mean() + F.relu(1. + fake_uni).mean()

        d_loss *= self.cfg['gan_w']
        self.d_losses['disc'] = d_loss

        return d_loss

    def d_backward(self):
        """
        d_backward
        """
        with utils.temporary_freeze(self.gen):
            d_loss = sum(self.d_losses.values())
            d_loss.backward()

    def g_backward(self):
        """
        g_backward
        """
        with utils.temporary_freeze(self.disc):
            g_loss = sum(self.g_losses.values())
            g_loss.backward()

    def save(self, cur_loss, method, save_freq=None):
        """
        Args:
            method: all / last
                all: save checkpoint by step
                last: save checkpoint to 'last.pth'
                all-last: save checkpoint by step per save_freq and
                          save checkpoint to 'last.pth' always
        """
        if method not in ['all', 'last', 'all-last']:
            return

        step_save = False
        last_save = False
        if method == 'all' or (method == 'all-last' and self.step % save_freq == 0):
            step_save = True
        if method == 'last' or method == 'all-last':
            last_save = True
        assert step_save or last_save

        save_dic = {
            'generator': self.gen.state_dict(),
            'generator_ema': self.gen_ema.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'optimizer': self.g_optim.state_dict(),
            'epoch': self.step,
            'loss': cur_loss
        }
        if self.disc is not None:
            save_dic['discriminator'] = self.disc.state_dict()
            save_dic['d_optimizer'] = self.d_optim.state_dict()
            save_dic['d_scheduler'] = self.d_scheduler.state_dict()

        ckpt_dir = self.cfg['work_dir'] / "checkpoints" / self.cfg['unique_name']
        step_ckpt_name = "{:06d}-{}.pth".format(self.step, self.cfg['name'])
        last_ckpt_name = "last.pth"
        step_ckpt_path = Path.cwd() / ckpt_dir / step_ckpt_name
        last_ckpt_path = ckpt_dir / last_ckpt_name

        log = ""
        if step_save:
            torch.save(save_dic, str(step_ckpt_path))
            log = "Checkpoint is saved to {}".format(step_ckpt_path)

            if last_save:
                utils.rm(last_ckpt_path)
                last_ckpt_path.symlink_to(step_ckpt_path)
                log += " and symlink to {}".format(last_ckpt_path)

        if not step_save and last_save:
            utils.rm(last_ckpt_path)  # last 가 symlink 일 경우 지우고 써줘야 함.
            torch.save(save_dic, str(last_ckpt_path))
            log = "Checkpoint is saved to {}".format(last_ckpt_path)

        self.logger.info("{}\n".format(log))

    def baseplot(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val
        }

        if self.disc is not None:
            tag_scalar_dic.update({
                'train/d_loss': losses.disc.val,
                'train/g_loss': losses.gen.val,
                'train/d_real_font': discs.real_font.val,
                'train/d_real_uni': discs.real_uni.val,
                'train/d_fake_font': discs.fake_font.val,
                'train/d_fake_uni': discs.fake_uni.val,
            })

    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}  D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  R_font {D.real_font_acc.avg:7.3f}  F_font {D.fake_font_acc.avg:7.3f}"
            "  R_uni {D.real_uni_acc.avg:7.3f}  F_uni {D.fake_uni_acc.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
            .format(step=self.step, L=losses, D=discs, S=stats))
