from .base_trainer import BaseTrainer
import utils
from datasets import cyclize
import torch

torch.autograd.set_detect_anomaly = True


class CombinedTrainer(BaseTrainer):
    """
    CombinedTrainer
    """

    def __init__(self, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                 logger, evaluator, cv_loaders, cfg):  # cls_char
        super().__init__(gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                         logger, evaluator, cv_loaders, cfg)

    def train(self, loader, st_step=1, max_step=100000, component_embeddings=None, chars_sim_dict=None):
        # loader中存放了一个batch的数据
        """
        train
        """
        self.gen.train()
        if self.disc is not None:
            self.disc.train()

        # loss stats
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen", "contrastive")
        # discriminator stats
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni")
        # etc stats
        stats = utils.AverageMeters("B_style", "B_target")
        self.step = st_step
        self.clear_losses()
        self.logger.info("Start training FewShot ...")

        while True:
            for (in_style_ids, in_imgs, trg_style_ids, trg_uni_ids, trg_imgs,
                 content_imgs, trg_unis, style_sample_index, trg_sample_index, ref_unis) in cyclize(loader):
                """
                in_style_ids:reference font的index,长度为3
                in_imgs:reference image list
                trg_style_ids:生成的目标font的index,长度为1
                trg_uni_ids:生成目标字符的index
                trg_imgs:生成目标字符的GT image
                content_imgs:参考的内容字符image
                trg_unis:需要生成的字符、需要重构的字符
                style_sample_index:loader传入的index,长度为3
                trg_sample_index:目标的index
                len(loader)代表full train dataset需要迭代的次数
                """
                epoch = self.step // len(loader)
                B = trg_imgs.shape[0]
                stats.updates({
                    "B_style": in_imgs.shape[0],
                    "B_target": B
                })

                in_style_ids = in_style_ids.cuda()  # [font1 x 3,font2 x 3,...,fontn x 3];num=len(cfg.batch_size)
                in_imgs = in_imgs.cuda()  # [B*3*2,C,H,W]   [B*3*2,1,128,128] 每一个batch内有
                content_imgs = content_imgs.cuda()  # [B*2,C,H,W]
                trg_uni_disc_ids = trg_uni_ids.cuda()
                trg_style_ids = trg_style_ids.cuda()
                trg_imgs = trg_imgs.cuda()

                #  复制codebook为batch
                bs_component_embeddings = self.get_codebook_detach(component_embeddings)

                ####################################################
                # infer
                ####################################################

                # 得到风格特征
                self.gen.encode_write_comb(in_style_ids, style_sample_index, in_imgs[0])  # [B*3,256,16,16]

                # 生成目标图像
                out_1, style_components_1 = self.gen.read_decode(trg_style_ids, trg_sample_index,
                                                                 content_imgs[0],
                                                                 bs_component_embeddings,
                                                                 trg_unis,
                                                                 ref_unis,
                                                                 chars_sim_dict)  # fake_img && 变换后的特征 && qs风格化的部件

                self.gen.encode_write_comb(in_style_ids, style_sample_index, in_imgs[1])  # [B*3,256,16,16]

                _, style_components_2 = self.gen.read_decode(trg_style_ids, trg_sample_index,
                                                             content_imgs[1],
                                                             bs_component_embeddings,
                                                             trg_unis,
                                                             ref_unis,
                                                             chars_sim_dict)  # fake_img && 变换后的特征 && qs风格化的部件

                # reconstruct img
                self_infer_imgs, style_components, feat_recons = self.gen.infer(trg_style_ids, trg_imgs[0],
                                                                                trg_style_ids,
                                                                                trg_sample_index, content_imgs[0],
                                                                                bs_component_embeddings)

                real_font, real_uni = self.disc(trg_imgs[0], trg_style_ids,
                                                trg_uni_disc_ids[0::self.num_postive_samples])
                # GT图像以及font id和character id
                fake_font, fake_uni = self.disc(out_1.detach(), trg_style_ids,
                                                trg_uni_disc_ids[0::self.num_postive_samples])

                fake_font_recon, fake_uni_recon = self.disc(self_infer_imgs.detach(), trg_style_ids,
                                                            trg_uni_disc_ids[0::self.num_postive_samples])
                self.add_gan_d_loss(real_font, real_uni, fake_font + fake_font_recon,
                                    fake_uni + fake_uni_recon)

                # 辨别器计算梯度并更新参数(固定生成器的参数)
                self.d_backward()  # 计算反向传播求解梯度
                self.d_optim.step()  # 更新权重参数
                self.d_scheduler.step()  # 通过step_size来更新学习率
                self.d_optim.zero_grad()  # 清空梯度

                fake_font, fake_uni = self.disc(out_1, trg_style_ids, trg_uni_disc_ids[0::self.num_postive_samples])

                # reconstruction
                # fake_font_recon, fake_uni_recon = 0, 0
                fake_font_recon, fake_uni_recon = self.disc(self_infer_imgs, trg_style_ids,
                                                            trg_uni_disc_ids[0::self.num_postive_samples])
                self.add_gan_g_loss(real_font, real_uni, fake_font + fake_font_recon,
                                    fake_uni + fake_uni_recon)
                self.add_pixel_loss(out_1, trg_imgs[0], self_infer_imgs)
                self.style_contrastive_loss(style_components_1, style_components_2, self.batch_size)

                # 生成器参数反向传播并更新(固定辨别器的参数)
                self.g_backward()  # 计算反向传播求解梯度
                self.g_optim.step()  # 更新权重参数
                self.g_scheduler.step()  # 通过step_size来更新学习率
                self.g_optim.zero_grad()  # 清空梯度

                discs.updates({
                    "real_font": real_font.mean().item(),
                    "real_uni": real_uni.mean().item(),
                    "fake_font": fake_font.mean().item(),
                    "fake_uni": fake_uni.mean().item(),
                }, B)

                loss_dic = self.clear_losses()
                losses.updates(loss_dic, B)  # accum loss stats

                self.accum_g()
                if self.step % self.cfg['tb_freq'] == 0:
                    self.baseplot(losses, discs, stats)

                if self.step % self.cfg['print_freq'] == 0:
                    self.log(losses, discs, stats)
                    self.logger.debug("GPU Memory usage: max mem_alloc = %.1fM / %.1fM",
                                      torch.cuda.max_memory_allocated() / 1000 / 1000,
                                      torch.cuda.max_memory_reserved() / 1000 / 1000)
                    losses.resets()
                    discs.resets()
                    stats.resets()

                if self.step % self.cfg['val_freq'] == 0:
                    epoch = self.step / len(loader)
                    self.logger.info("Validation at Epoch = {:.3f}".format(epoch))
                    self.evaluator.cp_validation(self.gen_ema, self.cv_loaders, self.step,
                                                 bs_component_embeddings, chars_sim_dict)
                    self.save(loss_dic['g_total'], self.cfg['save'], self.cfg.get('save_freq', self.cfg['val_freq']))

                if self.step >= max_step:
                    break

                self.step += 1

            if self.step >= max_step:
                break

        self.logger.info("Iteration finished.")

    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}   Contrastive {L.contrastive.avg:7.4f}"
            "  D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
            .format(step=self.step, L=losses, D=discs, S=stats))
