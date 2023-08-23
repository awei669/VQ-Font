from pathlib import Path

import torch
import numpy as np
import utils
import os
import tqdm
import cv2
from datasets import cyclize
from utils import Logger
from datasets import load_lmdb, read_data_from_lmdb


def torch_eval(val_fn):
    @torch.no_grad()
    def decorated(self, gen, *args, **kwargs):
        gen.eval()
        ret = val_fn(self, gen, *args, **kwargs)
        gen.train()
        return ret

    return decorated


class Evaluator:
    def __init__(self, env, env_get, cfg, logger, writer, batch_size, transform,
                 content_font, use_half=False):
        # torch.backends.cudnn.benchmark = True

        self.env = env
        self.env_get = env_get
        self.logger = logger
        self.writer = writer
        self.batch_size = batch_size
        self.transform = transform
        self.k_shot = cfg.kshot
        self.content_font = content_font
        self.use_half = use_half
        self.size = cfg.input_size

    def cp_validation(self, gen, cv_loaders, step, learned_components, chars_sim_dict, phase="fact", reduction='mean',
                      ext_tag=""):
        """
        cp_validation
        """
        # cv_loaders包含四个loader,
        for tag, loader in cv_loaders.items():
            self.comparable_val_saveimg(gen, loader, step, learned_components, chars_sim_dict,
                                        tag=f"comparable_{tag}_{ext_tag}",
                                        phase=phase, reduction=reduction)

    @torch_eval
    def comparable_val_saveimg(self, gen, loader, step, learned_components, chars_sim_dict, phase="fact",
                               tag='comparable', reduction='mean'):
        n_row = loader.dataset.n_uni_per_font  # 每个loader中未见过的字符数量
        compare_batches = self.infer_loader(gen, loader, learned_components, chars_sim_dict, phase=phase,
                                            reduction=reduction)
        comparable_grid = utils.make_comparable_grid(*compare_batches[::-1], nrow=n_row)
        self.writer.add_image(tag, comparable_grid, global_step=step)
        return comparable_grid

    @torch_eval
    def infer_loader(self, gen, loader, learned_components, chars_sim_dict, phase, reduction="mean"):
        # 分别对传入的loader进行推理,即验证当前模型的生成能力
        outs = []
        trgs = []
        styles = []

        for i, (style_ids, style_imgs, trg_ids, trg_unis, style_sample_index,
                trg_sample_index, content_imgs, trg_uni, style_unis, *trg_imgs) in enumerate(loader):

            # 对4类字符进行验证
            out, _, _ = gen.infer(style_ids, style_imgs, style_sample_index, trg_ids, content_imgs,
                                  learned_components, trg_uni, style_unis, chars_sim_dict, k_shot_tag=True,
                                  reduction=reduction)

            batch_size = out.shape[0]
            out_images = out.detach().cpu().numpy()
            out_duplicate = np.ones((batch_size * self.k_shot, 1, self.size, self.size))
            for idx in range(batch_size):
                for j in range(self.k_shot):
                    out_duplicate[idx * self.k_shot + j, ...] = out_images[idx, ...]

            outs.append(torch.Tensor(out_duplicate))

            for style_img in style_imgs:
                style_duplicate = np.ones((1, 1, self.size, self.size))
                style_duplicate[:, :, :, :] = style_img.unsqueeze(1).detach().cpu()
                styles.append(torch.Tensor(style_duplicate))

            if trg_imgs:
                trg_images = trg_imgs[0].detach().cpu().numpy()
                trg_duplicate = np.zeros((batch_size * self.k_shot, 1, self.size, self.size))
                for idx in range(batch_size):
                    for j in range(self.k_shot):
                        trg_duplicate[idx * self.k_shot + j, ...] = trg_images[idx, ...]
                trgs.append(torch.Tensor(trg_duplicate))

        ret = (torch.cat(outs).float(),)
        if trgs:
            ret += (torch.cat(trgs).float(),)

        ret += (torch.cat(styles).float(),)
        return ret

    def normalize(self, tensor, eps=1e-5):
        """ Normalize tensor to [0, 1] """
        # eps=1e-5 is same as make_grid in torchvision.
        minv, maxv = tensor.min(), tensor.max()
        tensor = (tensor - minv) / (maxv - minv + eps)
        return tensor

    @torch_eval
    def save_each_imgs(self, gen, loader, ori_img_root, learned_components, chars_sim_dict, save_dir, reduction='mean'):
        '''
        save_each_imgs
        '''
        font_name = os.path.basename(save_dir)
        output_folder = os.path.join(save_dir, 'images')
        os.makedirs(output_folder, exist_ok=True)
        ch_list_check = []

        i = 0
        while i < len(loader):
            for i, (style_ids, style_imgs, trg_ids, trg_unis, style_uni, style_sample_index, trg_sample_index,
                    content_imgs, trg_uni, style_unis) in enumerate(loader):

                print(i)

                out, _, _ = gen.infer(style_ids, style_imgs, style_sample_index, trg_ids, content_imgs,
                                      learned_components, trg_uni, style_unis, chars_sim_dict, k_shot_tag=True,
                                      reduction=reduction)
                dec_unis = trg_unis.detach().cpu().numpy()
                style_dec_unis = style_uni.detach().cpu().numpy()
                font_ids = trg_ids.detach().cpu().numpy()
                images = out.detach().cpu()  # [B, 1, 128, 128]
                for idx, (dec_uni, font_id, image) in enumerate(zip(dec_unis, font_ids, images)):
                    font_name = loader.dataset.fonts[font_id]  # name.ttf
                    uni = hex(dec_uni)[2:].upper().zfill(4)
                    ch = '\\u{:s}'.format(uni).encode().decode('unicode_escape')
                    image = self.normalize(image)
                    final_img = torch.permute(torch.clip(image * 255, min=0, max=255), (1, 2, 0)).cpu().numpy()
                    if final_img.shape[-1] == 1:
                        final_img = final_img.squeeze(-1)  # [128, 128]

                    dst_path = os.path.join(output_folder, ch + '.png')
                    ch_list_check.append(ch)
                    cv2.imwrite(dst_path, final_img)
            i += 1

        print('num_saved_img: ', len(ch_list_check))
        return output_folder
