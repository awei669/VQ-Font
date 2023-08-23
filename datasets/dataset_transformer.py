import torch
import json
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os
import cv2 as cv
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CombTrainDataset(Dataset):
    """
    CombTrainDataset
    """
    def __init__(self, env, env_get, avails, all_content_json, content_font, transform=None):
        self.env = env
        self.env_get = env_get

        self.num_Positive_samples = 2 #一种font中取几个样本
        self.k_shot = 3

        with open(all_content_json, 'r') as f:
            self.all_characters = json.load(f)

        self.avails = avails #可用于训练的字体 data_meta["train"]格式为{"fontname1":[charlist],...,fontname2":[charlist]}
        self.unis = sorted(self.all_characters)
        self.fonts = list(self.avails) #获取所有训练字体的name
        self.n_fonts = len(self.fonts) #num of fonts(train)
        self.n_unis = len(self.unis)

        self.content_font = content_font
        self.transform = transform  #数据预处理方法

    def random_get_trg(self, avails, font_name):
        target_list = list(avails[font_name])
        trg_uni = np.random.choice(target_list, self.num_Positive_samples * 4)
        return [str(trg_uni[i]) for i in range(0, self.num_Positive_samples * 4)]

    def sample_pair_style(self, font, ref_unis):

        try:
            imgs = torch.concat([self.env_get(self.env, font, uni, self.transform) for uni in ref_unis])
            # print(imgs.shape)
        except:
            return None

        return imgs

    def __getitem__(self, index):
        font_idx = index % self.n_fonts #获取到font的索引
        font_name = self.fonts[font_idx] #获取到fontname
        while True:
            # randomly choose target
            style_unis = self.random_get_trg(self.avails, font_name) #从font中随机选择n*4个字符,前n个字符作为重构字符，后面3n个字符作为ref
            trg_unis = style_unis[:self.num_Positive_samples] #前n个作为重构的目标字符
            sample_index = torch.tensor([index])

            avail_unis = self.avails[font_name]

            ref_unis = style_unis[self.num_Positive_samples:] #后面的3n个字符作为参考字符，不重复
            # print(style_unis)
            # print(trg_unis)
            # print(ref_unis)

            style_imgs = torch.stack([self.sample_pair_style(font_name, ref_unis[i*3:(i+1)*3])for i in range(0, self.num_Positive_samples)], 0)

            # style_imgs = self.sample_pair_style(font_name, ref_unis) #参考字符的图片,len=3*n,n为正样本数量
            # print("style_imgs",style_imgs.shape)

            if style_imgs is None:
                continue

            #add trg_imgs
            trg_imgs = torch.stack([self.env_get(self.env, font_name, uni, self.transform)
                                  for uni in trg_unis],0)

            # trg_imgs = torch.concat([self.env_get(self.env, font_name, uni, self.transform)
            #                         for uni in trg_unis])

            # print("trg_imgs", trg_imgs.shape)

            trg_uni_ids = [self.unis.index(uni) for uni in trg_unis] #目标字符的index
            font_idx = torch.tensor([font_idx])

            content_imgs = torch.stack([self.env_get(self.env, self.content_font, uni, self.transform)
                                      for uni in trg_unis], 0) #从内容字体中选出目标字符

            # content_imgs = torch.concat([self.env_get(self.env, self.content_font, uni, self.transform)
            #                             for uni in trg_unis])  # 从内容字体中选出目标字符

            # print("content_imgs", content_imgs.shape)

            ret = (
                torch.repeat_interleave(font_idx, style_imgs.shape[1]),
                style_imgs,
                torch.repeat_interleave(font_idx, trg_imgs.shape[1]),
                torch.tensor(trg_uni_ids),
                trg_imgs,
                content_imgs,
                trg_unis[0],
                torch.repeat_interleave(sample_index, style_imgs.shape[1]),
                sample_index,
                ref_unis[:self.k_shot]
            )

            return ret

    def __len__(self):
        return sum([len(v) for v in self.avails.values()])


    @staticmethod
    def collate_fn(batch):
        (style_ids, style_imgs,
         trg_ids, trg_uni_ids, trg_imgs, content_imgs, trg_unis, style_sample_index, trg_sample_index,ref_unis ) = zip(*batch)

        ret = (
            torch.concat(style_ids), #做reference的font的index
            torch.cat(style_imgs,1).unsqueeze_(2), #reference image set
            torch.concat(trg_ids), #目标font的index，跟reference相同
            torch.concat(trg_uni_ids), #目标字符的index
            torch.cat(trg_imgs,1).unsqueeze_(2), #重构的目标字符图片
            torch.cat(content_imgs,1).unsqueeze_(2), #获取内容的内容字符图片
            trg_unis, #目标的字符
            torch.concat(style_sample_index),
            torch.concat(trg_sample_index),
            ref_unis
        )

        return ret


class CombTestDataset(Dataset):
    """
    CombTestDataset
    """

    def __init__(self, env, env_get, target_fu, avails, all_content_json, content_font, language="chn",
                 transform=None, ret_targets=True):

        self.fonts = list(target_fu) #获取font列表,数量为cfg.cv_n_fonts
        self.n_uni_per_font = len(target_fu[list(target_fu)[0]]) #每种font中有cfg.cv_n_unis个字符
        self.fus = [(fname, uni) for fname, unis in target_fu.items() for uni in unis] #生成key-value [(font1-char1), (font2-char1)]
        # print(self.fus)
        # self.unis = sorted(set.union(*map(set, avails.values()))) #获取所有的字符
        self.env = env
        self.env_get = env_get
        self.avails = avails

        self.transform = transform
        self.ret_targets = ret_targets
        self.content_font = content_font

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                       }

        self.to_int = to_int_dict[language.lower()]

    def sample_pair_style(self, avail_unis):
        style_unis = random.sample(avail_unis, 3)
        return list(style_unis)

    def __getitem__(self, index):
        font_name, trg_uni = self.fus[index] #根据index确定重构字符的以及font名称
        font_idx = self.fonts.index(font_name)
        sample_index = torch.tensor([index])

        avail_unis = self.avails[font_name]
        style_unis = self.sample_pair_style(avail_unis) #在style font中选择三个参考

        try:
            a = [self.env_get(self.env, font_name, uni, self.transform) for uni in style_unis]
        except:
            print(font_name, style_unis)

        style_imgs = torch.stack(a)

        font_idx = torch.tensor([font_idx])
        trg_dec_uni = torch.tensor([self.to_int(trg_uni)])

        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform) #取出内容img

        ret = (
            torch.repeat_interleave(font_idx, len(style_imgs)),
            style_imgs,
            font_idx,
            trg_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)),  # style sample index
            sample_index,  # trg sample index
            content_img,
            trg_uni,
            style_unis
        )

        if self.ret_targets:
            try:
                trg_img = self.env_get(self.env, font_name, trg_uni, self.transform)
            except:
                trg_img = torch.ones(size=(1,128,128))
            ret +=(trg_img,)

        return ret


    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):

        style_ids, style_imgs, trg_ids, trg_unis, style_sample_index, trg_sample_index, content_imgs, trg_uni, style_unis, *left = list(zip(*batch))
        ret = (
            torch.cat(style_ids),    #font_index
            torch.cat(style_imgs),   #reference images
            torch.cat(trg_ids),      #font_index
            torch.cat(trg_unis),     #目标字符
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1),   #内容字符图片
            trg_uni,
            style_unis
        )

        if left:
            trg_imgs = left[0]
            ret += (torch.concat(trg_imgs).unsqueeze_(1),)

        return ret


class CombTrain_VQ_VAE_dataset(Dataset):
    """
    CombTrain_VQ_VAE_dataset,用于训练components码本的dataset，训练数据从content_font中取
    """

    def __init__(self, root, transform = None):
        self.img_path = root
        self.transform = transform
        self.imgs = self.read_file(self.img_path)
        # img = Image.open(self.imgs[0])
        # img = self.transform(img)
        # print(img.shape)


    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list


    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img) #Tensor [C H W] [1 128 128]
        return img

    def __len__(self):

        return len(self.imgs)


class FixedRefDataset(Dataset):
    '''
    FixedRefDataset
    '''
    def __init__(self, env, env_get, target_dict, ref_unis, k_shot,
                 all_content_json, content_font, language="chn",  transform=None, ret_targets=True):
        '''
        ref_unis: target unis
        target_dict: {style_font: [uni1, uni2, uni3]}
        '''
        self.target_dict = target_dict
        self.ref_unis = sorted(ref_unis)
        self.fus = [(fname, uni) for fname, unis in target_dict.items() for uni in unis]
        self.k_shot = k_shot
        with open(all_content_json, 'r') as f:
            self.cr_mapping = json.load(f)

        self.content_font = content_font
        self.fonts = list(target_dict)

        self.env = env
        self.env_get = env_get

        self.transform = transform
        self.ret_targets = ret_targets

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                       }

        self.to_int = to_int_dict[language.lower()]


    def sample_pair_style(self, font, style_uni):
        style_unis = random.sample(style_uni, 3)
        imgs = torch.concat([self.env_get(self.env, font, uni, self.transform) for uni in style_unis])
        return imgs, list(style_unis)

    def __getitem__(self, index):
        fname, trg_uni = self.fus[index]
        sample_index = torch.tensor([index])

        fidx = self.fonts.index(fname)
        avail_unis = list(set(self.ref_unis) - set([trg_uni]))
        style_imgs, style_unis = self.sample_pair_style(fname, self.ref_unis)

        fidces = torch.tensor([fidx])
        trg_dec_uni = torch.tensor([self.to_int(trg_uni)])
        style_dec_uni = torch.tensor([self.to_int(style_uni) for style_uni in style_unis])

        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)

        ret = (
            torch.repeat_interleave(fidces, len(style_imgs)),  # fidces,
            style_imgs,
            fidces,
            trg_dec_uni,
            style_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)),
            sample_index,
            content_img,
            trg_uni,
            style_unis
        )

        if self.ret_targets:
            trg_img = self.env_user_get(self.env_user, fname, trg_uni, self.transform)
            ret += (trg_img,)

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):
        style_ids, style_imgs, trg_ids, trg_unis, style_uni, style_sample_index, trg_sample_index, content_imgs, trg_uni,\
        style_unis, *left = list(zip(*batch))
        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_uni),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1),
            trg_uni,
            style_unis
        )
        if left:
            trg_imgs = left[0]
            ret += (torch.concat(trg_imgs).unsqueeze_(1),)

        return ret









