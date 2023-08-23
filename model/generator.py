import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder import dec_builder, Integrator
from model.content_encoder import content_enc_builder
from model.references_encoder import comp_enc_builder
from model.Component_Attention_Module import ComponentAttentiomModule, Get_style_components
from model.memory import Memory


class Generator(nn.Module):
    """
    Generator
    """

    def __init__(self, C_in, C, C_out, cfg, comp_enc, dec, content_enc, integrator_args):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.kshot = cfg.kshot
        self.component_encoder = comp_enc_builder(C_in, C, **comp_enc)  # 构建部件风格编码器
        self.mem_shape = self.component_encoder.out_shape  # [256, 16, 16]
        assert self.mem_shape[-1] == self.mem_shape[-2]  # H == W

        self.Get_style_components = Get_style_components()
        self.Get_style_components_1 = Get_style_components()
        self.Get_style_components_2 = Get_style_components()
        self.cam = ComponentAttentiomModule()

        # memory
        self.memory = Memory()

        self.shot = cfg.kshot

        C_content = content_enc['C_out']
        C_reference = comp_enc['C_out']
        self.content_encoder = content_enc_builder(C_in, C, **content_enc)

        self.decoder = dec_builder(
            C, C_out, **dec
        )

        self.Integrator = Integrator(C * 8, **integrator_args, C_content=C_content, C_reference=C_reference)

        self.Integrator_local = Integrator(C * 8, **integrator_args, C_content=C_content, C_reference=0)

    def reset_memory(self):
        """
        reset memory
        """
        self.memory.reset_memory()

    def read_decode(self, target_style_ids, trg_sample_index, content_imgs, learned_components, trg_unis, ref_unis,
                    chars_sim_dict, reset_memory=True, reduction='mean'):
        """
        decode
        :param target_style_ids:
        :param trg_sample_index:
        :param content_imgs:
        :param reset_memory:
        :param reduction:
        :return:
        """

        # print(target_style_ids)
        # print(trg_sample_index)

        reference_feats = self.memory.read_chars(target_style_ids, trg_sample_index, reduction=reduction)
        reference_feats = torch.stack([x for x in reference_feats])  # 参考图片特征[B,3,C,H,W]

        # print("reference_feats", reference_feats.shape)

        # print("content_imgs",content_imgs.shape)
        content_feats = self.content_encoder(content_imgs)  # 目标内容图片[B,C,H,W]
        # print("content_feats", content_feats.shape)

        try:
            style_components = self.Get_style_components(learned_components, reference_feats)  # [B,N,C]
            style_components = self.Get_style_components_1(style_components, reference_feats)
            style_components = self.Get_style_components_2(style_components, reference_feats)
        except Exception as e:
            traceback.print_exc()

        sr_features = self.cam(content_feats, learned_components, style_components)  # 变换后的特征
        # print(sr_features.shape)

        # print("sr_features",sr_features.shape)
        # print("content_feats", content_feats.shape)

        # 融合全局的风格特征global_feature
        global_style_features = self.Get_style_global(trg_unis, ref_unis, reference_feats, chars_sim_dict)
        all_features = self.Integrator(sr_features, content_feats, global_style_features)
        # all_features = self.Integrator_infer(sr_features, content_feats, None)
        out = self.decoder(all_features)  # 解码器生成图片

        if reset_memory:
            self.reset_memory()

        return out, style_components

    def encode_write_comb(self, style_ids, style_sample_index, style_imgs, reset_memory=True):
        """
        encode && memory component features
        :param style_ids:
        :param style_sample_index:
        :param style_imgs:
        :param reset_memory:
        :return:
        """

        if reset_memory:
            self.reset_memory()

        feats = self.component_encoder(style_imgs)
        feat_scs = feats["last"]
        self.memory.write_comb(style_ids, style_sample_index, feat_scs)
        # 将部件编码器中的内容按照风格id以及在当前font中的index存放features

        return feat_scs

    def CosineSimilarity(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

    def Get_style_global(self, trg_unis, ref_unis, reference_feats, chars_sim_dict):
        """
        计算reference_set经过content_encodr和conten_feature的相似性权重
        reference_content_features [KB C H W]
        content_feature [B C H W]
        reference_feats [B K C H W]
        """
        # 转化成 [KB CHW]
        # vetor_reference = reference_content_features.view(reference_content_features.shape[0], -1)
        # vetor_content = content_feature.view(content_feature.shape[0], -1)
        # global_feature = torch.zeros_like(content_feature)

        list_trg_chars = list(trg_unis)
        list_ref_unis = list(ref_unis)
        # print(list_trg_chars)
        # print(chars_sim_dict.keys())
        # print("ref_unis",ref_unis)
        B, K, C, H, W = reference_feats.shape
        global_feature = torch.zeros([B, C, H, W]).cuda()
        for i in range(0, B):
            distance_0 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][0]]

            distance_1 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][1]]

            distance_2 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][2]]

            weight = torch.tensor([distance_0, distance_1, distance_2])
            t = 1
            weight = F.softmax(weight / t, dim=0)

            global_feature[i] = reference_feats[i][0] * weight[0] + reference_feats[i][1] * weight[1] \
                                + reference_feats[i][2] * weight[2]

        return global_feature

    def infer(self, in_style_ids, in_imgs, style_sample_index, trg_style_ids,
              content_imgs, learned_components, trg_unis=None, ref_unis=None, chars_sim_dict=None,
              reduction="mean", k_shot_tag=False):
        """
        generate images
        :return:生成图片以及其特征
        """
        in_style_ids = in_style_ids.cuda()
        in_imgs = in_imgs.cuda()  # GT图像

        infer_size = content_imgs.size()[0]
        learned_components = learned_components[:infer_size]

        content_imgs = content_imgs.cuda()

        reference_feats = self.encode_write_comb(in_style_ids, style_sample_index, in_imgs)  # [B,C,H,W]

        if not k_shot_tag:
            reference_feats = reference_feats.unsqueeze(1)  # [B,K,C,H,W]GT作为参考K=1
        else:
            KB, C, H, W = reference_feats.size()
            reference_feats = torch.reshape(reference_feats, (KB // self.shot, self.shot, C, H, W))

        # print(reference_feats.shape)
        # 参考的font的style_id以及其中的图片和编号,得到reference_feature_map_list,参考图像为GT[B,C,H,W]

        content_feats = self.content_encoder(content_imgs)  # 目标内容图片
        content_feats = content_feats.cuda()
        # print(content_feats.shape) #[B,C,H,W]   [B,256,16,16]

        style_components = self.Get_style_components(learned_components, reference_feats)  # [B,N,C]
        style_components = self.Get_style_components_1(style_components, reference_feats)
        style_components = self.Get_style_components_2(style_components, reference_feats)

        sr_features = self.cam(content_feats, learned_components, style_components)  # 变换后的特征
        # print(sr_features.shape)

        if k_shot_tag:
            global_style_features = self.Get_style_global(trg_unis, ref_unis, reference_feats, chars_sim_dict)
            all_features = self.Integrator(sr_features, content_feats, global_style_features)
            # all_features = self.Integrator_infer(sr_features, content_feats, None)
        else:
            all_features = self.Integrator(sr_features, content_feats, reference_feats.squeeze())
            # all_features = self.Integrator_infer(sr_features, content_feats, None)

        out = self.decoder(all_features)  # 解码器生成图片,decoder只能接受256通道，在送入decoder之前需要将风格特征和内容特征融合后送入

        return out, style_components, sr_features

