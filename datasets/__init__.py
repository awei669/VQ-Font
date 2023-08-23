from .ttf_utils import get_filtered_chars, read_font, render

from .lmdbutils import (load_lmdb, load_json, read_data_from_lmdb)
from .dataset_transformer import CombTrainDataset, CombTestDataset, CombTrain_VQ_VAE_dataset, FixedRefDataset
from torch.utils.data import DataLoader
from .datautils import uniform_sample, cyclize


def get_comb_trn_loader(env, env_get, cfg, train_dict, transform, **kwargs):
    """
    get_comb_trn_loader
    """
    dset = CombTrainDataset(
        env,
        env_get,
        train_dict,
        all_content_json=cfg.all_content_json,
        content_font=cfg.content_font,
        **cfg.get('dset_args', {}),
        transform=transform
    )

    loader = DataLoader(dset, batch_size=cfg.batch_size, batch_sampler=None,
                        collate_fn=dset.collate_fn, **kwargs)

    # loader = DataLoader(dset, batch_size=cfg.batch_size, batch_sampler=None, **kwargs)

    return dset, loader


def get_comb_test_loader(env, env_get, target_dict, cfg, avails, transform, ret_targets=True, **kwargs):
    """
    get_comb_test_loader
    """
    dset = CombTestDataset(
        env,
        env_get,
        target_dict,
        avails,
        all_content_json=cfg.all_content_json,
        content_font=cfg.content_font,
        language=cfg.language,
        transform=transform,
        ret_targets=ret_targets
    )

    # loader = DataLoader(dset, batch_size=cfg.batch_size, **kwargs)

    loader = DataLoader(dset, batch_size=cfg.batch_size,
                        collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


def get_cv_comb_loaders(env, env_get, cfg, data_meta, transform, **kwargs):
    n_unis = cfg.cv_n_unis
    n_fonts = cfg.cv_n_fonts

    ufs = uniform_sample(data_meta["valid"]["unseen_fonts"], n_fonts)
    sfs = uniform_sample(data_meta["valid"]["seen_fonts"], n_fonts)
    sus = uniform_sample(data_meta["valid"]["seen_unis"], n_unis)  # ['6E6B']
    uus = uniform_sample(data_meta["valid"]["unseen_unis"], n_unis)  # ['6E6B']

    sfsu_dict = {fname: sus for fname in sfs}  # 见过的字符见过的字体 6x20
    sfuu_dict = {fname: uus for fname in sfs}  # 没有见过的字符见过字体 6x20
    ufsu_dict = {fname: sus for fname in ufs}  # 见过的字符没见过的字体 6x20
    ufuu_dict = {fname: uus for fname in ufs}  # 没有见过的字符和字体 6x20

    cv_loaders = {
        'sfsu': get_comb_test_loader(env, env_get, sfsu_dict, cfg, data_meta['avail'], transform, **kwargs)[1],
        'sfuu': get_comb_test_loader(env, env_get, sfuu_dict, cfg, data_meta['avail'], transform, **kwargs)[1],
        'ufsu': get_comb_test_loader(env, env_get, ufsu_dict, cfg, data_meta['avail'], transform, **kwargs)[1],
        'ufuu': get_comb_test_loader(env, env_get, ufuu_dict, cfg, data_meta['avail'], transform, **kwargs)[1]
        }

    return cv_loaders


def get_fixedref_loader(env, env_get, target_dict, ref_unis, cfg, transform, **kwargs):
    """
    inference
    """
    dset = FixedRefDataset(env,
                           env_get,
                           target_dict,
                           ref_unis,
                           k_shot=cfg.kshot,
                           all_content_json=cfg.all_content_json,
                           content_font=cfg.content_font,
                           language=cfg.language,
                           transform=transform,
                           ret_targets=False
                           )

    loader = DataLoader(dset, batch_size=cfg.batch_size,
                        collate_fn=dset.collate_fn, **kwargs)

    return loader
