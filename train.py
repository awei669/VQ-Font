import json
import sys
import torch
import torch.optim as optim
from pathlib import Path
import argparse
from sconf import Config, dump_args
import utils
import numpy as np
from utils import Logger
from torchvision import transforms
from datasets import (load_lmdb, load_json, read_data_from_lmdb,
                      get_comb_trn_loader, get_cv_comb_loaders)
from trainer import load_checkpoint, CombinedTrainer
from model import generator_dispatch, disc_builder
from model.modules import weights_init
from evaluator import Evaluator


def setup_args_and_config():
    """
    setup_args_and_configs
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("config_paths", nargs="+", help="path/to/config.yaml")
    parser.add_argument("--resume", default=None, help="path/to/saved/.pth")
    parser.add_argument("--use_unique_name", default=False, action="store_true",
                        help="whether to use name with timestamp")

    args, left_argv = parser.parse_known_args()
    assert not args.name.endswith(".yaml")

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml",
                 colorize_modified_item=True)
    cfg.argv_update(left_argv)

    cfg.work_dir = Path(cfg.work_dir)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    if args.use_unique_name:
        timestamp = utils.timestamp()
        unique_name = "{}_{}".format(timestamp, args.name)
    else:
        unique_name = args.name

    cfg.unique_name = unique_name
    cfg.name = args.name

    (cfg.work_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "checkpoints" / unique_name).mkdir(parents=True, exist_ok=True)

    if cfg.save_freq % cfg.val_freq:
        raise ValueError("save_freq has to be multiple of val_freq.")

    return args, cfg


def setup_transforms(cfg):
    """
    setup_transforms
    """
    size = cfg.input_size
    tensorize_transform = [transforms.Resize((size, size)), transforms.ToTensor()]
    if cfg.dset_aug.normalize:
        tensorize_transform.append(transforms.Normalize([0.5], [0.5]))
        cfg.g_args.dec.out = "tanh"

    trn_transform = transforms.Compose(tensorize_transform)
    val_transform = transforms.Compose(tensorize_transform)
    return trn_transform, val_transform


def load_pretrain_vae_model(load_path='path/to/save/pre-train_VQ-VAE', gen=None):
    vae_state_dict = torch.load(load_path, map_location=torch.device('cuda:0'))
    component_objects = vae_state_dict["_vq_vae._embedding.weight"]

    del_key = []
    for key, _ in vae_state_dict.items():
        if "encoder" in key:
            del_key.append(key)

    i = 0
    for param in gen.content_encoder.parameters():
        param.data = vae_state_dict[del_key[i]]
        i += 1
        param.requires_grad = False

    return component_objects


def train(args, cfg, ddp_gpu=-1):
    """
    train
    :param atgs:
    :param cfg:
    :param ddp_gpu:
    :return:
    """
    torch.cuda.set_device(ddp_gpu)
    logger_path = cfg.work_dir / "logs" / "{}.log".format(cfg.unique_name)
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)

    image_scale = 0.6
    writer_path = cfg.work_dir / "runs" / cfg.unique_name
    eval_image_path = cfg.work_dir / "images" / cfg.unique_name
    writer = utils.TBDiskWriter(writer_path, eval_image_path, scale=image_scale)

    args_str = dump_args(args)
    # if is_main_worker(ddp_gpu):
    logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
    logger.info("Args:\n{}".format(args_str))
    logger.info("Configs:\n{}".format(cfg.dumps()))
    logger.info("Unique name: {}".format(cfg.unique_name))
    logger.info("Get dataset ...")

    content_font = cfg.content_font

    trn_transform, val_transform = setup_transforms(cfg)

    env = load_lmdb(cfg.data_path)  # 载入数据库环境lmdb
    env_get = lambda env, x, y, transform: transform(read_data_from_lmdb(env, f'{x}_{y}')['img'])
    # x传入font_path;y传入字符的Unicode编码
    data_meta = load_json(cfg.data_meta)  # load train.json

    get_trn_loader = get_comb_trn_loader
    get_cv_loaders = get_cv_comb_loaders
    Trainer = CombinedTrainer  # 定义trainer

    # 定义训练dset以及dataloader
    trn_dset, trn_loader = get_trn_loader(env,
                                          env_get,
                                          cfg,
                                          data_meta["train"],
                                          trn_transform,
                                          num_workers=cfg.n_workers,
                                          shuffle=True,
                                          drop_last=True)

    # 定义验证dset以及dataloader
    cv_loaders = get_cv_loaders(env,
                                env_get,
                                cfg,
                                data_meta,
                                val_transform,
                                num_workers=0,
                                shuffle=False,
                                drop_last=True)

    logger.info("Build Few-shot model ...")
    # generator
    g_kwargs = cfg.get("g_args", {})
    g_cls = generator_dispatch()
    gen = g_cls(1, cfg.C, 1, cfg, **g_kwargs)
    gen.cuda()
    gen.apply(weights_init(cfg.init))

    logger.info("Load pre-train model...")
    component_objects = load_pretrain_vae_model(cfg.vae_pth, gen)

    if cfg.gan_w > 0.:
        d_kwargs = cfg.get("d_args", {})
        disc = disc_builder(cfg.C, trn_dset.n_fonts, trn_dset.n_unis, **d_kwargs)
        # trn_dset.n_fonts训练集中的字体数,trn_dset.n_unis数据集中所有的字符
        disc.cuda()
        disc.apply(weights_init(cfg.init))
    else:
        disc = None

    g_optim = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)
    d_optim = optim.Adam(disc.parameters(), lr=cfg.d_lr, betas=cfg.adam_betas)
    gen_scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=cfg['step_size'], gamma=cfg['gamma'])
    dis_scheduler = torch.optim.lr_scheduler.StepLR(d_optim, step_size=cfg['step_size'], gamma=cfg['gamma']) \
        if disc is not None else None

    # logger.info("Gen struct:{}"
    #             "Dis struct:{}"
    #             .format(gen, disc))

    st_step = 1
    if args.resume:
        st_step, loss = load_checkpoint(args.resume, gen, disc, g_optim, d_optim, gen_scheduler, dis_scheduler)
        logger.info("Resumed checkpoint from {} (Step {}, Loss {:7.3f})".format(
            args.resume, st_step - 1, loss))
        if cfg.overwrite:
            st_step = 1
        else:
            pass

    envaluator = Evaluator(env,
                           env_get,
                           cfg,
                           logger,
                           writer,
                           cfg.batch_size,
                           val_transform,
                           content_font,
                           use_half=cfg.use_half)

    trainer = Trainer(gen, disc, g_optim, d_optim, gen_scheduler, dis_scheduler,
                      logger, envaluator, cv_loaders, cfg)

    with open(cfg.sim_path, 'r+') as file:
        chars_sim = file.read()

    chars_sim_dict = json.loads(chars_sim)  # 将json格式文件转化为python的字典文件

    trainer.train(trn_loader, st_step, cfg["iter"], component_objects, chars_sim_dict)


def main():
    args, cfg = setup_args_and_config()
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train(args, cfg)


if __name__ == "__main__":
    main()
