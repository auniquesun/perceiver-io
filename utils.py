import os
import shutil
import logging
import wandb

import torch
from parser import args

from perceiver.model.core import PerceiverEncoder
from perceiver.model.image import ImageInputAdapter
from perceiver.model.pointcloud import PointCloudInputAdapter

import torchvision.transforms as transforms


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_pos = 0
        self.num_neg = 0
        self.total = 0

    def update(self, num_pos, num_neg, n=1):
        self.num_pos += num_pos
        self.num_neg += num_neg
        self.total += n

    def pos_count(self, pred, label):
        # torch.eq(a,b): Computes element-wise equality
        results = torch.eq(pred, label)
        # x.item(): only one element tensors can be converted to Python scalars
        return results.sum().item()


class Logger(object):
    def __init__(self, logger_name='PointMAE', log_level=logging.INFO, log_path='runs', log_file='train.log'):
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
        file_handler = logging.FileHandler(os.path.join(log_path, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger

    def write(self, msg, rank=-1):
        if rank == 0:
            self.logger.info(msg)


def build_model():
    pc_input_adapter = PointCloudInputAdapter(
        pointcloud_shape=(args.num_pt_points, 3),
        num_input_channels=args.num_latent_channels,
        num_groups=args.num_groups,
        group_size=args.group_size)

    # Generic Perceiver encoder
    pc_model = PerceiverEncoder(
        input_adapter=pc_input_adapter,
        num_latents=args.num_pc_latents,  # N
        num_latent_channels=args.num_latent_channels,  # D
        num_cross_attention_heads=args.num_ca_heads,
        num_cross_attention_qk_channels=pc_input_adapter.num_input_channels,  # C
        num_cross_attention_v_channels=None,
        num_cross_attention_layers=args.num_ca_layers,
        first_cross_attention_layer_shared=False,
        cross_attention_widening_factor=args.mlp_widen_factor,
        num_self_attention_heads=args.num_sa_heads,
        num_self_attention_qk_channels=None,
        num_self_attention_v_channels=None,
        num_self_attention_layers_per_block=args.num_sa_layers_per_block,
        num_self_attention_blocks=args.num_sa_blocks,
        first_self_attention_block_shared=True,
        self_attention_widening_factor=args.mlp_widen_factor,
        dropout=args.atten_drop)

    img_input_adapter = ImageInputAdapter(
        image_shape=(224, 224, 3),  # M = 224 * 224
        num_frequency_bands=64)

    # Generic Perceiver encoder
    img_model = PerceiverEncoder(
        input_adapter=img_input_adapter,
        num_latents=args.num_img_latents,  # N
        num_latent_channels=args.num_latent_channels,  # D
        num_cross_attention_heads=args.num_ca_heads,
        num_cross_attention_qk_channels=args.num_latent_channels,  # C
        num_cross_attention_v_channels=None,
        num_cross_attention_layers=args.num_ca_layers,
        first_cross_attention_layer_shared=False,
        cross_attention_widening_factor=args.mlp_widen_factor,
        num_self_attention_heads=args.num_sa_heads,
        num_self_attention_qk_channels=None,
        num_self_attention_v_channels=None,
        num_self_attention_layers_per_block=args.num_sa_layers_per_block,
        num_self_attention_blocks=args.num_sa_blocks,
        first_self_attention_block_shared=True,
        self_attention_widening_factor=args.mlp_widen_factor,
        dropout=args.atten_drop)    

    return pc_model, img_model


def init(exp_name, main_program, model_name):
    if not os.path.exists('runs'):
        os.makedirs('runs')
    if not os.path.exists(os.path.join('runs', exp_name)):
        os.makedirs(os.path.join('runs', exp_name))
    if not os.path.exists(os.path.join('runs', exp_name, 'files')):
        os.makedirs(os.path.join('runs', exp_name, 'files'))
    if not os.path.exists(os.path.join('runs', exp_name, 'models')):
        os.makedirs(os.path.join('runs', exp_name, 'models'))

    shutil.copy(main_program, os.path.join('runs', exp_name, 'files'))
    shutil.copy(f'perceiver/model/core/{model_name}', os.path.join('runs', exp_name, 'files'))
    shutil.copy('utils.py', os.path.join('runs', exp_name, 'files'))
    
    # Actually, wandb login once is enough
    os.environ["WANDB_BASE_URL"] = args.wb_url
    wandb.login(key=args.wb_key)
