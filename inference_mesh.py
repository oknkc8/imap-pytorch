import os
import datetime

import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import argparse

from imap.model.nerf_copy import NERF
from imap.utils.utils import *
from imap.utils.torch_math import *
from imap.utils.sdf_meshing import *
from imap.data.tum_dataset_factory import TUMDatasetFactory
from config import cfg, update_config

from tensorboardX import SummaryWriter
from tqdm import tqdm
from loguru import logger

import pdb


def main():
    """
    Create Log Files
    """
    parser = argparse.ArgumentParser(description='A PyTorch Implementation Neural Mapping')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--checkpoint',
                        help='experiment checkpoint file name',
                        required=True,
                        type=str)
    
    args = parser.parse_args()
    update_config(cfg, args)

    folder_path = os.path.dirname(args.cfg)
    ckpt_path = os.path.join(folder_path, 'checkpoints', os.path.basename(args.checkpoint))
    result_ply_path = os.path.join(folder_path, os.path.basename(args.checkpoint).split('.')[0] + '_mesh.ply')

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


    """
    Create NeRF Model & Optimizer
    """
    dataset = TUMDatasetFactory().make_dataset(cfg)
    model = NERF(cfg, dataset.camera_info())
    model.cuda()


    """
    Load Checkpoint
    """
    logger.info("Loading model{}".format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict['model'])
       

    """
    Inference
    """
    logger.info("Extracting Mesh...")
    extract_mesh(model, 
                 volume_size = 0.3, 
                 level = 0.0,
                 N = 256,
                 filepath = result_ply_path,
                 show_progress = True,
                 ray_chunk = 16*1024)

    logger.info("Done!")



if __name__ == '__main__':
    main()