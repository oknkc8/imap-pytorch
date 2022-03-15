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

    # ready for all frame
    cfg.defrost()
    cfg.dataset.frame_indices = 0
    cfg.log_img_scale = 2
    cfg.log_img_batch_count = 50
    cfg.freeze()

    folder_path = os.path.dirname(args.cfg)
    ckpt_path = os.path.join(folder_path, 'checkpoints', os.path.basename(args.checkpoint))
    result_folder_path = os.path.join(folder_path, os.path.basename(args.checkpoint).split('.')[0] + '_video')

    if not os.path.isdir(result_folder_path):
        os.makedirs(result_folder_path)

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


    """
    Prepare Dataset
    """
    test_dataset = TUMDatasetFactory().make_dataset(cfg)


    """
    Create NeRF Model & Optimizer
    """
    model = NERF(cfg, test_dataset.camera_info())
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
    logger.info("Inference...")
    img_outputs = []
    with torch.no_grad():
        model.eval()
        for img_idx in tqdm(range(0, test_dataset._num_frames)):
            val_batch = test_dataset.val_get_item(img_idx, scale=cfg.log_img_scale)
            # val_batch = tocuda(val_batch)
            val_batch_size = val_batch["pixels"].shape[0] // cfg.log_img_batch_count
            output_fine_color = []
            output_fine_depth = []

            for i in tqdm(range(cfg.log_img_batch_count)):
                val_outputs = model.forward(tocuda(val_batch["pixels"][i * val_batch_size:i * val_batch_size + val_batch_size]),
                                            tocuda(val_batch["depth"][val_batch["pixels_long"][i * val_batch_size:i * val_batch_size + val_batch_size][:, 1],\
                                                                val_batch["pixels_long"][i * val_batch_size:i * val_batch_size + val_batch_size][:, 0]]),
                                            tocuda(val_batch["camera_position"][i * val_batch_size:i * val_batch_size + val_batch_size]),
                                            tocuda(val_batch["viewdir"][i * val_batch_size:i * val_batch_size + val_batch_size]))
                output_fine_color.append(val_outputs['fine_color'].detach())
                output_fine_depth.append(val_outputs['fine_depth'].detach())

                if i == cfg.log_img_batch_count - 1 and (i+1) * val_batch_size < val_batch["pixels"].shape[0]:
                    val_outputs = model.forward(tocuda(val_batch["pixels"][(i+1) * val_batch_size:]),
                                                tocuda(val_batch["depth"][val_batch["pixels_long"][(i+1) * val_batch_size:][:, 1],\
                                                                    val_batch["pixels_long"][(i+1) * val_batch_size:][:, 0]]),
                                                tocuda(val_batch["camera_position"][(i+1) * val_batch_size:]),
                                                tocuda(val_batch["viewdir"][(i+1) * val_batch_size:]))
                    output_fine_color.append(val_outputs['fine_color'].detach())
                    output_fine_depth.append(val_outputs['fine_depth'].detach())

                del val_outputs

            reconstructed_img = torch.cat(output_fine_color, dim=0).reshape(
                                            val_batch["color"].shape[0] // cfg.log_img_scale,
                                            val_batch["color"].shape[1] // cfg.log_img_scale, 3).detach().cpu().numpy()
            mean = test_dataset.camera_info()._color_mean
            std = test_dataset.camera_info()._color_std
            reconstructed_img = reconstructed_img * std[None, None] + mean[None, None]
            reconstructed_img = np.clip(reconstructed_img / 255., 0, 1)
            reconstructed_depth = torch.cat(output_fine_depth, dim=0).reshape(
                                            val_batch["color"].shape[0] // cfg.log_img_scale,
                                            val_batch["color"].shape[1] // cfg.log_img_scale).detach().cpu().numpy()

            img_gt = val_batch["color"].detach().cpu().numpy() * std[None, None] + mean[None, None]
            img_gt = np.clip(img_gt / 255., 0, 1)
            depth_gt = val_batch["depth"].detach().cpu().numpy()

            prev_rgb = cv2.hconcat([cv2.resize(reconstructed_img, (img_gt.shape[1], img_gt.shape[0])), img_gt])
            prev_depth = cv2.applyColorMap((cv2.hconcat([cv2.resize(reconstructed_depth, (depth_gt.shape[1], depth_gt.shape[0])), depth_gt]) \
                                                / depth_gt.max() * 255).astype(np.uint8),
                                            cv2.COLORMAP_JET).astype(np.float32) / 255.
            prev_output = cv2.vconcat([prev_rgb, prev_depth])

            cv2.imwrite(os.path.join(result_folder_path, "result_img_{:04d}.jpg".format(img_idx)),
                            prev_output * 255)
            img_outputs.append(prev_output * 255)

            del val_batch
        
    
    """
    Make Video
    """
    print()
    logger.info("Make Video...")
    size = img_outputs[0].size()
    out = cv2.VideoWriter(os.path.join(result_folder_path, "result_video.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for img in tqdm(img_outputs):
        out.write(img)
    out.release()

    logger.info("Done!")



if __name__ == '__main__':
    main()