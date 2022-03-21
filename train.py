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
    parser.add_argument('--annotation',
                        help='annotation for experiment',
                        default=None,
                        type=str)
    args = parser.parse_args()
    update_config(cfg, args)

    timestamp = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%m-%d-%H-%M")
    selected_frame = str(len(cfg.dataset.frame_indices)) + 'scenes' \
                if type(cfg.dataset.frame_indices) == list else str(cfg.dataset.frame_indices) + 'scenes'
    annotation = '(' \
                + cfg.model.mode + '_' \
                + cfg.model.model_type + '_' \
                + ('2mlp_' if cfg.model.mlp_color else '') \
                + str(cfg.model.coarse_sample_bins) + '_' + str(cfg.model.fine_sample_bins) + 'sampling_' \
                + cfg.model.upsample + '_' \
                + ('encoding_' if cfg.model.pos_encoding else '') \
                + cfg.model.loss_type + '_' \
                + selected_frame + '_' \
                + 'lr{:.0e}'.format(cfg.learning_rate) \
                + ('_' + args.annotation if args.annotation is not None else '') + ')'
    logdir = os.path.join(cfg.logdir, timestamp + annotation)

    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    if not os.path.isdir(os.path.join(logdir, "log_imgs")):
        os.makedirs(os.path.join(logdir, "log_imgs"))

    if not os.path.isdir(os.path.join(logdir, "checkpoints")):
        os.makedirs(os.path.join(logdir, "checkpoints"))

    with open(os.path.join(logdir, "hparams.yaml"), "w") as f:
        f.write(cfg.dump())

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_path = os.path.join(logdir, f'{current_time_str}_{cfg.mode}.log')
    print('creating log file', logfile_path)
    logger.remove()
    logger.add(logfile_path, format="{time} {level} {message}", level="INFO")
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

    tb_writer = SummaryWriter(logdir)
    # save_hparams(tb_writer, cfg)


    """
    Prepare Dataset & Dataloader
    """
    train_dataset = TUMDatasetFactory().make_dataset(cfg)
    train_dataloader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)


    """
    Create NeRF Model & Optimizer
    """
    model = NERF(cfg, train_dataset.camera_info())
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1)


    """
    Load Checkpoint
    """
    start_epoch = 0
    if cfg.resume:
        logger.info("Loading model{}".format(cfg.load_ckpt_path))
        state_dict = torch.load(cfg.load_ckpt_path)
        model.load_state_dict(state_dict["model"], strict=False)
        optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        start_epoch = state_dict['epoch'] + 1

    logger.info("start at epoch {}".format(start_epoch))
    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    """
    Traininng
    """
    for epoch in range(start_epoch, cfg.max_epochs):
        logger.info('Epoch {}:'.format(epoch))

        loss_mean = 0

        for idx, batch in tqdm(enumerate(train_dataloader)):
            model.train()
            torch.cuda.empty_cache()

            global_step = len(train_dataloader) * epoch + idx
            batch = tocuda(batch)

            optimizer.zero_grad()

            # forward
            outputs, losses = model.loss(batch)
            loss = losses["total_loss"]
            loss_mean = (loss_mean * idx + loss.item()) / (idx + 1)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            logger.info('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, lr = {:.5f}'.format(\
                    epoch, cfg.max_epochs, idx, len(train_dataloader), loss.item(),
                    optimizer.param_groups[0]['lr']))

            save_scalars(tb_writer, 'train', losses, global_step)
            # pdb.set_trace()
            save_histograms(tb_writer, 'train', {"sdf" : outputs["fine_sdf"]}, global_step)

            if cfg.model.mode == 'volsdf':
                # ab = {'alpha': model.alpha, 'beta': model._beta}
                ab = {'beta': model._beta}
                save_scalars(tb_writer, 'train', ab, global_step)

            if (idx+1) % cfg.log_img_freq_iter == 0:
                with torch.no_grad():
                    model.eval()
                    for img_idx in tqdm(range(0, train_dataset._num_frames, train_dataset._num_frames // (1 if cfg.num_log_img - 1 == 0 else cfg.num_log_img - 1))):
                        val_batch = train_dataset.val_get_item(img_idx, scale=cfg.log_img_scale)
                        val_batch = tocuda(val_batch)
                        val_batch_size = val_batch["pixels"].shape[0] // cfg.log_img_batch_count
                        output_fine_color = []
                        output_fine_depth = []

                        for i in tqdm(range(cfg.log_img_batch_count)):
                            # pdb.set_trace()
                            val_outputs = model.forward(val_batch["pixels"][i * val_batch_size:i * val_batch_size + val_batch_size],
                                                    val_batch["depth"][val_batch["pixels_long"][i * val_batch_size:i * val_batch_size + val_batch_size][:, 1],\
                                                                       val_batch["pixels_long"][i * val_batch_size:i * val_batch_size + val_batch_size][:, 0]],
                                                    val_batch["camera_position"][i * val_batch_size:i * val_batch_size + val_batch_size],
                                                    val_batch["viewdir"][i * val_batch_size:i * val_batch_size + val_batch_size])
                            output_fine_color.append(val_outputs['fine_color'].detach())
                            output_fine_depth.append(val_outputs['fine_depth'].detach())

                            if i == cfg.log_img_batch_count - 1 and (i+1) * val_batch_size < val_batch["pixels"].shape[0]:
                                val_outputs = model.forward(val_batch["pixels"][(i+1) * val_batch_size:],
                                                            val_batch["depth"][val_batch["pixels_long"][(i+1) * val_batch_size:][:, 1],\
                                                                               val_batch["pixels_long"][(i+1) * val_batch_size:][:, 0]],
                                                            val_batch["camera_position"][(i+1) * val_batch_size:],
                                                            val_batch["viewdir"][(i+1) * val_batch_size:])
                                output_fine_color.append(val_outputs['fine_color'].detach())
                                output_fine_depth.append(val_outputs['fine_depth'].detach())

                            del val_outputs

                        reconstructed_img = torch.cat(output_fine_color, dim=0).reshape(
                                                        val_batch["color"].shape[0] // cfg.log_img_scale,
                                                        val_batch["color"].shape[1] // cfg.log_img_scale, 3).detach().cpu().numpy()
                        mean = train_dataset.camera_info()._color_mean
                        std = train_dataset.camera_info()._color_std
                        reconstructed_img = reconstructed_img * std[None, None] + mean[None, None]
                        reconstructed_img = np.clip(reconstructed_img / 255., 0, 1)
                        reconstructed_depth = torch.cat(output_fine_depth, dim=0).reshape(
                                                        val_batch["color"].shape[0] // cfg.log_img_scale,
                                                        val_batch["color"].shape[1] // cfg.log_img_scale).detach().cpu().numpy()

                        img_gt = val_batch["color"].detach().cpu().numpy() * std[None, None] + mean[None, None]
                        img_gt = np.clip(img_gt / 255., 0, 1)
                        depth_gt = val_batch["depth"].detach().cpu().numpy()

                        # pdb.set_trace()
                        prev_rgb = cv2.hconcat([cv2.resize(reconstructed_img, (img_gt.shape[1], img_gt.shape[0])), img_gt])
                        # prev_depth = cv2.applyColorMap((cv2.hconcat([cv2.resize(reconstructed_depth, (depth_gt.shape[1], depth_gt.shape[0])), depth_gt]) \
                        #                                     / train_dataset.camera_info().get_default_depth() * 255).astype(np.uint8),
                        #                                cv2.COLORMAP_JET).astype(np.float32) / 255.
                        prev_depth = cv2.applyColorMap((cv2.hconcat([cv2.resize(reconstructed_depth, (depth_gt.shape[1], depth_gt.shape[0])), depth_gt]) \
                                                            / depth_gt.max() * 255).astype(np.uint8),
                                                       cv2.COLORMAP_JET).astype(np.float32) / 255.
                        prev_output = cv2.vconcat([prev_rgb, prev_depth])

                        cv2.imwrite(os.path.join(logdir, "log_imgs", "log_img_{:04d}_{:04d}_{:04d}.jpg".format(epoch, idx, img_idx)),
                                        prev_output * 255)
                        prev_output = cv2.cvtColor(prev_output, cv2.COLOR_BGR2RGB)
                        save_images(tb_writer, 'train', {"log_img_{}".format(img_idx): torch.from_numpy(prev_output).permute(2,0,1)}, global_step)

                        del val_batch

            del batch
            del outputs
            del losses

        scheduler.step(loss_mean)

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            "{}/model_{:0>4}.ckpt".format(os.path.join(logdir, "checkpoints"), epoch)
        )


if __name__ == '__main__':
    main()