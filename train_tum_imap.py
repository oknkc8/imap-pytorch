import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PYTHON_PATHS = [".."]
import sys
for path in PYTHON_PATHS:
    if path not in sys.path:
        sys.path.append(path)
        
from imap.data.image_rendering_data_module import ImageRenderingDataModule
from imap.model.nerf import NERF
from imap.utils import UniversalFactory
from pytorch_lightning.utilities.parsing import AttributeDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# import os
from pathlib import Path
import torch
from tqdm.notebook import tqdm
from scipy.spatial.transform import Rotation

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from imap.utils.torch_math import *
import numpy as np

import pdb

color_image_path = str(Path("test_datasets/tum rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031458.159638.png"))
depth_image_path = str(Path("test_datasets/tum rgbd/rgbd_dataset_freiburg1_desk/depth/1305031458.144808.png"))
color_image = cv2.imread(color_image_path)
depth_image = cv2.imread(depth_image_path, -1).astype(np.float32) / 5000
depth_image = np.clip(depth_image, 0, 4)

dataset_params = {'dataset_path': "test_datasets/tum rgbd/", 
                  'scene_name': "rgbd_dataset_freiburg1_desk", 
                  'association_file_name': "data_association_file.txt", 
                #   'frame_indices': 5,
                #   'frame_indices': [23, 60, 131, 206, 257, 325, 368, 407, 455, 558],
                #   'frame_indices': [131, 325],
                  'frame_indices': [325],
                  'distance_koef': 0.1,
                  'clip_distance_threshold': 4.,
                  'batch_size': 4096,
                  }
# dataset_params=1
data_module = ImageRenderingDataModule('tum', **dataset_params)

parameters = AttributeDict(
    name="NERF",
    # optimizer=AttributeDict(lr=5e-3),
    optimizer=AttributeDict(lr=1e-4, betas="0.9 0.999"),
    scheduler=AttributeDict(mode='min', factor=0.5, patience=1),
    encoding_dimension=93,
    course_sample_bins=320,
    fine_sample_bins=16,
    depth_loss_koef=1,
    color_loss_koef=5.,
    encoding_sigma=25,
    optimize_positions=False,
    minimal_depth=0.01,
    truncation=0.05,
    sc_factor=1,
    # mode='density',
    mode='sdf',
    # model_type='relu',
    model_type='sine',
    pos_encoding=True,
    sdf_loss_koef=6e4,
    fs_loss_koef=1e2,
    loss_type='l2',
)
factory = UniversalFactory([NERF])
model = factory.make_from_parameters(parameters, camera_info=data_module.camera_info())

logger_path = os.path.join("logs")
trainer_parameters = {
    "max_epochs": 10,
    "checkpoint_every_n_val_epochs": 1,
    "gpus": 1,
    # "check_val_every_n_epoch": 2,
    "log_every_n_steps": 1,
    "gradient_clip_val": 1.,
    "gradient_clip_algorithm": "norm",
    "default_root_dir": logger_path
}
# task.connect(trainer_parameters)
# model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss',
#     every_n_val_epochs=trainer_parameters["checkpoint_every_n_val_epochs"])
trainer = factory.kwargs_function(pl.Trainer)(
    logger=TensorBoardLogger(logger_path, name="tmp_sdf_sine_color_pos_encoding_color5_depth1_sdf6e4_fs1e2_l2_lr1e-4_single_scene", log_graph=True),
    callbacks=[],
    **trainer_parameters
)

trainer.fit(model, data_module)

torch.save(model.state_dict(), \
    os.path.join(logger_path, "tmp_sdf_sine_color_pos_encoding_color5_depth1_sdf6e4_fs1e2_l2_lr1e-4_single_scene", \
                              "tmp_sdf_sine_color_pos_encoding_color5_depth1_sdf6e4_fs1e2_l2_lr1e-4_single_scene.pth"))



# model.load_state_dict(torch.load("test_l1.pth"))





# test_dataset_params = {'dataset_path': "test_datasets/tum rgbd/", 
#                   'scene_name': "rgbd_dataset_freiburg1_desk", 
#                   'association_file_name': "data_association_file.txt", 
#                   'frame_indices' : [325],
# #                   'frame_indices': [7, 100, 183, 195, 303, 386],
# #                   'frame_indices': [131, 325],
# #                   'frame_indices': [325],
#                   'distance_koef': 0.1,
#                   'clip_distance_threshold': 4.,
#                   'mode': 'test'}
# # dataset_params=1
# test_data_module = ImageRenderingDataModule('tum', **test_dataset_params)

# parameters = AttributeDict(
#     name="NERF",
#     optimizer=AttributeDict(lr=1e-4),
#     encoding_dimension=93,
#     course_sample_bins=32,
#     fine_sample_bins=12,
#     depth_loss_koef=1.,
#     color_loss_koef=5.,
#     encoding_sigma=25,
#     optimize_positions=False,
#     minimal_depth=0.01,
#     truncation=0.05,
#     sc_factor=1,
#     mode='sdf',
#     model_type='sine',
#     pos_encoding=False,
#     sdf_loss_koef=0.01,
# )
# factory = UniversalFactory([NERF])
# model = factory.make_from_parameters(parameters, camera_info=test_data_module.camera_info())


# def get_position(translation, rotation):
#     matrix_position = np.eye(4)
#     matrix_position[:3, 3] = translation
#     matrix_position[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
#     return matrix_position

# def render_small(color_mean, color_std, img_gt, depth_gt, img, depth):
#     img_gt = img_gt * color_std[None, None] + color_mean[None, None]
#     img_gt = np.clip(img_gt / 255, 0, 1)
    
#     # depth_gt = 1. / depth_gt

#     # depth = 1. / depth

#     prev_rgb = cv2.hconcat([cv2.resize(img, (img_gt.shape[1], img_gt.shape[0])), img_gt])
#     prev_depth = cv2.cvtColor(cv2.hconcat([cv2.resize(depth, (depth_gt.shape[1], depth_gt.shape[0])), depth_gt]), cv2.COLOR_GRAY2RGB)
#     prev = cv2.vconcat([prev_rgb, prev_depth])
    
#     cv2.imshow("demo", prev)
#     cv2.waitKey(1000)



# scale  = 3
# y, x = np.meshgrid(range(color_image.shape[0] // scale), range(color_image.shape[1] // scale))                       
# pixels = np.array([x, y], dtype=np.float32).T * scale
# pixels = torch.tensor(pixels).cuda().reshape(-1, 2)

# pixels_long = np.array([x, y]).T * scale
# pixels_long = torch.tensor(pixels_long).cuda().reshape(-1, 2)

# model = model.cuda()
# model.eval()

# num_frames = test_data_module._dataset._num_frames
# for idx in range(num_frames):
#   position = test_data_module._dataset._positions[idx]
#   img_gt = test_data_module._dataset._color_images[idx]
#   depth_gt = test_data_module._dataset._depth_images[idx]
#   depths = torch.tensor(test_data_module._dataset._depth_images[idx]).cuda()

#   positions = torch.repeat_interleave(torch.tensor(position.astype(np.float32))[None], pixels.shape[0],
#                                   dim=0).cuda()

#   batch_size = pixels.shape[0]
#   batch_count = pixels.shape[0] // batch_size
#   output_coarse_color = []
#   output_coarse_depth = []
#   output_fine_color = []
#   output_fine_depth = []
#   with torch.no_grad():
#       for i in range(batch_count):
#         #   pdb.set_trace()
#           x = pixels_long[i * batch_size:i * batch_size + batch_size][:,0]
#           y = pixels_long[i * batch_size:i * batch_size + batch_size][:,1]
#           output = model(pixels[i * batch_size:i * batch_size + batch_size],
#                          depths[y,x],
#                          positions[i * batch_size:i * batch_size + batch_size])
#           output_coarse_color.append(output['coarse_color'])
#           output_coarse_depth.append(output['coarse_depth'])
#           output_fine_color.append(output['fine_color'])
#           output_fine_depth.append(output['fine_depth'])

#   reconstructed_image = torch.cat(output_fine_color, dim=0).reshape(color_image.shape[0] // scale, color_image.shape[1] // scale, 3).detach().cpu().numpy()
#   mean = data_module.camera_info()._color_mean
#   std = data_module.camera_info()._color_std
#   reconstructed_image = reconstructed_image * std[None, None] + mean[None, None]
#   reconstructed_image = np.clip(reconstructed_image / 255., 0, 1)
#   # figure = plt.figure(dpi=200)
#   # plt.imshow(cv2.cvtColor(reconstructed_image.astype(np.float32), cv2.COLOR_RGB2BGR))

#   reconstructed_depth = torch.cat(output_fine_depth, dim=0).reshape(color_image.shape[0] // scale, color_image.shape[1] // scale).detach().cpu().numpy()

#   render_small(mean, std, img_gt, depth_gt, reconstructed_image, reconstructed_depth)
#   # figure = plt.figure(dpi=200)
#   # plt.imshow(1. /reconstructed_depth)
