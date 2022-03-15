from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from .camera_info import CameraInfo
from .image_rendering_dataset import ImageRenderingDataset

import pdb
import os

DEFAULT_CAMERA_MATRIX = np.array([[525.0, 0, 319.5],
                                     [0, 525.0, 239.5],
                                     [0, 0, 1.]], dtype=np.float32)


class TUMDatasetFactory(object):
    @staticmethod
    def make_dataset(cfg, camera_matrix=DEFAULT_CAMERA_MATRIX):
        sequence_directory = Path(cfg.dataset.dataset_path) / cfg.dataset.scene_name
        association_file = sequence_directory / cfg.dataset.association_file_name
        print(f"Reading {association_file}")
        
        cfg.defrost()
        associations = pd.read_csv(association_file, names=[i for i in range(12)], sep=' ')
        positions = associations.iloc[:, 5:].values
        if type(cfg.dataset.frame_indices) is list:
            if len(cfg.dataset.frame_indices) == 0:
                total_length = positions.shape[0]
                cfg.dataset.frame_indices = [i for i in range(total_length)]
        else:
            total_length = positions.shape[0]
            if cfg.dataset.frame_indices == 0:
                cfg.dataset.frame_indices = [i for i in range(total_length)]
            else:
                cfg.dataset.frame_indices = [i for i in range(0, total_length, total_length // cfg.dataset.frame_indices)]
        positions = [TUMDatasetFactory.tum_position_to_matrix(positions[i]) for i in cfg.dataset.frame_indices]
        color_image_paths = [str(sequence_directory / associations.iloc[i, 1]) for i in cfg.dataset.frame_indices]
        depth_image_paths = [str(sequence_directory / associations.iloc[i, 3]) for i in cfg.dataset.frame_indices]
        cfg.freeze()

        positions = np.array(positions, dtype=np.float32)
        color_images = np.array([cv2.imread(x).astype(np.float32) for x in color_image_paths])
        depth_images = np.array(
            [cv2.imread(x, cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000. for x in depth_image_paths])
        camera_info = CameraInfo(clip_depth_distance_threshold=cfg.dataset.clip_distance_threshold, camera_matrix=camera_matrix,
                                 distance_koef=cfg.dataset.distance_koef, focal=525.0)
        return ImageRenderingDataset(color_images, depth_images, positions, camera_info, cfg.mode)

    @staticmethod
    def tum_position_to_matrix(tum_position):
        """
        Convert TUM position format to matrix form
        :param tum_position: [tx ty tz qx qy qz qw]
        :return:
        """
        matrix_form = np.eye(4)
        rotation = R.from_quat(tum_position[3:])
        matrix_form[:3, :3] = rotation.as_matrix()
        matrix_form[:3, 3] = tum_position[:3]
        return matrix_form
