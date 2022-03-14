import numpy as np
import torch
import open3d as o3d
from torch.utils import data
from tqdm import tqdm

from ..utils.torch_math import back_project_pixel

import pdb


class ImageRenderingDataset(data.Dataset):
    def __init__(self, color_images, depth_images, positions, camera_info, mode='train'):
        assert color_images.shape[:3] == depth_images.shape
        assert positions.shape == (color_images.shape[0], 4, 4)
        print(f"Read {color_images.shape} images array")

        self._camera_info = camera_info
        # camera_info.update_color_normalization_parameters(color_images)
        self._color_images = camera_info.process_color_image(color_images)
        self._depth_images = camera_info.process_depth_image(depth_images)
        self._positions = camera_info.process_positions(positions)
        self._focal = camera_info.get_focal()

        self._image_count = self._color_images.shape[0]
        self._normals = []
        
        # pdb.set_trace()
        """
        for idx in tqdm(range(self._image_count)):
            if mode == 'train':
                self._normals.append(self.get_normals(idx))
            else:
                self._normals.append(-np.ones_like(self._depth_images[idx]))
        self._normals = np.stack(self._normals, axis=0)
        """
        # pdb.set_trace()
        self._num_frames = len(self._positions)
        print(f"Dataset size: {len(self)} pixels")

    def __len__(self):
        color_image_shape = self._color_images.shape
        return color_image_shape[0] * color_image_shape[1] * color_image_shape[2]

    def __getitem__(self, index):
        image_count, height, width = self._color_images.shape[:3]
        image_index = index // (width * height)
        y = (index % (width * height)) // width
        x = (index % (width * height)) % width
        viewdir = self.get_ray_dir(y, x, self._positions[image_index])
        return {
            "pixel": np.array([x, y], dtype=np.float32),
            "color": self._color_images[image_index, y, x],
            "depth": self._depth_images[image_index, y, x],
            "viewdir": viewdir,
            # "normal": self._normals[image_index, y, x],
            "camera_position": self._positions[image_index]
        }
    
    def get_ray_dir(self, y, x, pose):
        _, height, width = self._color_images.shape[:3]
        
        # pdb.set_trace()
        # y, x = np.meshgrid(range(width), range(height))
        y = np.array([y], dtype=np.float32)
        x = np.array([x], dtype=np.float32)
        # dirs = np.stack([(y + 0.5 - height * 0.5) / self._focal, -(x + 0.5 - width * 0.5) / self._focal, -np.ones_like(y)], -1)
        dirs = np.stack([(x + 0.5 - width * 0.5) / self._focal, (y + 0.5 - height * 0.5) / self._focal, np.ones_like(y)], -1)
        ray_dirs = np.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)
        return ray_dirs[0]

    def val_get_item(self, image_index, scale=2):
        _, height, width = self._color_images.shape[:3]
        
        # pdb.set_trace()
        y, x = np.meshgrid(range(height // scale), range(width // scale))
        pixels = (np.array([x, y], dtype=np.float32).T * scale).reshape(-1, 2)
        pixels_long = (np.array([x, y]).T * scale).reshape(-1, 2)
                                                                                                                     
        positions = torch.repeat_interleave(
            torch.tensor(self._positions[image_index].astype(np.float32))[None], pixels.shape[0], dim=0)
        
        # pdb.set_trace()
        viewdirs = []
        for i in range(pixels.shape[0]):
            viewdir = self.get_ray_dir(pixels_long[i][1], pixels_long[i][0], self._positions[image_index])
            viewdirs.append(viewdir)
        # pdb.set_trace()
        viewdirs = np.stack(viewdirs, 0)

        return{
            "img_idx": image_index,
            "pixels": torch.tensor(pixels),
            "pixels_long": torch.tensor(pixels_long),
            "color": torch.tensor(self._color_images[image_index]),
            "depth": torch.tensor(self._depth_images[image_index]),
            "viewdir": torch.tensor(viewdirs),
            "camera_position": torch.tensor(positions)
        }

    def camera_info(self):
        return self._camera_info

    def num_frames(self):
        return len(self._positions)

    def get_normals(self, index, scale=1):
        y, x = np.meshgrid(range(self._color_images.shape[1] // scale), range(self._color_images.shape[2] // scale))                       
        int_pixels = (np.array([x, y], dtype=np.int32).T * scale).reshape(-1, 2)
        depths = torch.tensor(self._depth_images[index][int_pixels[:, 1], int_pixels[:, 0]]).float()
        pixels = torch.tensor(int_pixels.astype(np.float32))
        position = self._positions[index]
        
        position = torch.repeat_interleave(
            torch.tensor(position.astype(np.float32))[None], depths.shape[0], dim=0)
        inverted_camera_matrix = torch.tensor(
            self._camera_info.get_inverted_camera_matrix().astype(np.float32))

        back_projected_points = \
             back_project_pixel(pixels, depths, position, inverted_camera_matrix).cpu().detach().numpy()
        
        # get normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(back_projected_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        normals = np.asarray(pcd.normals).reshape(self._color_images.shape[1], self._color_images.shape[2], -1)
        return normals