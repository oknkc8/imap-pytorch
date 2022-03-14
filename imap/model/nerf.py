import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad

from .base_lightning_model import BaseLightningModule
from .gaussian_positional_encoding import GaussianPositionalEncoding
from .mlp import MLP, SingleBVPNet
from ..utils.torch_math import back_project_pixel, matrix_from_9d_position

import pdb

class NERF(BaseLightningModule):
    def __init__(self, parameters, camera_info):
        super().__init__(parameters)
        self.save_hyperparameters()
        input_dimension = parameters.encoding_dimension if parameters.pos_encoding else 3
        # self._mlp = MLP(parameters.encoding_dimension, 4)
        if parameters.model_type == 'relu':
            self._mlp = MLP(input_dimension, 4)
        elif parameters.model_type == 'sine':
            self._mlp = SingleBVPNet(type='sine', 
                                     in_features=input_dimension,
                                     out_features=4,
                                     num_hidden_layers=4)
        
        self._camera_info = camera_info
        self._positional_encoding = GaussianPositionalEncoding(encoding_dimension=parameters.encoding_dimension,
                                                               sigma=parameters.encoding_sigma)
        self._inverted_camera_matrix = torch.tensor(self._camera_info.get_inverted_camera_matrix())
        self._default_color = torch.tensor(self._camera_info.get_default_color())
        self._default_depth = torch.tensor(self._camera_info.get_default_depth())
        self._loss = nn.L1Loss(reduction="none")
        self._mseloss = nn.MSELoss(reduction="none")
        self._positions = None
        self._truncation = parameters.truncation
        self._sc_factor = parameters.sc_factor
        self._type = parameters.model_type
        self._mode = parameters.mode
        self._pos_encoding = parameters.pos_encoding

    def forward(self, pixel, depth, camera_position):
        with torch.no_grad():
            coarse_sampled_depths = self.stratified_sample_depths(
                pixel.shape[0],
                pixel.device,
                self.hparams.course_sample_bins,
                not self.training)
        coarse_sdf, coarse_coord, coarse_color, coarse_depths, coarse_weights, coarse_depth_variance = self.reconstruct_color_and_depths(
            coarse_sampled_depths,
            pixel,
            camera_position,
            self._mlp)
        with torch.no_grad():
            fine_sampled_depths = self.hierarchical_sample_depths(
                coarse_weights,
                pixel.shape[0],
                pixel.device,
                self.hparams.fine_sample_bins,
                coarse_sampled_depths,
                not self.training)
        fine_sampled_depths = torch.cat([fine_sampled_depths, coarse_sampled_depths], dim=0)
        fine_sdf, fine_coord, fine_color, fine_depths, fine_weights, fine_depth_variance = self.reconstruct_color_and_depths(
            fine_sampled_depths,
            pixel,
            camera_position,
            self._mlp)

        return {'coarse_color': coarse_color,
                'coarse_depth': coarse_depths,
                'fine_color': fine_color,
                'fine_depth': fine_depths,
                'coarse_depth_variance': coarse_depth_variance,
                'fine_depth_variance': fine_depth_variance,
                'coarse_sdf': coarse_sdf,
                'fine_sdf': fine_sdf,
                'coarse_coord': coarse_coord,
                'fine_coord': fine_coord,
                'fine_sampled_depths': fine_sampled_depths,
                }

    def reconstruct_color_and_depths(self, sampled_depths, pixels, camera_positions, mlp_model):
        bins_count = sampled_depths.shape[0]
        sampled_depths = torch.sort(sampled_depths, dim=0).values
        sampled_depths = sampled_depths.reshape(-1)
        pixels = self.repeat_tensor(pixels, bins_count)
        camera_positions = self.repeat_tensor(camera_positions, bins_count)
        back_projected_points = back_project_pixel(pixels, sampled_depths, camera_positions,
                                                   self._inverted_camera_matrix).requires_grad_(True)
        if self._pos_encoding:
            encodings = self._positional_encoding(back_projected_points)
            prediction = mlp_model(encodings)
        else:
            prediction = mlp_model(back_projected_points)

        colors = torch.sigmoid(prediction[:, :3]).reshape(bins_count, -1, 3)
        density = prediction[:, 3].reshape(bins_count, -1) # density == sdf
        depths = sampled_depths.reshape(bins_count, -1)
        
        if self._mode == 'density':
            weights = self.calculate_weights(density, depths)
        elif self._mode == 'sdf':
            weights = self.calculate_weights_from_sdf(density, depths)

        reconstructed_color = self.reconstruct_color(colors, weights, self._default_color, self._type)
        reconstructed_depths = self.reconstruct_depth(depths, weights, self._default_depth, self._type)
        with torch.no_grad():
            reconstructed_depth_variance = self.reconstruct_depth_variance(depths, weights, reconstructed_depths,
                                                                           self._default_depth, self._type)

        density = prediction[:, 3:]

        return density, back_projected_points, \
               reconstructed_color, reconstructed_depths, weights, reconstructed_depth_variance

    def stratified_sample_depths(self, batch_size, device, bins_count, deterministic=False):
        if deterministic:
            depth_delta = (self._default_depth.item() - self.hparams.minimal_depth) / bins_count
            result = torch.arange(self.hparams.minimal_depth, self._default_depth.item(), depth_delta, device=device)
            result = torch.repeat_interleave(result[:, None], batch_size, dim=1)
            return result
        uniform = torch.rand((bins_count, batch_size), device=device)
        uniform[0] = 1
        result = (torch.arange(bins_count, device=device)[:, None] + uniform - 1
                  ) * (self._default_depth - self.hparams.minimal_depth) / (bins_count - 1) + self.hparams.minimal_depth
        return result

    def hierarchical_sample_depths(self, weights, batch_size, device, bins_count, bins, deterministic=False):
        weights = weights.transpose(1, 0)[:, :-1] + 1e-10
        pdf = weights / torch.sum(weights, dim=1)[:, None]
        cdf = torch.cumsum(pdf, dim=1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], 1)
        minimal_bin = bins[0]
        bins = (torch.roll(bins, 1) + bins) / 2
        bins[0] = minimal_bin
        bins = bins.transpose(1, 0)

        if deterministic:
            uniform = torch.arange(bins_count, device=device) / bins_count + 1. / 2 / bins_count
            uniform = torch.repeat_interleave(uniform[None], batch_size, dim=0)
        else:
            uniform = torch.rand((batch_size, bins_count), device=device).contiguous()
        indexes = torch.searchsorted(cdf, uniform, right=True)
        index_below = self.clip_indexes(indexes - 1, 0, bins.shape[1] - 1)
        index_above = self.clip_indexes(indexes, 0, bins.shape[1] - 1)

        denominator = torch.gather(cdf, 1, index_above) - torch.gather(cdf, 1, index_below)
        denominator = torch.where(denominator < 1e-10, torch.ones_like(denominator), denominator)
        t = (uniform - torch.gather(cdf, 1, index_below)) / denominator
        bins_below = torch.gather(bins, 1, index_below)
        bins_above = torch.gather(bins, 1, index_above)
        hierarchical_sample = bins_below + t * (bins_above - bins_below)
        return hierarchical_sample.transpose(1, 0)

    @staticmethod
    def clip_indexes(indexes, minimal, maximal):
        result = torch.max(minimal * torch.ones_like(indexes), indexes)
        result = torch.min(maximal * torch.ones_like(indexes), result)
        return result

    @staticmethod
    def repeat_tensor(tensor, bins_count):
        result = torch.repeat_interleave(tensor[None], bins_count, dim=0)
        result = result.reshape(-1, *tensor.shape[1:])
        return result

    def calculate_weights(self, densities, depths):
        weights = []
        product = 1
        densities = torch.logsumexp(torch.cat([torch.zeros_like(densities)[None], densities[None]], dim=0), dim=0)
        for i in range(len(depths)):
            if i < len(depths) - 1:
                depth_delta = depths[i + 1] - depths[i]
            else:
                depth_delta = self._default_depth - depths[i]
            hit_probability = 1 - torch.exp(-densities[i] * depth_delta)
            weights.append(hit_probability * product)
            product = product * (1 - hit_probability)
        weights.append(product)
        return torch.stack(weights, dim=0)

    def calculate_weights_from_sdf(self, sdf, depths):
        weights = F.sigmoid(sdf / self._truncation) * F.sigmoid(-sdf / self._truncation)

        signs = sdf[1:, :] * sdf[:-1, :]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        indices = torch.argmax(mask, axis=0)
        indices = torch.unsqueeze(indices, 0)
        depth_min = torch.gather(depths, 0, indices)
        mask = torch.where(depths < depth_min + self._sc_factor * self._truncation, torch.ones_like(depths), torch.zeros_like(depths))

        weights = weights * mask
        weights = weights / (torch.sum(weights, dim=0, keepdim=True) + 1e-8)

        return weights


    @staticmethod
    def reconstruct_color(colors, weights, default_color, type):
        if type == 'density':
            return torch.sum(colors * weights[:-1, :, None], dim=0
                            ) + default_color.to(colors.device)[None] * weights[-1, :, None]
        else:
            return torch.sum(colors * weights[:, :, None], dim=0)

    @staticmethod
    def reconstruct_depth(depths, weights, default_depth, type):
        if type == 'density':
            return torch.sum(depths * weights[:-1, :], dim=0) + default_depth.to(depths.device)[None] * weights[-1]
        else:
            return torch.sum(depths * weights[:, :], dim=0)

    @staticmethod
    def reconstruct_depth_variance(depths, weights, mean_depths, default_depth, type):
        if type == 'density':
            return torch.sum((depths - mean_depths[None]) ** 2 * weights[:-1], dim=0
                            ) + (default_depth.to(depths.device)[None] - mean_depths) ** 2 * weights[-1]
        else:
            return torch.sum((depths - mean_depths[None]) ** 2 * weights, dim=0)

    def loss(self, batch, reduction=True):
        camera_position = self.positions_from_batch(batch)
        output = self.forward(batch["pixel"], batch["depth"], camera_position)

        # pdb.set_trace()
        
        mask = (batch["depth"] > 1e-12) & (batch["depth"] < self._default_depth)
        valid_weight = batch["depth"].nelement() / (mask.count_nonzero() + 1e-18)

        if self.hparams.loss_type == 'l1':
            coarse_image_loss = torch.mean(self._loss(output['coarse_color'], batch["color"]), dim=1)
            coarse_depth_weights = 1. / (torch.sqrt(output['coarse_depth_variance']) + 1e-18) * mask
            coarse_depth_loss = self._loss(output['coarse_depth'] * coarse_depth_weights, batch["depth"] * coarse_depth_weights)

            fine_image_loss = torch.mean(self._loss(output['fine_color'], batch["color"]), dim=1)
            fine_depth_weights = 1. / (torch.sqrt(output['fine_depth_variance']) + 1e-18) * mask
            fine_depth_loss = self._loss(output['fine_depth'] * fine_depth_weights, batch["depth"] * fine_depth_weights)
        elif self.hparams.loss_type == 'l2':
            coarse_image_loss = torch.mean(self._mseloss(output['coarse_color'], batch["color"]), dim=1)
            coarse_depth_loss = self._mseloss(output['coarse_depth'], batch["depth"]) * valid_weight

            fine_image_loss = torch.mean(self._mseloss(output['fine_color'], batch["color"]), dim=1)
            fine_depth_loss = self._mseloss(output['fine_depth'], batch["depth"]) * valid_weight

        if self._mode == 'sdf':
            fine_sampled_depths = output['fine_sampled_depths'].transpose(0, 1)
            batch_size = fine_sampled_depths.shape[0]
            fine_sdf = output['fine_sdf'].reshape(-1, batch_size).transpose(0, 1)
            target_depth = batch["depth"].unsqueeze(-1)
            freespace_constraint, sdf_constraint = self.get_sdf_loss(fine_sdf, fine_sampled_depths, target_depth)

            sdf_loss = freespace_constraint * self.hparams.fs_loss_koef \
                     + sdf_constraint * self.hparams.sdf_loss_koef \

        else:
            sdf_constraint = freespace_constraint = grad_constraint = 0.0
            sdf_loss = 0.0


        # image_loss = coarse_image_loss + fine_image_loss
        # depth_loss = coarse_depth_loss + fine_depth_loss
        image_loss = fine_image_loss
        depth_loss = fine_depth_loss

        loss = self.hparams.color_loss_koef * image_loss + self.hparams.depth_loss_koef * depth_loss

        if reduction:
            coarse_depth_loss = torch.mean(coarse_depth_loss)
            coarse_image_loss = torch.mean(coarse_image_loss)
            fine_depth_loss = torch.mean(fine_depth_loss)
            fine_image_loss = torch.mean(fine_image_loss)
            loss = torch.mean(loss)

        loss = loss + sdf_loss

        losses = {
            "course_image_loss": coarse_image_loss,
            "course_depth_loss": coarse_depth_loss,
            "fine_image_loss": fine_image_loss,
            "fine_depth_loss": fine_depth_loss,
            "sdf_constraint": sdf_constraint,
            "freespace_constraint": freespace_constraint,
            "loss": loss
        }
        return output, losses

    def positions_from_batch(self, batch):
        if "camera_position" in batch.keys():
            return batch["camera_position"]
        indexes = batch["frame_index"]
        return matrix_from_9d_position(self._positions[indexes])

    def set_positions(self, position):
        self._positions = position
    
    def gradient(self, y, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        return grad

    def get_sdf_loss(self, sdf, sampled_depths, target_depth):
        front_mask, sdf_mask, fs_weight, sdf_weight = self.get_masks(sampled_depths, target_depth)

        fs_loss = self._mseloss(sdf * front_mask, torch.ones_like(sdf) * front_mask) * fs_weight
        sdf_loss = self._mseloss((sampled_depths + sdf * self._truncation) * sdf_mask, target_depth * sdf_mask) * sdf_weight

        return fs_loss.mean(), sdf_loss.mean()
    
    def get_masks(self, sampled_depths, target_depths):
        front_mask = torch.where(sampled_depths < (target_depths - self._truncation), torch.ones_like(sampled_depths), torch.zeros_like(sampled_depths))
        back_mask = torch.where(sampled_depths > (target_depths + self._truncation), torch.ones_like(sampled_depths), torch.zeros_like(sampled_depths))
        depth_mask = torch.where(target_depths > 0.0, torch.ones_like(target_depths), torch.zeros_like(target_depths))
        sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

        num_fs_points = torch.count_nonzero(front_mask)
        num_sdf_points = torch.count_nonzero(sdf_mask)
        fs_weight = 1.0 - (num_fs_points / (num_fs_points + num_sdf_points))
        sdf_weight = 1.0 - (num_sdf_points / (num_fs_points + num_sdf_points))

        return front_mask, sdf_mask, fs_weight, sdf_weight
