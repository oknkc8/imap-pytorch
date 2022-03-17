from os import spawnlp
from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .gaussian_positional_encoding import GaussianPositionalEncoding
from .mlp import MLP, SingleBVPNet
from ..utils.torch_math import back_project_pixel, matrix_from_9d_position

import pdb

class NERF(nn.Module):
    def __init__(self, cfg, camera_info):
        super().__init__()
        self._cfg = cfg

        input_dimension = cfg.model.encoding_dimension if cfg.model.pos_encoding else 3
        if cfg.model.model_type == 'relu':
            if not cfg.model.mlp_color:
                self._mlp = MLP(input_dimension, 4,
                                hidden_features=cfg.model.hidden_features,
                                num_hidden_layers=cfg.model.mlp_num_layers)
                self._mlp_color = None
            else:
                self._mlp = MLP(input_dimension, 1,
                                hidden_features=cfg.model.hidden_features,
                                num_hidden_layers=cfg.model.mlp_num_layers,
                                last_ac='tanh' if cfg.model.mode == 'sdf' else None)
                self._mlp_color = MLP(input_dimension + self._mlp.last_hidden_dim, 3,
                                      hidden_features=cfg.model.hidden_features,
                                      num_hidden_layers=cfg.model.mlp_color_num_layers)
        elif cfg.model.model_type == 'sine':
            if not cfg.model.mlp_color:
                self._mlp = SingleBVPNet(type='sine',
                                        in_features=input_dimension,
                                        out_features=4,
                                        hidden_features=cfg.model.hidden_features,
                                        num_hidden_layers=cfg.model.mlp_num_layers)
                self._mlp_color = None
            else:
                self._mlp = SingleBVPNet(type='sine',
                                        in_features=input_dimension,
                                        out_features=1,
                                        hidden_features=cfg.model.hidden_features,
                                        num_hidden_layers=cfg.model.mlp_num_layers)
                self._mlp_color = SingleBVPNet(type='sine',
                                               in_features=input_dimension + self._mlp.last_hidden_dim,
                                               out_features=3,
                                               hidden_features=cfg.model.hidden_features,
                                               num_hidden_layers=cfg.model.mlp_color_num_layers)

        self._camera_info = camera_info
        self._positional_encoding = GaussianPositionalEncoding(in_features=3,
                                                               encoding_dimension=cfg.model.encoding_dimension,
                                                               sigma=cfg.model.encoding_sigma,
                                                               use_only_sin=False)
        self._positional_encoding_dir = GaussianPositionalEncoding(in_features=3,
                                                                   encoding_dimension=cfg.model.encoding_dimension,
                                                                   sigma=cfg.model.encoding_sigma,
                                                                   use_only_sin=False)
        self._inverted_camera_matrix = torch.tensor(self._camera_info.get_inverted_camera_matrix())
        self._default_color = torch.tensor(self._camera_info.get_default_color())
        self._default_depth = torch.tensor(self._camera_info.get_default_depth())
        self._loss = nn.L1Loss(reduction="none")
        self._mseloss = nn.MSELoss(reduction="none")
        self._positions = None
        self._truncation = cfg.model.truncation * cfg.model.sc_factor
        # self._sc_factor = cfg.model.sc_factor
        self._mode = cfg.model.mode
        self._pos_encoding = cfg.model.pos_encoding

        # self._alpha = nn.Parameter(data=torch.Tensor([cfg.model.alpha]), requires_grad=True)
        self._beta = nn.Parameter(data=torch.Tensor([cfg.model.beta]), requires_grad=True)
        # self._alpha = (1. / self._beta).cuda()

        self.ln_s = nn.Parameter(data=torch.Tensor([-np.log(self._cfg.model.variance_init) / self._cfg.model.speed_factor]), requires_grad=True)
        self.speed_factor = self._cfg.model.speed_factor

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)

    def forward(self, pixel, target_depths, camera_position, viewdir):
        with torch.no_grad():
            coarse_sampled_depths = self.stratified_sample_depths(
                pixel.shape[0],
                pixel.device,
                self._cfg.model.coarse_sample_bins,
                not self.training)
        if self._cfg.model.upsample == 'default':
            coarse_sdf, coarse_coord, coarse_color, coarse_depths, coarse_weights, coarse_depth_variance = self.reconstruct_color_and_depths(
                target_depths,
                coarse_sampled_depths,
                pixel,
                camera_position,
                viewdir,
                self._mlp,
                self._mlp_color)
        with torch.no_grad():
            if self._cfg.model.upsample == 'default':
                fine_sampled_depths = self.hierarchical_sample_depths(
                    coarse_weights,
                    pixel.shape[0],
                    pixel.device,
                    self._cfg.model.fine_sample_bins,
                    coarse_sampled_depths,
                    not self.training)
                fine_sampled_depths = torch.cat([fine_sampled_depths, coarse_sampled_depths], dim=0)

            elif self._cfg.model.upsample == 'neus':
                n_upsample_iters = 4
                # coarse_sdf = coarse_sdf.reshape(self._cfg.model.coarse_sample_bins, -1)
                coarse_sdf = self.get_sdf_value(coarse_sampled_depths, pixel, camera_position, self._mlp)
                coarse_sdf = coarse_sdf.reshape(self._cfg.model.coarse_sample_bins, -1)
                sampled_depths = coarse_sampled_depths
                for i in range(n_upsample_iters):
                    prev_sdf, next_sdf = coarse_sdf[:-1, ...], coarse_sdf[1:, ...]
                    prev_sampled_depths, next_sampled_depths = sampled_depths[:-1, ...], sampled_depths[1:, ...]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (next_sampled_depths - prev_sampled_depths + 1e-5)
                    prev_dot_val = torch.cat([torch.zeros_like(dot_val[:1, ...]).cuda(), dot_val[:-1, ...]], dim=0)
                    dot_val = torch.stack([prev_dot_val, dot_val], dim=0)
                    dot_val, _ = torch.min(dot_val, dim=0, keepdim=False)
                    dot_val = dot_val.clamp(-10.0, 0.0)

                    dist = (next_sampled_depths - prev_sampled_depths)
                    prev_estimated_sdf = mid_sdf - dot_val * dist * 0.5
                    next_estimated_sdf = mid_sdf + dot_val * dist * 0.5

                    prev_cdf = self.cdf_phi_s(prev_estimated_sdf, 64 * (2**i))
                    next_cdf = self.cdf_phi_s(next_estimated_sdf, 64 * (2**i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                    if not torch.isfinite(alpha.mean()):
                        pdb.set_trace()
                    weight = self.alpha_to_w(alpha)
                    if not torch.isfinite(weight.mean()):
                        pdb.set_trace()
                    # print(alpha.min(), alpha.max())
                    # print()
                    fine_sampled_depths = self.hierarchical_sample_depths(
                        weight,
                        pixel.shape[0],
                        pixel.device,
                        self._cfg.model.fine_sample_bins // n_upsample_iters,
                        sampled_depths,
                        not self.training)
                    # pdb.set_trace()
                    if not torch.isfinite(fine_sampled_depths.mean()):
                        pdb.set_trace()
                    sampled_depths = torch.cat([sampled_depths, fine_sampled_depths], dim=0)
                    
                    sdf_fine = self.get_sdf_value(fine_sampled_depths, pixel, camera_position, self._mlp).reshape(fine_sampled_depths.shape[0], -1)
                    if not torch.isfinite(sdf_fine.mean()):
                        pdb.set_trace()
                    # pdb.set_trace()
                    coarse_sdf = torch.cat([coarse_sdf, sdf_fine], dim=0)
                    sampled_depths, sampled_depths_sort_indices = torch.sort(sampled_depths, dim=0)
                    if (sampled_depths_sort_indices >= sampled_depths.shape[0]).sum() > 0:
                        pdb.set_trace()
                    # pdb.set_trace
                    coarse_sdf = torch.gather(coarse_sdf, 0, sampled_depths_sort_indices)

                fine_sampled_depths = sampled_depths

        # fine_sampled_depths = torch.cat([fine_sampled_depths, coarse_sampled_depths], dim=0)
        if self._cfg.model.upsample == 'default':
            fine_sdf, fine_coord, fine_color, fine_depths, fine_weights, fine_depth_variance = self.reconstruct_color_and_depths(
                target_depths,
                fine_sampled_depths,
                pixel,
                camera_position,
                viewdir,
                self._mlp,
                self._mlp_color)

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

        elif self._cfg.model.upsample == 'neus':
            # pdb.set_trace()
            fine_mid_sampled_depths = (fine_sampled_depths[1:, ...] + fine_sampled_depths[:-1, ...]) * 0.5

            fine_sdf, fine_coord, _, _ = self.get_sdf_value_with_grad(fine_sampled_depths, pixel, camera_position, self._mlp)
            if not torch.isfinite(fine_sdf.mean()):
                pdb.set_trace()

            _, _, last_geo_feature, _ = self.get_sdf_value_with_grad(fine_mid_sampled_depths, pixel, camera_position, self._mlp)
            viewdirs = self.repeat_tensor(viewdir, fine_mid_sampled_depths.shape[0])
            encodings_dir = self._positional_encoding_dir(viewdirs)
            color, _ = self._mlp_color(torch.cat([last_geo_feature, encodings_dir], dim=-1))

            cdf, opactiy_alpha = self.sdf_to_alpha(fine_sdf.reshape(fine_sampled_depths.shape[0], -1), self.forward_s())
            visibility_weights = self.alpha_to_w(opactiy_alpha)
            fine_color = torch.sum(visibility_weights[..., None] * color.reshape(fine_mid_sampled_depths.shape[0], -1, 3), 0)
            fine_depths = torch.sum(visibility_weights / (visibility_weights.sum(0, keepdim=True) + 1e-5) * fine_mid_sampled_depths, 0)

            return {'coarse_color': fine_color,
                    'coarse_depth': fine_depths,
                    'fine_color': fine_color,
                    'fine_depth': fine_depths,
                    'fine_sdf': fine_sdf,
                    'fine_coord': fine_coord,
                    'fine_sampled_depths': fine_sampled_depths,
                    }


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

    def get_sdf_value(self, sampled_depths, pixels, camera_positions, mlp_model):
        
        bins_count = sampled_depths.shape[0]
        # pdb.set_trace()
        sampled_depths = torch.sort(sampled_depths, dim=0).values
        sampled_depths = sampled_depths.reshape(-1)
        pixels = self.repeat_tensor(pixels, bins_count)
        camera_positions = self.repeat_tensor(camera_positions, bins_count)
        back_projected_points = back_project_pixel(pixels, sampled_depths, camera_positions,
                                                   self._inverted_camera_matrix).requires_grad_(True)

        if self._pos_encoding:
            encodings = self._positional_encoding(back_projected_points)
            prediction, _ = mlp_model(encodings)
        else:
            prediction, _ = mlp_model(back_projected_points)

        return prediction

    def get_sdf_value_with_grad(self, sampled_depths, pixels, camera_positions, mlp_model):
        bins_count = sampled_depths.shape[0]
        sampled_depths = torch.sort(sampled_depths, dim=0).values
        sampled_depths = sampled_depths.reshape(-1)
        pixels = self.repeat_tensor(pixels, bins_count)
        camera_positions = self.repeat_tensor(camera_positions, bins_count)
        back_projected_points = back_project_pixel(pixels, sampled_depths, camera_positions,
                                                   self._inverted_camera_matrix).requires_grad_(True)

        with torch.enable_grad():
            if self._pos_encoding:
                encodings = self._positional_encoding(back_projected_points)
                prediction, last_geo_feature = mlp_model(encodings)
            else:
                prediction, last_geo_feature = mlp_model(back_projected_points)
            sdf_grad = torch.autograd.grad(prediction, back_projected_points,
                                           torch.ones_like(prediction, device='cuda'),
                                           create_graph=True,
                                           retain_graph=True,
                                           only_inputs=True)[0]

        return prediction, back_projected_points, last_geo_feature, sdf_grad

    def get_sdf_color_value_query(self, xyz):
        encodings = self._positional_encoding(xyz)
        encodings_dir = self._positional_encoding_dir(torch.zeros_like(xyz).cuda())

        sdf, last_geo_feature = self._mlp(encodings)
        color, _ = self._mlp_color(torch.cat([last_geo_feature, encodings_dir], dim=-1))
        color = torch.sigmoid(color).reshape(-1, 3)
        sdf = sdf.reshape(-1, 1)

        return sdf, color


    def reconstruct_color_and_depths(self, target_depths, sampled_depths, pixels, camera_positions, viewdirs, mlp_model, mlp_model_color):
        bins_count = sampled_depths.shape[0]
        sampled_depths = torch.sort(sampled_depths, dim=0).values
        sampled_depths = sampled_depths.reshape(-1)
        pixels = self.repeat_tensor(pixels, bins_count)
        viewdirs = self.repeat_tensor(viewdirs, bins_count)
        camera_positions = self.repeat_tensor(camera_positions, bins_count)
        back_projected_points = back_project_pixel(pixels, sampled_depths, camera_positions,
                                                   self._inverted_camera_matrix).requires_grad_(True)

        if mlp_model_color is None:
            if self._pos_encoding:
                encodings = self._positional_encoding(back_projected_points)
                prediction, _ = mlp_model(encodings)
            else:
                prediction, _ = mlp_model(back_projected_points)
            # pdb.set_trace()
            colors = torch.sigmoid(prediction[:, :3]).reshape(bins_count, -1, 3)
            density = prediction[:, 3].reshape(bins_count, -1) # density == sdf
        else:
            if self._pos_encoding:
                encodings = self._positional_encoding(back_projected_points)
                encodings_dir = self._positional_encoding_dir(viewdirs)

                prediction1, last_feature = mlp_model(encodings)
                prediction2, _ = mlp_model_color(torch.cat([last_feature, encodings_dir], dim=-1))
            else:
                prediction1, last_feature = mlp_model(back_projected_points)
                prediction2, _ = mlp_model_color(torch.cat([last_feature, viewdirs], dim=-1))
            colors = torch.sigmoid(prediction2).reshape(bins_count, -1, 3)
            density = prediction1.reshape(bins_count, -1)

        if self._mode == 'volsdf':
            if self._cfg.model.volsdf_ab_fixed:
                # self._alpha.requires_grad_(False)
                self._beta.requires_grad_(False)
            # density = self.sdf_to_density(density, self._alpha, self._beta)
            density = self.sdf_to_density(density, self._beta)

        depths = sampled_depths.reshape(bins_count, -1)

        if self._mode == 'density' or self._mode == 'volsdf':
            weights = self.calculate_weights(density, depths)
        elif self._mode == 'sdf':
            weights = self.calculate_weights_from_sdf(density, depths, target_depths)

        reconstructed_color = self.reconstruct_color(colors, weights, self._default_color, self._mode)
        reconstructed_depths = self.reconstruct_depth(depths, weights, self._default_depth, self._mode)
        with torch.no_grad():
            reconstructed_depth_variance = self.reconstruct_depth_variance(depths, weights, reconstructed_depths,
                                                                           self._default_depth, self._mode)

        if mlp_model_color is None:
            density = prediction[:, 3:]
        else:
            density = prediction1

        return density, back_projected_points, \
               reconstructed_color, reconstructed_depths, weights, reconstructed_depth_variance

    def stratified_sample_depths(self, batch_size, device, bins_count, deterministic=False):
        if deterministic:
            depth_delta = (self._default_depth.item() - self._cfg.model.minimal_depth) / bins_count
            result = torch.arange(self._cfg.model.minimal_depth, self._default_depth.item(), depth_delta, device=device)
            result = torch.repeat_interleave(result[:, None], batch_size, dim=1)
            return result
        uniform = torch.rand((bins_count, batch_size), device=device)
        uniform[0] = 1
        result = (torch.arange(bins_count, device=device)[:, None] + uniform - 1
                  ) * (self._default_depth - self._cfg.model.minimal_depth) / (bins_count - 1) + self._cfg.model.minimal_depth
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
        index_below = self.clip_indexes(indexes - 1, 0, cdf.shape[1] - 1)
        index_above = self.clip_indexes(indexes, 0, cdf.shape[1] - 1)

        denominator = torch.gather(cdf, 1, index_above) - torch.gather(cdf, 1, index_below)
        denominator = torch.where(denominator < 1e-10, torch.ones_like(denominator), denominator)
        t = (uniform - torch.gather(cdf, 1, index_below)) / denominator

        index_below = self.clip_indexes(indexes - 1, 0, bins.shape[1] - 1)
        index_above = self.clip_indexes(indexes, 0, bins.shape[1] - 1)
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

    def calculate_weights_from_sdf(self, sdf, depths, target_depths):
        """
        # pdb.set_trace()
        # front_mask, sdf_mask, _, _ = self.get_masks(depths, target_depths)
        weights = F.sigmoid(sdf / self._truncation) * F.sigmoid(-sdf / self._truncation)

        signs = sdf[1:, :] * sdf[:-1, :]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        indices = torch.argmax(mask, axis=0)
        indices = torch.unsqueeze(indices, 0)
        depth_min = torch.gather(depths, 0, indices)
        # pdb.set_trace()
        mask = torch.where(depths < depth_min + self._truncation, torch.ones_like(depths), torch.zeros_like(depths))
        # mask = torch.where((depths < depth_min + self._truncation) & (depths > depth_min - self._truncation), torch.ones_like(depths), torch.zeros_like(depths))

        weights = weights * mask
        weights = weights / (torch.sum(weights, dim=0, keepdim=True) + 1e-8)
        """

        weights = []
        s = self._truncation
        product = 1
        for i in range(len(depths)):
            hit_probability = 4 * (F.sigmoid(sdf[i, :] / s) * F.sigmoid(-sdf[i, :] / s))
            weights.append(hit_probability * product)
            product = product * (1 - hit_probability)
        weights = torch.stack(weights, 0)

        # pdb.set_trace()
        weights = weights / (torch.sum(weights, dim=0, keepdim=True) + 1e-12)
        # weights = weights / (torch.max(weights, dim=0, keepdim=True)).values

        return weights

    def sdf_to_density(self, sdf, beta):
        exp = torch.exp(-torch.abs(sdf) / beta)
        psi = torch.where(sdf >= 0, 0.5 * exp, 1 - 0.5 * exp)

        return (1. / beta) * psi

    def cdf_phi_s(self, x, s):
        return torch.sigmoid(x * s)

    def sdf_to_alpha(self, sdf, s):
        cdf = self.cdf_phi_s(sdf, s)
        opacity_alpha = (cdf[:-1, ...] - cdf[1:, ...]) / (cdf[:-1, ...] + 1e-5)
        opacity_alpha = torch.clamp_min(opacity_alpha, 0)
        return cdf, opacity_alpha

    def sdf_to_w(self, sdf, s):
        device = sdf.device
        cdf, opacity_alpha = self.sdf_to_alpha(sdf, s)

        shifted_transparency = torch.cat([torch.ones([1, *opacity_alpha.shape[1:]]).cuda(), 1.0 - opacity_alpha + 1e-5], dim=0)

        visibility_weights = opacity_alpha * torch.cumprod(shifted_transparency, dim=0)[:-1, ...]

        return cdf, opacity_alpha, visibility_weights

    def alpha_to_w(self, alpha):
        shifted_transparency = torch.cat([torch.ones([1, *alpha.shape[1:]]).cuda(), 1.0 - alpha + 1e-5], dim=0)
        if not torch.isfinite(shifted_transparency.mean()):
            pdb.set_trace()

        visibility_weights = alpha * torch.cumprod(shifted_transparency, dim=0)[:-1, ...]
        if not torch.isfinite(visibility_weights.mean()):
            pdb.set_trace()

        return visibility_weights

    @staticmethod
    def reconstruct_color(colors, weights, default_color, mode):
        if mode == 'density' or mode == 'volsdf':
            return torch.sum(colors * weights[:-1, :, None], dim=0
                            ) + default_color.to(colors.device)[None] * weights[-1, :, None]
        else:
            return torch.sum(colors * weights[:, :, None], dim=0)

    @staticmethod
    def reconstruct_depth(depths, weights, default_depth, mode):
        if mode == 'density' or mode == 'volsdf':
            return torch.sum(depths * weights[:-1, :], dim=0) + default_depth.to(depths.device)[None] * weights[-1]
        else:
            return torch.sum(depths * weights[:, :], dim=0)


    @staticmethod
    def reconstruct_depth_variance(depths, weights, mean_depths, default_depth, mode):
        if mode == 'density' or mode == 'volsdf':
            return torch.sum((depths - mean_depths[None]) ** 2 * weights[:-1], dim=0
                            ) + (default_depth.to(depths.device)[None] - mean_depths) ** 2 * weights[-1]
        else:
            return torch.sum((depths - mean_depths[None]) ** 2 * weights, dim=0)

    def loss(self, batch, reduction=True):
        camera_position = self.positions_from_batch(batch)
        output = self.forward(batch["pixel"], batch["depth"], camera_position, batch["viewdir"])

        # pdb.set_trace()

        mask = (batch["depth"] > 1e-12) & (batch["depth"] < self._default_depth)
        max_depth_mask = (batch["depth"] < self._default_depth)[..., None]
        valid_weight = batch["depth"].nelement() / (mask.count_nonzero() + 1e-18)

        if self._cfg.model.loss_type == 'l1':
            coarse_image_loss = torch.mean(self._loss(output['coarse_color'] * mask, batch["color"] * mask), dim=1)
            coarse_depth_weights = 1. / (torch.sqrt(output['coarse_depth_variance']) + 1e-18) * mask
            coarse_depth_loss = self._loss(output['coarse_depth'] * coarse_depth_weights, batch["depth"] * coarse_depth_weights)

            fine_image_loss = torch.mean(self._loss(output['fine_color'] * mask, batch["color"] * mask), dim=1)
            fine_depth_weights = 1. / (torch.sqrt(output['fine_depth_variance']) + 1e-18) * mask
            fine_depth_loss = self._loss(output['fine_depth'] * fine_depth_weights, batch["depth"] * fine_depth_weights)

        elif self._cfg.model.loss_type == 'l2':
            coarse_image_loss = torch.mean(self._mseloss(output['coarse_color'] * mask, batch["color"] * mask), dim=1)
            coarse_depth_loss = self._mseloss(output['coarse_depth'] * mask, batch["depth"] * mask) * valid_weight

            fine_image_loss = torch.mean(self._mseloss(output['fine_color'] * mask, batch["color"] * mask), dim=1)
            fine_depth_loss = self._mseloss(output['fine_depth'] * mask, batch["depth"] * mask) * valid_weight

        if self._mode == 'sdf' or self._mode == 'volsdf':
            fine_sampled_depths = output['fine_sampled_depths'].transpose(0, 1)
            batch_size = fine_sampled_depths.shape[0]
            fine_sdf = output['fine_sdf'].reshape(-1, batch_size).transpose(0, 1)
            target_depth = batch["depth"].unsqueeze(-1)
            freespace_constraint, sdf_constraint = self.get_sdf_loss(fine_sdf, fine_sampled_depths, target_depth)

            sdf_loss = freespace_constraint * self._cfg.model.fs_loss_koef \
                     + sdf_constraint * self._cfg.model.sdf_loss_koef

            grad_constraint = 0.0
            if self._cfg.model.grad_loss_koef > 0.0:
                grad = self.gradient(output['fine_sdf'], output['fine_coord'])
                grad = torch.norm(grad, dim=-1)
                grad_constraint = self._mseloss(grad, torch.ones_like(grad)).mean()
                sdf_loss = sdf_loss + grad_constraint * self._cfg.model.grad_loss_koef

        else:
            sdf_constraint = freespace_constraint = grad_constraint = 0.0
            sdf_loss = 0.0


        image_loss = coarse_image_loss + fine_image_loss
        depth_loss = coarse_depth_loss + fine_depth_loss
        # image_loss = fine_image_loss
        # depth_loss = fine_depth_loss

        loss = self._cfg.model.color_loss_koef * image_loss + self._cfg.model.depth_loss_koef * depth_loss

        if reduction:
            coarse_depth_loss = torch.mean(coarse_depth_loss)
            coarse_image_loss = torch.mean(coarse_image_loss)
            fine_depth_loss = torch.mean(fine_depth_loss)
            fine_image_loss = torch.mean(fine_image_loss)
            loss = torch.mean(loss)

        loss = loss + sdf_loss

        losses = {
            "coarse_image_loss": coarse_image_loss,
            "coarse_depth_loss": coarse_depth_loss,
            "fine_image_loss": fine_image_loss,
            "fine_depth_loss": fine_depth_loss,
            "sdf_constraint": sdf_constraint,
            "freespace_constraint": freespace_constraint,
            "gradient_constraint": grad_constraint,
            "total_loss": loss
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
        # weight_mask = self.get_weight_masks(sampled_depths, target_depth)

        fs_loss = self._mseloss(sdf * front_mask, torch.ones_like(sdf) * front_mask) * fs_weight
        sdf_loss = self._mseloss((sampled_depths + sdf * self._truncation) * sdf_mask, target_depth * sdf_mask) * sdf_weight
        # fs_loss = self._mseloss(sdf * front_mask, torch.ones_like(sdf) * front_mask) * weight_mask * fs_weight
        # sdf_loss = self._mseloss((sampled_depths + sdf * self._truncation) * sdf_mask, target_depth * sdf_mask) * weight_mask * sdf_weight
        # sdf_loss = self._mseloss((target_depth + sdf * self._truncation) * sdf_mask, sampled_depths * sdf_mask) * sdf_weight

        if not torch.isfinite(fs_loss.mean()) or not torch.isfinite(sdf_loss.mean()):
            pdb.set_trace()

        return fs_loss.mean(), sdf_loss.mean()

    def get_masks(self, sampled_depths, target_depths):
        # pdb.set_trace()
        front_mask = torch.where(sampled_depths < (target_depths - self._truncation), torch.ones_like(sampled_depths), torch.zeros_like(sampled_depths))
        back_mask = torch.where(sampled_depths > (target_depths + self._truncation), torch.ones_like(sampled_depths), torch.zeros_like(sampled_depths))
        depth_mask = torch.where(target_depths > 0.0, torch.ones_like(target_depths), torch.zeros_like(target_depths))
        sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask
        front_mask = front_mask * depth_mask

        num_fs_points = torch.count_nonzero(front_mask)
        num_sdf_points = torch.count_nonzero(sdf_mask)
        fs_weight = 1.0 - (num_fs_points / (num_fs_points + num_sdf_points + 1e-12))
        sdf_weight = 1.0 - (num_sdf_points / (num_fs_points + num_sdf_points + 1e-12))

        return front_mask, sdf_mask, fs_weight, sdf_weight

    def get_weight_masks(self, sampled_depths, target_depths):
        front_mask = torch.where(sampled_depths < (target_depths - self._truncation), torch.ones_like(sampled_depths), torch.zeros_like(sampled_depths))
        back_mask = torch.where(sampled_depths > (target_depths + self._truncation), torch.ones_like(sampled_depths), torch.zeros_like(sampled_depths))
        depth_mask = torch.where(target_depths > 0.0, torch.ones_like(target_depths), torch.zeros_like(target_depths))
        sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

        # pdb.set_trace()
        distance_mask = ((target_depths - sampled_depths) / self._truncation + 1e-8)
        distance_mask = torch.where(sdf_mask > 0.0, distance_mask, torch.ones_like(distance_mask))
        # weight_mask = 1.0 / (distance_mask * distance_mask)
        weight_mask = 1.0 / (torch.abs(distance_mask)) * depth_mask
        # weight_mask = torch.where(sdf_mask != 0.0, weight_mask, torch.ones_like(weight_mask))

        return weight_mask
