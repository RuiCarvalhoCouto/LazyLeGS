#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from collections import OrderedDict

import torch
from torch import nn
import numpy as np
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class ImageTensorCache:
    def __init__(self, max_size):
        if max_size < 0:
            raise ValueError("--image_cache_size must be >= 0")
        self.max_size = max_size
        self._cache = OrderedDict()

    def get(self, key):
        if self.max_size == 0:
            return None
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def put(self, key, value):
        if self.max_size == 0:
            return
        self._cache[key] = value.detach().cpu()
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_path=None, image_resolution=None, image_cache=None, lazy_images=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.lazy_images = lazy_images
        self.image_path = os.path.abspath(image_path) if image_path is not None else None
        self.image_resolution = image_resolution
        self.image_cache = image_cache
        self._original_image = None

        if self.lazy_images:
            if self.image_path is None or self.image_resolution is None:
                raise ValueError("Lazy cameras require image_path and image_resolution")
            self.image_width = int(self.image_resolution[0])
            self.image_height = int(self.image_resolution[1])
        else:
            self._original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self._original_image.shape[2]
            self.image_height = self._original_image.shape[1]

            if gt_alpha_mask is not None:
                self._original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self._original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def _preprocess_image(pil_image, resolution):
        resized_image_rgb = PILtoTorch(pil_image, resolution)
        gt_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)

        if resized_image_rgb.shape[0] == 4:
            gt_image *= resized_image_rgb[3:4, ...]

        return gt_image.detach().cpu()

    def _load_original_image_cpu(self):
        cache_key = (self.image_path, int(self.image_resolution[0]), int(self.image_resolution[1]))

        if self.image_cache is not None:
            cached_image = self.image_cache.get(cache_key)
            if cached_image is not None:
                return cached_image

        with Image.open(self.image_path) as img:
            image = self._preprocess_image(img, self.image_resolution)

        if self.image_cache is not None:
            self.image_cache.put(cache_key, image)

        return image

    @property
    def original_image(self):
        if not self.lazy_images:
            return self._original_image

        image = self._load_original_image_cpu()
        if self.data_device.type == "cpu":
            return image
        return image.to(self.data_device)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

