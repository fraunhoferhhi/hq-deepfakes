import logging
import zlib
import pickle

import cv2 
import numpy as np

import torch
from torch import nn


## MODEL
# CONFIGS
EB4_RDA = {
        "model_input_size": 256,
        "model_output_size": 256,
        "enc_architecture": "efficientnet_b4",
        "enc_load_weights": True,
        "bottleneck_in_encoder": True,
        "bottleneck_size": 1024,
        "bottleneck_type": "dense",
        "bottleneck_norm": None,
        "bottleneck_tanh": False,
        "fc_depth": 1,
        "fc_min_filters": 1280,
        "fc_max_filters": 1280,
        "fc_dims": 8,
        "fc_filter_slope": -0.5,
        "fc_dropout": 0.0,
        "fc_upsampler": "upsample",
        "fc_upsamples": 1,
        "fc_upsample_filters": 1280,
        "split_fc": False,
        "enable_gblock": False,
        "fc_gblock_depth": 3,
        "fc_gblock_min_filters": 512,
        "fc_gblock_max_filters": 512,
        "fc_gblock_dropout": 0.0,
        "fc_gblock_filter_slope": -0.5,
        "split_gblock": False,
        "split_decoders": True,
        "dec_upscale_method": "subpixel",
        "dec_norm": None,
        "dec_min_filters": 128,
        "dec_max_filters": 512,
        "dec_filter_slope": -0.33,
        "dec_res_blocks": 2,
        "dec_output_kernel": 3, 
        "dec_gaussian": False,
        "dec_skip_last_residual": True
    }

# BLOCKS
# upscale blocks
class UpscaleBlock(nn.Module):
    '''
    Subpixel upscaling block based on the one in the faceswap tool
    inputs:
        n_filters: Amount of output filters
        scale: amount to scale height and width of output
    ''' 
    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        self.conv = nn.Conv2d(in_channels, out_channels*self.scale*2, kernel_size, 1, "same")

    def forward(self, x):
        bs, _, height, width = x.shape
        out = self.conv(x)
        out = out.view(bs, self.scale, self.scale, self.out_channels, height, width)
        out = torch.permute(out, (0,3,4,1,5,2))
        out = torch.reshape(out, (bs, self.out_channels, height*self.scale, width*self.scale))

        return out

class Upsampling2D(nn.Module):
    '''
    simple upscale via Upsample
    '''

    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

        self.up = nn.Upsample(scale_factor=self.scale_factor, mode=self.mode)

    def forward(self, x):
        return self.up(x)

class UpscaleResizeImagesBlock(nn.Module):
    '''
    Upscale Block with 1 Upsample, 1 Conv and 1 Conv Transpose
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2, mode="bilinear"):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.mode = mode

        self.up = Upsampling2D(self.scale, self.mode)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=1, padding="same")
        self.conv_T = nn.ConvTranspose2d(self.in_channels, self.out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        x_1 = self.up(x)
        x_1 = self.conv(x_1)

        x_2 = self.conv_T(x, output_size=x_1.size())
        retval = x_1 + x_2

        return retval


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, use_bias=True, kernel_size=3, stride=1):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, "same", bias=use_bias)
        self.act_0 = nn.LeakyReLU(negative_slope=0.2)
        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, "same", bias=use_bias)
        self.act_1 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        input = x.clone()

        x = self.conv_0(x)
        # x = self.act_0(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.conv_1(x)
        out = x + input
        # out = self.act_1(out)
        out = nn.LeakyReLU(negative_slope=0.2)(out)

        return out

# regularization
class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        
        self.sigma=sigma

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x)*self.sigma

        return x

# style 
class GBlock(nn.Module):
    def __init__(self, x_dim=1280, style_dim=512):
        super().__init__()
        self.dense_nodes = 512
        self.dense_recursions = 3
        self.x_dim = x_dim

        self.style_dense = nn.Sequential(
            nn.Linear(style_dim, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, 512)
        )

        self.x_prep = nn.Sequential(
            nn.Conv2d(in_channels=x_dim, out_channels=x_dim, kernel_size=3, stride=1, padding="same"),
            GaussianNoise()
        )

        self.gb_dense_1 = nn.Linear(512, x_dim)
        self.gb_dense_2 = nn.Linear(512, x_dim)
        self.gb_noise = nn.Sequential(
            GaussianNoise(),
            nn.Conv2d(in_channels=x_dim, out_channels=x_dim, kernel_size=1, padding="same")
        )
        self.gb_conv = nn.Conv2d(x_dim, x_dim, 3, padding="same")

    def forward(self, x):
        style = x[-1]
        x = x[0]

        style = self.style_dense(style)
        x = self.x_prep(x)

        x = self._gblock_step(x, style, filters=self.x_dim)

        return x

    def _gblock_step(self, x, style, filters=1280, recursions=2):
        for i in range(recursions):
            style_1 = self.gb_dense_1(style).view(-1, filters, 1, 1)
            style_2 = self.gb_dense_2(style).view(-1, filters, 1, 1)

            noise = self.gb_noise(x)

            if i == recursions -1:
                x = self.gb_conv(x)

            x = AdaInNorm()([x, style_1, style_2])
            x += noise
            x = nn.LeakyReLU(0.2)(x)

        return x

class AdaInNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        beta = x[1]
        gamma = x[2]
        x = x[0]

        N, C = x.size()[:2]

        mu = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

        sigma = x.view(N, C, -1).var(dim=2) + 1e-5
        sigma = sigma.sqrt().view(N, C, 1, 1)

        x = (x - mu) / sigma
        x = x*gamma + beta

        return x


# reshape
class CustomReshape(nn.Module):
    def __init__(self, dim=8, depth=1280):
        super().__init__()
        self.dim = dim
        self.depth = depth

    def forward(self, x):
        return x.view((-1, self.depth, self.dim, self.dim))

class CustomFlatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.flatten(x, self.dim)



## DATASET
# AUGMENTATION

class Augmentator():
    def __init__(self, 
                 input_dim: int = 512, 
                 model_input_size: int = 256, 
                 model_output_size: int = 256, 
                 coverage_ratio: int = 0.875, 
                 no_flip: bool = False):
        '''
        Augmentator class.
        params:
            input_dim: dimension of faceset images
            model_input_size: dimension of model inputs 
            model_output_size: dimension of model outputs
            coverage_ratio: amount of images to be used (taken from center)
            no_flip: indicates if flipping is enabled/disabled

        '''
        self.coverage_ratio = coverage_ratio
        self.input_dim = input_dim
        self.model_input_size = model_input_size
        self.model_output_size = model_output_size
        self.no_flip = no_flip
        self.flip_chance = 0.5
        self.coverage = int(self.input_dim * self.coverage_ratio // 2) * 2

    def augment_classic(self, img):
        # randomly rotates, zooms, shifts and flips image
        logging.debug("Classic Augmentation")
        rotation_range = 10
        zoom_range = 5 / 100 
        shift_range = 5 / 100

        rotation = np.random.uniform(-rotation_range, rotation_range, 1).astype("float32")
        scale = np.random.uniform(1 - zoom_range, 1 + zoom_range, 1).astype("float32")
        tform = np.random.uniform(-shift_range, shift_range, size=(1,2)).astype("float32") * self.input_dim

        mat = cv2.getRotationMatrix2D((self.input_dim // 2, self.input_dim // 2), angle=rotation[0], scale=scale[0])
        mat[:, 2] = mat[:,2] + tform

        img = np.array(cv2.warpAffine(img, mat, (self.input_dim, self.input_dim), borderMode=cv2.BORDER_REPLICATE))

        return img

    def augment_color(self, img):
        # randomly performs clahe and lightness adjustments
        logging.debug("Augmenting color")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = self.random_clahe(img)
        img = self.random_lab(img)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

        return img

    def random_clahe(self, img):
        # randomly applies clahe color augmentation to img in lgb space with the following params
        clahe_chance = 0.5
        base_contrast = self.input_dim // 128
        clahe_max_size = 4

        perform_clahe = (np.random.rand(1) > clahe_chance)[0]
        if perform_clahe:
            grid_base = np.rint(np.random.uniform(0, clahe_max_size, 1)).astype("uint8")
            contrast_adjustment = (grid_base*(base_contrast // 2))
            grid_size = (contrast_adjustment + base_contrast)[0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))

            img[:, :, 0] = clahe.apply(img[:, :, 0])

        return img

    def random_lab(self, img):
        # randomly adjusts lightness/color
        amount_l = 30 / 100 
        amount_ab = 8 / 100
        adjust = np.array([amount_l, amount_ab, amount_ab], dtype="float32")
        random = np.random.rand(3).astype("float32")*(adjust*2) - adjust

        for idx in range(random.shape[-1]):
            adjustment = random[idx]
            if adjustment >= 0:
                img[:, :, idx] = ((255 - img[:, :, idx]) * adjustment) + img[:, :, idx]
            else:
                img[:, :, idx] = img[:, :, idx] * (1 + adjustment)

        return img

    def warp(self, img):
        warp_range_ = np.linspace(self.input_dim // 2 - self.coverage // 2,
                                  self.input_dim // 2 + self.coverage // 2, 5, dtype='float32')
        warp_mapx = np.broadcast_to(warp_range_, (5, 5)).astype("float32")
        warp_mapy = np.broadcast_to(warp_mapx.T, (5, 5)).astype("float32")

        warp_pad = int(1.25 * self.model_input_size)
        warp_slices = slice(warp_pad // 10, -warp_pad // 10)


        rands = np.random.normal(size=(2, 5, 5),
                                 scale=5).astype("float32")

        batch_maps = np.stack((warp_mapx, warp_mapy), axis=0) + rands
        batch_interp = np.array([cv2.resize(map, (warp_pad, warp_pad))[warp_slices, warp_slices] for map in batch_maps])

        warped_batch = cv2.remap(img, batch_interp[0], batch_interp[1], cv2.INTER_LINEAR)

        return warped_batch

    def skip_warp(self, img, size):
        tgt_slices = slice(self.input_dim // 2 - self.coverage // 2,
                           self.input_dim // 2 + self.coverage // 2)

        img = cv2.resize(img[tgt_slices, tgt_slices], (size, size), cv2.INTER_AREA)
        return img

    def process(self, img, mask):
        # apply all relevant augmentations to the img

        img = self.augment_color(img)

        img_mask = np.dstack((img, mask))
        img_mask = self.augment_classic(img_mask)
        img = img_mask[:,:,:3]
        mask = img_mask[:, :, 3]

        y = img.copy()
        y = self.skip_warp(y, self.model_output_size)

        img = self.warp(img)
        mask = self.skip_warp(mask, self.model_output_size)


        perform_flip = (np.random.rand(1) > self.flip_chance)[0] if not self.no_flip else False
        if perform_flip:
            logging.debug("Flip image, target and mask")
            img = np.flip(img, 1)
            y = np.flip(y, 1)
            mask = np.flip(mask, 1)

        return img.copy(), y.copy(), mask.copy()




