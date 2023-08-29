import numpy as np

import torch
from torch import nn
from torchvision.models import get_model

from .utils import EB4_RDA
from .utils import UpscaleBlock, Upsampling2D, UpscaleResizeImagesBlock, ResidualBlock, GaussianNoise, CustomReshape, CustomFlatten, GBlock


class custom(nn.Module):
    '''
    Custom model builder for deepfake autoencoders.
    input:
        cfg: name of the config to use, config holds the model specs 
    '''

    def __init__(self, cfg: str = "EB4_RDA"):
        super(custom, self).__init__()

        cfg = globals()[cfg]
        self.__dict__.update(cfg)

        self.encoder = self._build_encoder()

        self.inter_a = self._build_inter()
        self.inter_b = self._build_inter() if self.split_fc else self.inter_a

        if self.enable_gblock:
            self.inter_gb = self._build_gblock_inter()
            self.gblock_a = GBlock(x_dim = self.inter_output_filters, style_dim = self.fc_gblock_max_filters)
            self.gblock_b = GBlock(x_dim = self.inter_output_filters, style_dim = self.fc_gblock_max_filters) if self.split_gblock else self.gblock_a
        else:
            self.inter_gb, self.gblock_a, self.gblock_b = None, None, None

        self.decoder_input_dim = self._scale_dim(self.model_output_size, self.fc_dims * self.fc_upsamples * 2)

        self.decoder_a = self._build_decoder()
        self.decoder_b = self._build_decoder() if self.split_decoders else self.decoder_a

    # components
    def _build_encoder(self):
        # load efficientnet encoder and apply bottleneck if specified
        # missing: custom encoder

        dummy_input = torch.randn(1, 3, self.model_input_size, self.model_input_size)
        encoder = get_model(self.enc_architecture, weights="DEFAULT").features

        self.encoder_output_dim = encoder(dummy_input).shape[1:]

        if self.bottleneck_in_encoder:
            bottleneck = self._build_bottleneck()
            encoder = nn.Sequential(
                encoder,
                bottleneck
            )

        return encoder

    def _build_bottleneck(self):
        # build bottleneck layer, flatten before linear layers is included here
        sequence = []

        if self.bottleneck_norm:
            norm = self._get_norm_layer(self.bottleneck_norm, self.encoder_output_dim[0])
            sequence += norm
        if self.bottleneck_type == "dense":
            sequence += [CustomFlatten()]
            self.inter_input_filters = self.bottleneck_size

        layer = self._get_bottleneck_layer(self.bottleneck_type, self.bottleneck_size, self.encoder_output_dim[1])
        sequence += [layer]

        if self.bottleneck_type != "dense":
            sequence += [CustomFlatten()]
            self.inter_input_filters = self.encoder_output_dim[0]

        # if self.bottleneck_tanh:
        #     sequence += [torch.nn.Tanh()]

        bottleneck = nn.Sequential(*sequence)
        return bottleneck

    def _build_inter(self):
        # build intermediate blocks

        sequence = []

        # start with bottleneck if not in encoder
        if not self.bottleneck_in_encoder:
            bottleneck = self._build_bottleneck()
            sequence += bottleneck

        # inter input filters is set based on bottleneck type
        curve = self._get_filter_curve(self.fc_min_filters, self.fc_max_filters, self.fc_filter_slope, self.fc_depth)
        curve = [self.inter_input_filters / (self.fc_dims * self.fc_dims)] + curve

        for i in range(self.fc_depth):
            sequence += [nn.Dropout(p=self.fc_dropout), nn.Linear(int(curve[i] * self.fc_dims**2), curve[i+1] * self.fc_dims**2)]

        sequence += [CustomReshape(dim = self.fc_dims, depth = self.fc_max_filters)]

        in_filters = self.fc_max_filters
        for i in range(self.fc_upsamples):
            upsampler = self._get_upscaler(self.fc_upsampler, in_filters, self.fc_upsample_filters)
            sequence += [upsampler, nn.LeakyReLU(negative_slope=0.01)]
            in_filters = self.fc_upsample_filters

        self.inter_output_filters = self.fc_max_filters if self.fc_upsampler == "upsample" else self.fc_upsample_filters

        inter = nn.Sequential(*sequence)
        return inter
        
    def _build_gblock_inter(self):

        sequence = []

        # add bottleneck if not in encoder
        if not self.bottleneck_in_encoder:
            bottleneck = self._build_bottleneck()
            sequence += bottleneck

        curve = self._get_filter_curve(self.fc_gblock_min_filters, self.fc_gblock_max_filters, self.fc_gblock_filter_slope, self.fc_gblock_depth)
        curve = [self.inter_input_filters] + curve

        for i in range(self.fc_gblock_depth):
            sequence += [nn.Dropout(p=self.fc_gblock_dropout), nn.Linear(curve[i] , curve[i+1])]

        inter = nn.Sequential(*sequence)

        return inter

    def _build_decoder(self):

        num_upscales = int(np.log2(self.model_output_size / self.decoder_input_dim))

        curve = self._get_filter_curve(self.dec_max_filters, self.dec_min_filters, self.dec_filter_slope, num_upscales)
        curve = [self.fc_max_filters] + curve

        sequence = []
        last_filter = self.fc_max_filters
        for i in range(num_upscales):
            filters = curve[i+1]
            upscaler = self._get_upscaler(self.dec_upscale_method, last_filter, filters)
            sequence += [upscaler]

            if self.dec_gaussian:
                sequence += [GaussianNoise()]
            if self.dec_norm:
                norm = self._get_norm_layer(self.dec_norm, filters)
                sequence += [norm]
            
            sequence += [nn.LeakyReLU(negative_slope=0.2)]
            for _ in range(self.dec_res_blocks):
                if (i == num_upscales-1) and self.dec_skip_last_residual:
                    continue
                else:
                    sequence+=[ResidualBlock(filters)]

            last_filter = filters

        sequence += [
            nn.Conv2d(last_filter, 3, kernel_size=self.dec_output_kernel, stride=1, padding="same"),
            nn.Sigmoid()
        ]

        # this is not working properly for now
        if self.decoder_input_dim != self.fc_dims * self.fc_upsamples *2:
            old_dim = self.fc_dims * self.fc_upsamples *2
            depth = int(self.fc_max_filters*(old_dim / self.decoder_input_dim)**2)

            sequence = [CustomReshape(dim=self.decoder_input_dim, depth=depth)] + sequence

        decoder = nn.Sequential(*sequence)
        return decoder

    # helper functions
    def _get_bottleneck_layer(self, method, filters=None, kernel=None):
        if method == "dense":
            return nn.Linear(prod(self.encoder_output_dim), filters)
        if method == "average_pooling":
            return nn.AvgPool2d(kernel)
        if method == "max_pooling":
            return nn.MaxPool2d(kernel)

    def _get_norm_layer(self, method, filters):
        if method == "instance":
            return nn.InstanceNorm2d(filters)
        if method == "batch":
            return nn.BatchNorm2d(filters)
        if method == "layer":
            return nn.GroupNorm(1, filters)
        if method == "group":
            return nn.GroupNorm(filters // 32, filters)

    def _get_upscaler(self, method, in_filters, out_filters):
        if method == "resize_images":
            return UpscaleResizeImagesBlock(in_filters, out_filters)
        if method == "subpixel":
            return UpscaleBlock(in_filters, out_filters)
        if method == "upsample":
            return Upsampling2D()

    def _get_filter_curve(self, start, end, slope, num_points):
        x_axis = np.linspace(0., 1., num=num_points)
        y_axis = (x_axis - x_axis * slope) / (slope - abs(x_axis) * 2 * slope + 1)
        y_axis = y_axis * (end - start) + start
        retval = [int((y // 8) * 8) for y in y_axis]

        return retval

    def _scale_dim(self, target_resolution, original_dim):
        new_dim = target_resolution
        while new_dim > original_dim:
            next_dim = new_dim / 2
            if not next_dim.is_integer():
                break
            new_dim = int(next_dim)

        return new_dim

    # core
    def forward(self, x):
        x_0 = x[:, :3]
        x_1 = x[:, 3:]

        x_0 = self.encoder(x_0)
        x_1 = self.encoder(x_1)

        if self.enable_gblock:
            x_0_style = self.inter_gb(x_0)
            x_1_style = self.inter_gb(x_1)

        x_0 = self.inter_a(x_0)
        x_1 = self.inter_b(x_1)
        
        if self.enable_gblock:
            x_0 = self.gblock_a([x_0, x_0_style])
            x_1 = self.gblock_b([x_1, x_1_style])

        x_0 = self.decoder_a(x_0)
        x_1 = self.decoder_b(x_1)



        return x_0, x_1

    def forward_from_to(self, x_in, dec="b"):
        inter = self.inter_a if dec=="a" else self.inter_b
        gblock = self.gblock_a if dec == "a" else self. gblock_b
        decoder = self.decoder_a if dec == "a" else self.decoder_b

        x = x_in.clone()

        x = self.encoder(x)
        if self.enable_gblock:
            x_style = self.inter_gb(x)
        x = inter(x)
        if self.enable_gblock:
            x = gblock([x, x_style])
        x = decoder(x)

        return x
    
    def forward_from_latent_to(self, latent, dec="b"):
        inter = self.inter_a if dec=="a" else self.inter_b
        gblock = self.gblock_a if dec == "a" else self. gblock_b
        decoder = self.decoder_a if dec == "a" else self.decoder_b

        x = latent.clone()
        if self.enable_gblock:
            x_style = self.inter_gb(x)
        x = inter(x)
        if self.enable_gblock:
            x = gblock([x, x_style])
        x = decoder(x)

        return x

def prod(iterable):
    p = 1
    for n in iterable:
        p*= n
    return p



if __name__ == "__main__":
    model = custom()

    dummy = torch.zeros(8, 6, 256, 256)
    out = model(dummy)
    print(out[0].shape)
