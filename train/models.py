from inspect import isfunction

import torch
from torch import nn

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def dropout_nd(dims, use_nd, p=0.0, *args, **kwargs):
    if p == 0.0:
        return nn.Identity()
    elif not use_nd:
        return nn.Dropout(*args, **kwargs) # dropout of single entries in tensor
    elif dims == 1:
        return nn.Dropout(*args, **kwargs)
    elif dims == 2:
        return nn.Dropout2d(*args, **kwargs) # dropout of entire channels (e.g. RGB for an image)
    elif dims == 3:
        return nn.Dropout3d(*args, **kwargs)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def Normalization(group, ch):
    if group == -1:
        group = ch
    # return nn.GroupNorm(group, ch)
    # return GroupNorm32(group, ch) #NOTE: default from sr3
    # return {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[dims](ch)
    # return {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[dims](ch)
    # return nn.BatchNorm2d(ch)
    return nn.BatchNorm3d(ch) #NOTE: try this instead to find bottleneck in model

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dims, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners = True) #NOTE: try trilinear instead or nearest
        # self.up = nn.Upsample(scale_factor=2, mode="nearest") #NOTE: try trilinear instead or nearest
        self.conv = conv_nd(dims, dim, dim, 3, padding=1, padding_mode = "replicate") #TODO default padding mode would be "zeros", try "replicate"

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dims, dim):
        super().__init__()
        self.conv = conv_nd(dims, dim, dim, 3, 2, 1, padding_mode = "replicate")
        #                         in, out, kernel, stride, padding 

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, dims, dim, dim_out, groups=32, dropout=0, use_nd_dropout=False):
        super().__init__()
        self.block = nn.Sequential(
            Normalization(groups, dim), # basically same as nn.GroupNorm
            Swish(),
            dropout_nd(dims, use_nd_dropout, dropout), # dims tells whether to use dropout along None, 1d, 2d or 3d channels
            conv_nd(dims, dim, dim_out, 3, padding=1, padding_mode = "replicate"), # dim = in_channels, dim_out = out_channels, kernel_size = 3, pads with zeros
        )

    def forward(self, x):
        return self.block(x)
    
class UNet_no_attn(nn.Module):
    def __init__(
        self,
        dims=3, # SR3: 3
        in_channel=1, # SR3: 2
        out_channel=1, # SR3: 1
        inner_channel=4, # NOTE SR3 uses 64, we might not be able to do this ^^
        norm_groups=1,
        channel_mults=[1, 2, 4], # SR3: [1, 2, 4]
        res_blocks=1, # SR3: 4
        dropout=0, # SR3: 0.1
        image_size=64, # SR3: 32
        use_nd_dropout=False,
    ):
        super().__init__()
        self.dims = dims
        self.image_size = image_size

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [conv_nd(dims, in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(
                    Block(
                    dims,
                    pre_channel,
                    channel_mult,
                    groups=norm_groups,
                    dropout=dropout,
                    use_nd_dropout=use_nd_dropout,
                    )
                )
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(dims, pre_channel)) # uses conv. to downsample res by 2
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList(
            [
                Block(
                    dims,
                    pre_channel,
                    pre_channel,
                    groups=norm_groups,
                    dropout=dropout,
                    use_nd_dropout=use_nd_dropout,
                ),
                Block(
                    dims,
                    pre_channel,
                    pre_channel,
                    groups=norm_groups,
                    dropout=dropout,
                    use_nd_dropout=use_nd_dropout,
                )
            ]
        )

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = ind < 1
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(
                    
                    Block(
                    dims,
                    pre_channel + feat_channels.pop(),
                    channel_mult,
                    groups=norm_groups,
                    dropout=dropout,
                    use_nd_dropout=use_nd_dropout,
                    )
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(dims, pre_channel)) # upsamples resolution by factor 2 + conv., but channels stay the same
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(dims, pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x):

        feats = []
        for layer in self.downs:
            x = layer(x)
            feats.append(x) # for skip-connection

        for layer in self.mid:
            x = layer(x)

        for layer in self.ups:
            if isinstance(layer, Block):
                x = layer(torch.cat((x, feats.pop()), dim=1)) # for skip-connection
            else:
                x = layer(x)

        x = self.final_conv(x)
        
        return x
    
class My_UNet_3x(nn.Module):
    def __init__(
        self,
        dims=3, # SR3: 3
        in_channel=1, # SR3: 2
        out_channel=1, # SR3: 1
        inner_channel=2, # NOTE SR3 uses 64, we might not be able to do this ^^
        norm_groups=2,
        dropout=0.1, # SR3: 0.1
        image_size=32, # SR3: 32
        use_nd_dropout=False,
    ):
        super().__init__()
        self.dims = dims
        self.image_size = image_size

        feat_channels = []
        now_res = image_size
        downs = []
        
        feat_channels.append(in_channel)
        
        # Downsample
        downs.append(nn.Conv3d(in_channel, inner_channel, 3, 2, 1, padding_mode = "replicate")) # uses conv. to downsample res by 2 but can increase number of channels!
        now_res = now_res // 2
        
        # Activation
        downs.append(nn.BatchNorm3d(inner_channel))
        downs.append(Swish())

        feat_channels.append(inner_channel)
        # Downsample
        downs.append(nn.Conv3d(inner_channel, inner_channel, 3, 2, 1, padding_mode = "replicate")) # uses conv. to downsample res by 2 but can increase number of channels!
        now_res = now_res // 2
        
        # Activation
        downs.append(nn.BatchNorm3d(inner_channel))
        downs.append(Swish())

        feat_channels.append(inner_channel)
        # Downsample
        downs.append(nn.Conv3d(inner_channel, inner_channel, 3, 2, 1, padding_mode = "replicate")) # uses conv. to downsample res by 2 but can increase number of channels!
        now_res = now_res // 2

        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList(
            [
                Block(
                    dims,
                    inner_channel,
                    inner_channel,
                    groups=norm_groups,
                    dropout=dropout,
                    use_nd_dropout=use_nd_dropout,
                )
            ]
        )

        ups = []

        # Upsample
        # ups.append(nn.Upsample(scale_factor=2, mode="nearest"))# upsamples resolution by factor 2, but NO conv
        ups.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners = True))# upsamples resolution by factor 2, but NO conv #NOTE: try trilinear upsampling

        # Conv with residual
        ups.append(nn.Conv3d(inner_channel + feat_channels.pop(), inner_channel, 3, padding=1, padding_mode = "replicate"))

        now_res = now_res * 2

        # Activation
        downs.append(nn.BatchNorm3d(inner_channel))
        downs.append(Swish())

        # Upsample
        # ups.append(nn.Upsample(scale_factor=2, mode="nearest"))# upsamples resolution by factor 2, but NO conv
        ups.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners = True))# upsamples resolution by factor 2, but NO conv #NOTE: try trilinear upsampling

        # Conv with residual
        ups.append(nn.Conv3d(inner_channel + feat_channels.pop(), inner_channel, 3, padding=1, padding_mode = "replicate"))

        now_res = now_res * 2

        # Activation
        downs.append(nn.BatchNorm3d(inner_channel))
        downs.append(Swish())

        # Upsample
        # ups.append(nn.Upsample(scale_factor=2, mode="nearest"))# upsamples resolution by factor 2, but NO conv
        ups.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners = True))# upsamples resolution by factor 2, but NO conv #NOTE: try trilinear upsampling

        # Conv with residual
        ups.append(nn.Conv3d(inner_channel + feat_channels.pop(), out_channel, 3, padding=1, padding_mode = "replicate"))

        now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

    def forward(self, x):

        feats = []
        
        for layer in self.downs:
            if isinstance(layer, nn.Conv3d):
                feats.append(x) # for skip-connection
            x = layer(x)

        for layer in self.mid:
            x = layer(x)

        for layer in self.ups:
            if isinstance(layer, nn.Conv3d):
                x = layer(torch.cat((x, feats.pop()), dim=1)) # for skip-connection
            else:
                x = layer(x)

        return x