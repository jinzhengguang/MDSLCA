# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial
from typing import List

import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor, nn

from mmdet3d.models.layers.minkowski_engine_block import (
    IS_MINKOWSKI_ENGINE_AVAILABLE, MinkowskiBasicBlock, MinkowskiBottleneck,
    MinkowskiConvModule)
from mmdet3d.models.layers.sparse_block import (SparseBasicBlock,
                                                SparseBottleneck,
                                                make_sparse_convmodule,
                                                replace_feature)
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.models.layers.torchsparse_block import (TorchSparseBasicBlock,
                                                     TorchSparseBottleneck,
                                                     TorchSparseConvModule)
from mmdet3d.utils import OptMultiConfig

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse

if IS_MINKOWSKI_ENGINE_AVAILABLE:
    import MinkowskiEngine as ME


# from MinkowskiEngine.modules.senet_block import SELayer
class SELayerorigin(nn.Module):
    def __init__(self, channel, reduction=8, D=-1):
        # Global coords does not require coords_key
        super(SELayerorigin, self).__init__()
        self.fc = nn.Sequential(
            ME.MinkowskiLinear(channel, channel // reduction),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(channel // reduction, channel),
            ME.MinkowskiSigmoid()
        )
        self.pooling = ME.MinkowskiGlobalPooling()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        # Global coords does not require coords_key
        super(SELayer, self).__init__()
        self.convbnrelu = MinkowskiConvModule(channel, channel, 3)
        self.pooling = ME.MinkowskiGlobalPooling()
        self.fc = nn.Sequential(
            ME.MinkowskiLinear(channel, channel // reduction),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(channel // reduction, channel),
            ME.MinkowskiSigmoid()
        )
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        x = self.convbnrelu(x)
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class SEAttention(nn.Module):
    def __init__(self, channel, reduction=8, D=-1):
        super().__init__()
        self.fc = nn.Sequential(
            ME.MinkowskiLinear(channel, channel // reduction),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(channel // reduction, channel),
            ME.MinkowskiSigmoid()
        )
        self.pooling = ME.MinkowskiGlobalPooling()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class LocalSpatialAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        num_reduced_channels = 8
        self.conv1x1_1 = MinkowskiConvModule(in_channels=channel, out_channels=num_reduced_channels, kernel_size=1, dilation=1)
        self.conv1x1_2 = MinkowskiConvModule(in_channels=int(num_reduced_channels*5), out_channels=1, kernel_size=1, dilation=1)
        
        self.dilated_conv3x3 = MinkowskiConvModule(in_channels=num_reduced_channels, out_channels=num_reduced_channels, kernel_size=3, dilation=1)
        self.dilated_conv5x5 = MinkowskiConvModule(in_channels=num_reduced_channels, out_channels=num_reduced_channels, kernel_size=3, dilation=2)
        self.dilated_conv7x7 = MinkowskiConvModule(in_channels=num_reduced_channels, out_channels=num_reduced_channels, kernel_size=3, dilation=3)
        self.dilated_conv9x9 = MinkowskiConvModule(in_channels=num_reduced_channels, out_channels=num_reduced_channels, kernel_size=3, dilation=4)

    def forward(self, x):
        att = self.conv1x1_1(x)
        d1 = self.dilated_conv3x3(att)
        d2 = self.dilated_conv5x5(att)
        d3 = self.dilated_conv7x7(att)
        d4 = self.dilated_conv9x9(att)
        att = ME.cat(att, d1, d2, d3, d4)
        att = self.conv1x1_2(att)
        output = x * att
        return output


class ResidualPathME(nn.Module):
    def __init__(self, inc, outc, stride=1, dilation=1):
        super().__init__()
        self.basicconv = nn.Sequential(
            ME.MinkowskiConvolution(inc, outc, kernel_size=3, stride=stride, dilation=dilation, dimension=3),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )
        self.seatt = SEAttention(channel=outc)
        self.spatt = LocalSpatialAttention(channel=outc)
        self.convbnrelu = nn.Sequential(
            ME.MinkowskiConvolution(inc, outc, kernel_size=3, stride=stride, dilation=dilation, dimension=3),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        # out = self.relu(self.seatt(self.basicconv(x)) + x)
        # Squeeze-and-excitation networks, 2018 CVPR
        # All the attention you need: Global-local, spatial-channel attention for image retrieval, 2022 CVPR
        # Dual attention network for scene segmentation, 2019 CVPR
        # 2025-02-19 Jinzheng Guang
        x = self.basicconv(x)
        att = x + self.seatt(x) + self.spatt(x)
        out = self.convbnrelu(att)
        return out


class ResidualPath(nn.Module):
    def __init__(self, inc, outc, stride=1, dilation=1):
        super().__init__()
        self.basicconv = nn.Sequential(
            torchsparse.nn.Conv3d(inc, outc, kernel_size=3, dilation=dilation, stride=stride),
            torchsparse.nn.BatchNorm(outc),
            torchsparse.nn.ReLU(True)
        )
        self.catt = ChannelAttention(channel=outc)
        self.relu = torchsparse.nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.catt(self.basicconv(x)) + x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # 2023-03-21 Jinzheng Guang CBAM attention
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spnnavg = torchsparse.nn.GlobalAvgPool()
        self.spnnmax = torchsparse.nn.GlobalMaxPool()

    def forward(self, x):
        xtemp = x.F.contiguous().permute(1, 0)
        # maxp = self.spnnmax(x)
        avgp = self.spnnavg(x)
        max_result = self.maxpool(xtemp)
        avg_result = self.avgpool(avgp.permute(1,0))
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        outputs = xtemp * output
        x.F = outputs.permute(1, 0)
        return x


@MODELS.register_module()
class MinkUNetBackbone(BaseModule):
    r"""MinkUNet backbone with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        in_channels (int): Number of input voxel feature channels.
            Defaults to 4.
        base_channels (int): The input channels for first encoder layer.
            Defaults to 32.
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        encoder_channels (List[int]): Convolutional channels of each encode
            layer. Defaults to [32, 64, 128, 256].
        encoder_blocks (List[int]): Number of blocks in each encode layer.
        decoder_channels (List[int]): Convolutional channels of each decode
            layer. Defaults to [256, 128, 96, 96].
        decoder_blocks (List[int]): Number of blocks in each decode layer.
        block_type (str): Type of block in encoder and decoder.
        sparseconv_backend (str): Sparse convolutional backend.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 num_stages: int = 4,
                 encoder_channels: List[int] = [32, 64, 128, 256],
                 encoder_blocks: List[int] = [2, 2, 2, 2],
                 decoder_channels: List[int] = [256, 128, 96, 96],
                 decoder_blocks: List[int] = [2, 2, 2, 2],
                 block_type: str = 'basic',
                 sparseconv_backend: str = 'torchsparse',
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        assert sparseconv_backend in [
            'torchsparse', 'spconv', 'minkowski'
        ], f'sparseconv backend: {sparseconv_backend} not supported.'
        self.num_stages = num_stages
        self.sparseconv_backend = sparseconv_backend
        if sparseconv_backend == 'torchsparse':
            assert IS_TORCHSPARSE_AVAILABLE, \
                'Please follow `get_started.md` to install Torchsparse.`'
            input_conv = TorchSparseConvModule
            encoder_conv = TorchSparseConvModule
            decoder_conv = TorchSparseConvModule
            residual_block = TorchSparseBasicBlock if block_type == 'basic' \
                else TorchSparseBottleneck
            # for torchsparse, residual branch will be implemented internally
            residual_branch = None

            # 2025-02-14 Jinzheng Guang
            self.path = nn.ModuleList([
                nn.Sequential(
                    ResidualPath(encoder_channels[2], encoder_channels[2])
                ),
                nn.Sequential(
                    ResidualPath(encoder_channels[1], encoder_channels[1]),
                    ResidualPath(encoder_channels[1], encoder_channels[1])
                ),
                nn.Sequential(
                    ResidualPath(encoder_channels[0], encoder_channels[0]),
                    ResidualPath(encoder_channels[0], encoder_channels[0]),
                    ResidualPath(encoder_channels[0], encoder_channels[0])
                ),
                nn.Sequential(
                    ResidualPath(encoder_channels[0], encoder_channels[0]),
                    ResidualPath(encoder_channels[0], encoder_channels[0]),
                    ResidualPath(encoder_channels[0], encoder_channels[0]),
                    ResidualPath(encoder_channels[0], encoder_channels[0])
                )
            ])

        elif sparseconv_backend == 'spconv':
            if not IS_SPCONV2_AVAILABLE:
                warnings.warn('Spconv 2.x is not available,'
                              'turn to use spconv 1.x in mmcv.')
            input_conv = partial(
                make_sparse_convmodule, conv_type='SubMConv3d')
            encoder_conv = partial(
                make_sparse_convmodule, conv_type='SparseConv3d')
            decoder_conv = partial(
                make_sparse_convmodule, conv_type='SparseInverseConv3d')
            residual_block = SparseBasicBlock if block_type == 'basic' \
                else SparseBottleneck
            residual_branch = partial(
                make_sparse_convmodule,
                conv_type='SubMConv3d',
                order=('conv', 'norm'))
        elif sparseconv_backend == 'minkowski':
            assert IS_MINKOWSKI_ENGINE_AVAILABLE, \
                'Please follow `get_started.md` to install Minkowski Engine.`'
            input_conv = MinkowskiConvModule
            encoder_conv = MinkowskiConvModule
            decoder_conv = partial(
                MinkowskiConvModule,
                conv_cfg=dict(type='MinkowskiConvNdTranspose'))
            residual_block = MinkowskiBasicBlock if block_type == 'basic' \
                else MinkowskiBottleneck
            residual_branch = partial(MinkowskiConvModule, act_cfg=None)

            # from MinkowskiEngine.modules.senet_block import SELayer
            # self.se = SELayer(channel=256, reduction=16)
            # 2025-02-14 Jinzheng Guang
            self.path = nn.ModuleList([
                nn.Sequential(
                    ResidualPathME(encoder_channels[2], encoder_channels[2])
                ),
                nn.Sequential(
                    ResidualPathME(encoder_channels[1], encoder_channels[1]),
                    ResidualPathME(encoder_channels[1], encoder_channels[1])
                ),
                nn.Sequential(
                    ResidualPathME(encoder_channels[0], encoder_channels[0]),
                    ResidualPathME(encoder_channels[0], encoder_channels[0]),
                    ResidualPathME(encoder_channels[0], encoder_channels[0])
                ),
                nn.Sequential(
                    ResidualPathME(encoder_channels[0], encoder_channels[0]),
                    ResidualPathME(encoder_channels[0], encoder_channels[0]),
                    ResidualPathME(encoder_channels[0], encoder_channels[0]),
                    ResidualPathME(encoder_channels[0], encoder_channels[0])
                )
            ])

        self.conv_input = nn.Sequential(
            input_conv(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                indice_key='subm0'),
            input_conv(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                indice_key='subm0'))

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encoder_channels.insert(0, base_channels)
        decoder_channels.insert(0, encoder_channels[-1])

        for i in range(num_stages):
            encoder_layer = [
                encoder_conv(
                    encoder_channels[i],
                    encoder_channels[i],
                    kernel_size=2,
                    stride=2,
                    indice_key=f'spconv{i+1}')
            ]
            for j in range(encoder_blocks[i]):
                if j == 0 and encoder_channels[i] != encoder_channels[i + 1]:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i],
                            encoder_channels[i + 1],
                            downsample=residual_branch(
                                encoder_channels[i],
                                encoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{i+1}'))
                else:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i + 1],
                            encoder_channels[i + 1],
                            indice_key=f'subm{i+1}'))
            self.encoder.append(nn.Sequential(*encoder_layer))

            decoder_layer = [
                decoder_conv(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                    transposed=True,
                    indice_key=f'spconv{num_stages-i}')
            ]
            for j in range(decoder_blocks[i]):
                if j == 0:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1] + encoder_channels[-2 - i],
                            decoder_channels[i + 1],
                            downsample=residual_branch(
                                decoder_channels[i + 1] +
                                encoder_channels[-2 - i],
                                decoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{num_stages-i-1}'))
                else:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1],
                            decoder_channels[i + 1],
                            indice_key=f'subm{num_stages-i-1}'))
            self.decoder.append(
                nn.ModuleList(
                    [decoder_layer[0],
                     nn.Sequential(*decoder_layer[1:])]))

    def forward(self, voxel_features: Tensor, coors: Tensor) -> Tensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            Tensor: Backbone features.
        """
        if self.sparseconv_backend == 'torchsparse':
            x = torchsparse.SparseTensor(voxel_features, coors)
        elif self.sparseconv_backend == 'spconv':
            spatial_shape = coors.max(0)[0][1:] + 1
            batch_size = int(coors[-1, 0]) + 1
            x = SparseConvTensor(voxel_features, coors, spatial_shape,
                                 batch_size)
        elif self.sparseconv_backend == 'minkowski':
            x = ME.SparseTensor(voxel_features, coors)

        x = self.conv_input(x)
        laterals = [x]
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        decoder_outs = []
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer[0](x)

            if self.sparseconv_backend == 'torchsparse':
                x = torchsparse.cat((x, self.path[i](laterals[i])))
            elif self.sparseconv_backend == 'spconv':
                x = replace_feature(
                    x, torch.cat((x.features, laterals[i].features), dim=1))
            elif self.sparseconv_backend == 'minkowski':
                x = ME.cat(x, self.path[i](laterals[i]))

            x = decoder_layer[1](x)
            decoder_outs.append(x)

        if self.sparseconv_backend == 'spconv':
            return decoder_outs[-1].features
        else:
            return decoder_outs[-1].F
