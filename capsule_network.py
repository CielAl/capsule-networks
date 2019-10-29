"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829
PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
from typing import Union, Tuple
import sys
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

sys.setrecursionlimit(15000)


def softmax(inputs, dim=1):
    transposed_input = inputs.transpose(dim, len(inputs.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(inputs.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    NUM_ROUTING_ITERATIONS: int = 3

    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=None):
        super().__init__()
        if num_iterations is None:
            num_iterations = CapsuleLayer.NUM_ROUTING_ITERATIONS
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules: nn.ModuleList = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    # noinspection PyMethodMayBeStatic
    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        # noinspection PyTypeChecker
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes > 0:
            # x: N ,  [1],   num_route_nodes, [1],        VECTOR_DIM
            # W: [1]  CLASS, num_route_nodes, VECTOR_DIM, POSE_DIM)

            # priors: N, Class, num_nomdes, 1, POSE_DIM (VECTOR_DIM collapsed in mul, while other are fore
            # broadcasting)
            # todo memory consumption too huge
            priors = x[:, None, :, None, :] @ self.route_weights[None, :, :, :, :]

            logits = torch.zeros(*priors.size()).to(x.device)
            assert self.num_iterations > 0
            outputs = None
            for i in range(self.num_iterations):
                probabilities = softmax(logits, dim=2)
                outputs = self.squash((probabilities * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            # the original impl can work but is wrong
            # outputs = tuple(capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules)
            # capsule(x) returns N*VECTOR_DIM*H*W --> N*[H*W]*VECTOR_DIM
            # N*28*28*8
            stack_dim = -2
            outputs = tuple(capsule(x).permute([0, 2, 3, 1]).unsqueeze(dim=stack_dim)
                            for capsule in self.capsules)
            outputs = tuple(t.view(x.size(0), -1, t.size(-1)) for t in outputs)
            outputs = torch.cat(outputs, dim=stack_dim)
            # noinspection PyTypeChecker
            outputs = self.squash(outputs)

        return outputs


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
                ('pad', nn.ReflectionPad2d(padding=kernel_size//2)),
                ('conv', nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   bias=False,
                                   )
                 ),
                ('batchnorm', nn.BatchNorm2d(num_features=out_channels)),
                ('relu', nn.ReLU(inplace=True))
                ]
            ))

    def forward(self, x):
        return self.features(x)


class ResizeConv(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 target_size: Union[int, Tuple[int, ...]] = None,
                 scale_factor: Union[float, Tuple[float, ...]] = None,
                 mode: str = 'bilinear'):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
                ('up', nn.Upsample(size=target_size, scale_factor=scale_factor, mode=mode, align_corners=True)),
                ]
            ))
        self.features.add_module('conv_block',
                                 BasicBlock(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=1))

    def forward(self, x):
        return self.features(x)


class ConvDecoder(nn.Module):

    @staticmethod
    def size_fixation(length: int):
        num_downsample = int(np.ceil(np.log2(length)))
        size_recover = 2**num_downsample
        factor = length/size_recover
        return factor

    def __init__(self,
                 patch_shape,
                 in_channels,
                 kernel_size: int = 3):
        super().__init__()
        height, width, final_out_channel = patch_shape

        # for convenience and sake of earlier torch version, assume the fixation factor is homogeneous.
        height_f = ConvDecoder.size_fixation(height)
        weight_f = ConvDecoder.size_fixation(width)
        assert height_f == weight_f, f"Inconsistent fixation {[height_f, weight_f]}. {[height, width]}"
        num_upsample = int(np.ceil(np.log2(height)))
        map_channel = final_out_channel * (2**num_upsample)
        self.features = nn.Sequential()
        init_mapping = BasicBlock(in_channels=in_channels, out_channels=map_channel,
                                  kernel_size=1, stride=1)
        self.features.add_module('init', init_mapping)

        in_channels_upsample = map_channel
        for x in range(num_upsample):
            resize_block = ResizeConv(in_channels=in_channels_upsample,
                                      out_channels=in_channels_upsample//2,
                                      kernel_size=kernel_size,
                                      scale_factor=2)
            self.features.add_module(f'resize_{x}', resize_block)
            in_channels_upsample //= 2
        if height_f == 1:
            return
        fixation = ResizeConv(in_channels=in_channels_upsample,
                              out_channels=in_channels_upsample//2,
                              kernel_size=1,
                              scale_factor=height_f)
        self.features.add_module(f"fixation", fixation)

    def forward(self, x):
        return self.features(x)


class ConvEncoder(nn.Module):

    @staticmethod
    def bn_block(in_channel: int,
                 out_channel: int,
                 bn_channel: int = 32,
                 bottom_stride: int = 2) -> nn.Module:
        """
        Borrowed from torchvision. Do not use that for the baseline
        Args:
            in_channel ():
            out_channel ():
            bn_channel ():
            bottom_stride ():

        Returns:

        """
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channel, out_channels=bn_channel, kernel_size=1, stride=1, padding=0, bias=True)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(bn_channel, out_channels=bn_channel, stride=1, kernel_size=3, padding=1,
                                bias=True)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(bn_channel, out_channels=out_channel, kernel_size=1, bias=True,
                                stride=bottom_stride)),
            ('relu3', nn.ReLU(inplace=True))
         ]
        ))

    @staticmethod
    def down_path(patch_shape: Tuple[int, int, int],
                  init_feature_power: int = 5,
                  init_kernel_size: int = 7,
                  path_kernel_size: int = 3,
                  depth: int = 4):
        """

        Args:
            patch_shape ():
            init_feature_power ():
            init_kernel_size ():
            path_kernel_size ():
            depth (): Includes 1st layer.

        Returns:

        """
        assert len(patch_shape) == 3, 'Patch Shape must be 3d. Add singleton dimension if required.'
        assert depth > 0, f'Positive Depth{depth}'

        height, width, in_channels = patch_shape
        kernel_size_list = (init_kernel_size, ) + (path_kernel_size, ) * (depth - 1)
        # layer id s
        out_channels = 2**init_feature_power
        features = nn.Sequential()
        for idx, kernel in enumerate(kernel_size_list):
            features.add_module(f"block_{idx}", BasicBlock(in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           kernel_size=kernel))
            in_channels = out_channels
            out_channels *= 2
        # features and final # of channels
        return features, in_channels

    @staticmethod
    def _init_block_bn(patch_shape: Tuple[int, int, int],
                       init_feature: int = 32,
                       growth_rate: int = 32,
                       init_kernel_size: int = 7,
                       init_config: Tuple[int, ...] = (1, 1)
                       ):
        """
        Old approach. Redundant conv feature extractor
        Args:
            patch_shape ():
            init_feature ():
            growth_rate ():
            init_kernel_size ():
            init_config ():

        Returns:

        """
        assert len(patch_shape) == 3, 'Patch Shape must be 3d. Add singleton dimension if required.'
        height, width, in_channels = patch_shape
        features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, init_feature, kernel_size=init_kernel_size, stride=2,
                                padding=init_kernel_size // 2, bias=False)),
            # ('norm0', nn.BatchNorm2d(init_feature)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        num_in_channels = init_feature
        print(num_in_channels)
        for idx, num_layer in enumerate(init_config):
            assert num_layer > 0, '# layer must be positive'
            for ii in range(num_layer):
                if ii < num_layer - 1:
                    stride = 1
                else:
                    stride = 2
                btn_layer = CapsuleNet.bn_block(num_in_channels,
                                                num_in_channels + growth_rate,
                                                bn_channel=num_in_channels // 2,
                                                bottom_stride=stride)
                # print(btn_layer)
                features.add_module('bn_%d_%d' % (idx + 1, ii + 1), btn_layer)
                num_in_channels += growth_rate
        # print(num_in_channels)
        return features, num_in_channels

    def __init__(self,
                 patch_shape: Tuple[int, int, int],
                 init_feature_power: int = 5,
                 init_kernel_size: int = 7,
                 path_kernel_size: int = 3,
                 depth: int = 4):
        super().__init__()
        features, final_channel = ConvEncoder.down_path(
            patch_shape,
            init_feature_power,
            init_kernel_size=init_kernel_size,
            path_kernel_size=path_kernel_size,
            depth=depth
        )
        self.features = features
        self.final_channel = final_channel

    @property
    def depth(self):
        return len(self.features)

    def forward(self, x):
        return self.features(x)


class CapsuleNet(nn.Module):
    CONV_SIZE: int = 3
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def device(self):
        return self.__device

    def __init__(self, patch_shape: Tuple[int, int, int],
                 num_classes: int = 10,
                 init_feature_power: int = 5,
                 init_kernel_size: int = 7,
                 down_kernel_size: int = 3,
                 up_kernel_size: int = 3,
                 depth: int = 4,
                 primary_caps_num: int = 16,
                 primary_kernel_size: int = 3,
                 primary_kernel_num: int = 8,
                 pose_dim: int = 16,
                 ):
        super().__init__()
        height, width, in_channels = patch_shape

        # dimension-settings
        self._num_classes = num_classes
        self._pose_dim = pose_dim

        # prepare initial down-sampling blocks
        self.initial_conv = ConvEncoder(patch_shape=patch_shape,
                                        init_feature_power=init_feature_power,
                                        init_kernel_size=init_kernel_size,
                                        path_kernel_size=down_kernel_size,
                                        depth=depth
                                        )
        num_in_channels = self.initial_conv.final_channel
        self.primary_capsules = CapsuleLayer(num_capsules=primary_caps_num, num_route_nodes=-1,
                                             in_channels=num_in_channels,
                                             out_channels=primary_kernel_num,
                                             kernel_size=primary_kernel_size, stride=2)
        # extra division of 2 for primary caps
        # calculate width and height after down-sampling and primary kernel
        downsample_factor = 2**self.initial_conv.depth
        width_down = (width // downsample_factor - (primary_kernel_size-1)) // 2
        height_down = (height // downsample_factor - (primary_kernel_size-1)) // 2

        logger.debug(f"W:{width_down}, H{height_down}, C:{primary_caps_num}")
        assert width_down > 0 and height_down > 0, f"In correct size. Check depth. {width_down, height_down}"

        self.digit_capsules = CapsuleLayer(num_capsules=self.num_classes,
                                           num_route_nodes=primary_caps_num * width_down*height_down,
                                           in_channels=primary_kernel_num,
                                           out_channels=pose_dim)

        in_channels_decoder = self._pose_dim * self.num_classes
        self.decoder = ConvDecoder(patch_shape=patch_shape, in_channels=in_channels_decoder,
                                   kernel_size=up_kernel_size)

    def forward(self, x, y=None):
        x = self.initial_conv(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        batch_size = x.shape[0]
        x = x.squeeze()
        if batch_size < 1:
            x = x.unsqueeze(0)

        classes = (x ** 2).sum(dim=-1) ** 0.5  # probability as magnitude of vector (length)
        classes = F.softmax(classes, dim=-1)
        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = torch.eye(self.num_classes).to(x.device).index_select(dim=0, index=max_length_indices.data)
        decoder_input = (x * y[:, :, None]).view(batch_size, -1)
        # batch * (pose*class) * 1 * 1
        decoder_input = decoder_input[:, :, None, None]
        # logging.debug(f"\n{x.shape}|{y.shape}|{(x * y[:, :, None]).shape}|DecoderInput:{decoder_input.shape}")
        reconstructions = self.decoder(decoder_input)

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
