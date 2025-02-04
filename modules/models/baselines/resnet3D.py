# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from modules.modules.diffusion import conv_nd

#######################################################################
#######################################################################
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm: bool = True,
        num_groups: int = 1,
        dim: int = 3,
    ):
        super().__init__()
        self.conv1 = conv_nd(dim, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.bn1 = nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity()
        self.conv2 = conv_nd(dim, planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(num_groups, num_channels=planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_nd(dim, in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, self.expansion * planes) if norm else nn.Identity(),
            )

        self.activation = nn.GELU()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out = self.activation(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = out + self.shortcut(x)
        # out = self.activation(out)
        return out


class DilatedBasicBlock(nn.Module):
    """Basic block for Dilated ResNet

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        activation (str, optional): activation function. Defaults to "relu".
        norm (bool, optional): whether to use group normalization. Defaults to True.
        num_groups (int, optional): number of groups for group normalization. Defaults to 1.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm: bool = True,
        num_groups: int = 1,
        dim: int = 3,
    ):
        super().__init__()

        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        for dil in self.dilation:
            dilation_layers.append(
                conv_nd(
                    dim,
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity() for dil in self.dilation
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.dilation_layers, self.norm_layers):
            out = self.activation(layer(norm(out)))
        return out + x


class ResNet(nn.Module):
    """Class to support ResNet like feedforward architectures

    Args:
        n_input_scalar_components (int): Number of input scalar components in the model
        n_input_vector_components (int): Number of input vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        block (Callable): BasicBlock or DilatedBasicBlock or FourierBasicBlock
        num_blocks (List[int]): Number of blocks in each layer
        time_history (int): Number of time steps to use in the input
        time_future (int): Number of time steps to predict in the output
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
    """

    padding = 9

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        block,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int = 64,
        norm: bool = True,
        diffmode: bool = False,
        usegrid: bool = False,
        cond_channels: int = 0,
        dim: int = 3,
    ):
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.diffmode = diffmode
        self.usegrid = usegrid
        self.in_planes = hidden_channels
        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 3)
        if self.usegrid:
            insize += 2
        self.conv_in1 = conv_nd(
            dim,
            insize,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = conv_nd(
            dim,
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = conv_nd(
            dim,
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = conv_nd(
            dim,
            self.in_planes,
            time_future * (self.n_output_scalar_components + self.n_output_vector_components * 3),
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    num_blocks[i],
                    stride=1,
                    norm=norm,
                    dim=dim,
                )
                for i in range(len(num_blocks))
            ]
        )
        self.activation = nn.GELU()

        if cond_channels > 0:
            self.cond_emb = nn.Linear(cond_channels, hidden_channels)

    def _make_layer(
        self,
        block: Callable,
        planes: int,
        num_blocks: int,
        stride: int,
        norm: bool = True,
        dim: int = 3,   
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    norm=norm,
                    dim=dim,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def __repr__(self):
        return "ResNet"

    def forward(self, x: torch.Tensor, emb=None) -> torch.Tensor:
        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        if emb is not None:
            emb = self.cond_emb(emb)
            while emb.dim() < x.dim():
                emb = emb.unsqueeze(-1)

        for layer in self.layers:
            x = layer(x)
            if emb is not None:
                x = x + emb

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        return x

#######################################################################
#######################################################################