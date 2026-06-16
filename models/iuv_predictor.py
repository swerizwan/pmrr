from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class IUV_predict_layer(nn.Module):
    def __init__(self, feat_dim=256, final_cov_k=3, part_out_dim=25, with_uv=True):
        """
        Initialization of the IUV prediction module.

        Args:
        - feat_dim (int): Dimensionality of the input feature maps.
        - final_cov_k (int): Kernel size for the convolution layers.
        - part_out_dim (int): Number of output channels for part segmentation.
        - with_uv (bool): Whether to include UV prediction (default is True).
        """
        super().__init__()

        self.with_uv = with_uv
        if self.with_uv:
            # UV prediction layers
            self.predict_u = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=part_out_dim,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0
            )

            self.predict_v = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=part_out_dim,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0
            )

        # Annotation index prediction layers
        self.predict_ann_index = nn.Conv2d(
            in_channels=feat_dim,
            out_channels=15,
            kernel_size=final_cov_k,
            stride=1,
            padding=1 if final_cov_k == 3 else 0
        )

        # UV index prediction layers
        self.predict_uv_index = nn.Conv2d(
            in_channels=feat_dim,
            out_channels=part_out_dim,
            kernel_size=final_cov_k,
            stride=1,
            padding=1 if final_cov_k == 3 else 0
        )

        self.inplanes = feat_dim

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Helper function to create a layer block.

        Args:
        - block: The residual block to be used.
        - planes (int): Number of planes (channels) in the block.
        - blocks (int): Number of residual blocks.
        - stride (int): Stride for the convolution layers.

        Returns:
        - nn.Sequential: Sequential container of layers.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - dict: Dictionary containing predicted outputs.
        """
        return_dict = {}

        # Predict UV and annotation indices
        predict_uv_index = self.predict_uv_index(x)
        predict_ann_index = self.predict_ann_index(x)

        return_dict['predict_uv_index'] = predict_uv_index
        return_dict['predict_ann_index'] = predict_ann_index

        if self.with_uv:
            # If UV prediction is enabled, predict U and V channels
            predict_u = self.predict_u(x)
            predict_v = self.predict_v(x)
            return_dict['predict_u'] = predict_u
            return_dict['predict_v'] = predict_v
        else:
            # If UV prediction is disabled, return None for U and V channels
            return_dict['predict_u'] = None
            return_dict['predict_v'] = None
            # Alternatively, initialize with zeros:
            # return_dict['predict_u'] = torch.zeros(predict_uv_index.shape).to(predict_uv_index.device)
            # return_dict['predict_v'] = torch.zeros(predict_uv_index.shape).to(predict_uv_index.device)

        return return_dict
