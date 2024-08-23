"""A library to create fully connected neural networks.

Copyright (C) 2024 Exai Bio Inc. Authors

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import collections
from typing import Optional

from absl import logging
from torch import nn


def create_fully_connected_network(
    layers_dim: list[int],
    bias: bool = True,
    activation: Optional[str] = None,
    activation_args: Optional[dict] = None,
    add_batch_normalization: bool = False,
    add_layer_normalization: bool = False,
    dropout_rate: float = 0,
    no_activation_on_output_layer: bool = False,
    predictor_batch_norm: bool = False,
) -> nn.Sequential:
    """Creates a fully connected neural network.

    Args:
        layers_dim: A list of integers representing the number of neurons in
            each layer. This includes the input and output layers.  For example,
            [100, 50, 10] creates a network with two layers, where the input
            dimension is 100 and the output dimension is 10 and there is one
            hidden layer with dimension 50.
        bias: Whether to add a bias term to the linear layers.
        activation: The name of non-linear activation layer to apply on linear
            layers. This name should be the exact class name in `torch.nn`. If
            the activation is not set or is None, we apply `ReLU`. Note that
            without a non-linear activation function, the network is just a
            linear function, so we always apply an activation function. In
            cases, that we need the full range in output, you can disable the
            application of activation on the output layer by setting
            `no_activation_on_output_layer` argument below.
        activation_args: If it is not `None`, will be passed to the activation
            layer.
        add_batch_normalization: Whether to add batch normalization to the
            layers.
        add_layer_normalization: Whether to add layer normalization to the
            layers.
        dropout_rate: The dropout rate to use on the layers. If it is 0, no
            dropout is added.
        no_activation_on_output_layer: Whether to add an activation function
            to the output layer. If you need the full real range, you should
            set this to `True`, because most activations are bounded e.g. `ReLU`
            only outputs non-negative values.
        predictor_batch_norm: Whether to add batch normalization as the first step

    Returns:
        A fully connected neural network.

    Raises:
        ValueError: If the number of layers is less than 2.
    """
    if len(layers_dim) < 2:
        raise ValueError(
            f"The number of layers must be at least 2. {layers_dim=}"
        )

    if add_batch_normalization and add_layer_normalization:
        logging.warning(
            "Both batch and layer normalization are added. This is not a "
            "common practice for a fully-connected network and is not "
            "recommended."
        )

    if activation is None:
        activation = "ReLU"
    activation_class = getattr(nn, activation)
    activation_layer = activation_class()
    if activation_args is not None:
        activation_layer = activation_class(**activation_args)

    layers = []
    if predictor_batch_norm:
        layers.append(
            ("predictor_batch_norm",
             nn.BatchNorm1d(
                layers_dim[0], momentum=0.01, eps=0.001)))
    for i in range(len(layers_dim) - 1):
        in_features = layers_dim[i]
        out_features = layers_dim[i + 1]
        sub_layers = [nn.Linear(in_features, out_features, bias=bias)]

        if add_batch_normalization:
            sub_layers.append(
                nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001)
            )
        if add_layer_normalization:
            sub_layers.append(
                nn.LayerNorm(out_features, elementwise_affine=False)
            )

        if i < len(layers_dim) - 2 or not no_activation_on_output_layer:
            # In the output layer, `i == len(layers_dim) - 2`.
            sub_layers.append(activation_layer)

        if dropout_rate:
            sub_layers.append(nn.Dropout(p=dropout_rate))

        layers.append((f"fc_layer_{i}", nn.Sequential(*sub_layers)))
    return nn.Sequential(collections.OrderedDict(layers))
