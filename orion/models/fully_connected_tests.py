"""Tests for exai.models.fully_connected.

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

import torch
from absl.testing import parameterized
from orion.models import fully_connected
from torch import nn


class CreateFullyConnectedNetworkTest(parameterized.TestCase):
    @parameterized.named_parameters(
        {
            "testcase_name": "one_layer",
            "layers_dim": [2, 4],
            "expected_num_layers": 1,
        },
        {
            "testcase_name": "two_layers",
            "layers_dim": [2, 5, 4],
            "expected_num_layers": 2,
        },
    )
    def test_creates_fully_connected_network_properly(
        self,
        layers_dim,
        expected_num_layers,
    ):
        network = fully_connected.create_fully_connected_network(
            layers_dim=layers_dim,
        )

        self.assertEqual(expected_num_layers, len(network))
        for layer in network:
            # A layer should always be a `nn.Sequential` layer.
            self.assertTrue(isinstance(layer, nn.Sequential))
            # The first sublayer should always be a `nn.Linear` layer.
            self.assertTrue(isinstance(layer[0], nn.Linear))

        batch_size = 3
        input_dim = layers_dim[0]
        output_dim = layers_dim[-1]
        input_data = torch.randn(batch_size, input_dim)
        output_data = network(input_data)
        self.assertEqual(
            output_data.shape, torch.Size([batch_size, output_dim])
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "no_dimensions",
            "layers_dim": [],
        },
        {
            "testcase_name": "one_dimension",
            "layers_dim": [2],
        },
    )
    def test_raises_exception_when_not_enough_dimensions(self, layers_dim):
        with self.assertRaisesRegex(
            ValueError, "The number of layers must be at least 2. layers_dim=.*"
        ):
            _ = fully_connected.create_fully_connected_network(
                layers_dim=layers_dim
            )

    @parameterized.named_parameters(
        {
            "testcase_name": "without_bias",
            "bias": False,
        },
        {
            "testcase_name": "with_bias",
            "bias": True,
        },
    )
    def test_sets_linear_layer_bias_correctly(self, bias):
        network = fully_connected.create_fully_connected_network(
            layers_dim=[2, 4],
            bias=bias,
        )
        self.assertEqual(bias, network[0][0].bias is not None)

    def test_sets_linear_layer_dimensions_correctly(self):
        network = fully_connected.create_fully_connected_network(
            layers_dim=[2, 4, 3, 5],
        )

        self.assertLen(network, 3)

        self.assertEqual(2, network[0][0].in_features)
        self.assertEqual(4, network[0][0].out_features)

        self.assertEqual(4, network[1][0].in_features)
        self.assertEqual(3, network[1][0].out_features)

        self.assertEqual(3, network[2][0].in_features)
        self.assertEqual(5, network[2][0].out_features)

    @parameterized.named_parameters(
        {
            "testcase_name": "all_sublayers_are_added",
            "activation": "LeakyReLU",
            "activation_args": {"negative_slope": 0.1},
            "batch_normalization": True,
            "layer_normalization": True,
            "dropout_rate": 0.1,
            "expected_layer0_types": [
                nn.Linear,
                nn.BatchNorm1d,
                nn.LayerNorm,
                nn.LeakyReLU,
                nn.Dropout,
            ],
            "expected_layer1_types": [
                nn.Linear,
                nn.BatchNorm1d,
                nn.LayerNorm,
                nn.LeakyReLU,
                nn.Dropout,
            ],
        },
        {
            "testcase_name": "no_optional_sublayer_is_added",
            "expected_layer0_types": [nn.Linear, nn.ReLU],
            "expected_layer1_types": [nn.Linear, nn.ReLU],
        },
        {
            "testcase_name": "no_activation_on_output_layer",
            "activation": "Tanh",
            "no_activation_on_output_layer": True,
            "expected_layer0_types": [nn.Linear, nn.Tanh],
            "expected_layer1_types": [nn.Linear],
        },
        {
            "testcase_name": "batch_normalization_is_added",
            "batch_normalization": True,
            "expected_layer0_types": [nn.Linear, nn.BatchNorm1d, nn.ReLU],
            "expected_layer1_types": [nn.Linear, nn.BatchNorm1d, nn.ReLU],
        },
        {
            "testcase_name": "layer_normalization_is_added",
            "layer_normalization": True,
            "expected_layer0_types": [nn.Linear, nn.LayerNorm, nn.ReLU],
            "expected_layer1_types": [nn.Linear, nn.LayerNorm, nn.ReLU],
        },
        {
            "testcase_name": "dropout_is_added",
            "dropout_rate": 0.1,
            "expected_layer0_types": [nn.Linear, nn.ReLU, nn.Dropout],
            "expected_layer1_types": [nn.Linear, nn.ReLU, nn.Dropout],
        },
    )
    def test_adds_sublayers_correctly(
        self,
        expected_layer0_types,
        expected_layer1_types,
        activation=None,
        activation_args=None,
        batch_normalization=False,
        layer_normalization=False,
        dropout_rate=0.0,
        no_activation_on_output_layer=False,
    ):
        network = fully_connected.create_fully_connected_network(
            layers_dim=[2, 3, 4],
            activation=activation,
            activation_args=activation_args,
            add_batch_normalization=batch_normalization,
            add_layer_normalization=layer_normalization,
            dropout_rate=dropout_rate,
            no_activation_on_output_layer=no_activation_on_output_layer,
        )

        self.assertLen(network, 2)

        layer0_types = [type(sl) for sl in network[0]]
        self.assertEqual(expected_layer0_types, layer0_types)

        layer1_types = [type(sl) for sl in network[1]]
        self.assertEqual(expected_layer1_types, layer1_types)
