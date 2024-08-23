"""Module providing classes for Encoder/ Decoder network for VAEs.

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
from torch import nn
from orion.models import fully_connected


class Decoder(nn.Module):
    """
    Decodes data from latent space of ``n_input``
    dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Args:
        n_input: The dimensionality of the input (latent space)
        n_output: The dimensionality of the output (data space)
        n_cat_list: A list containing the number of categories
            for each category of interest. Each category will be
            included using a one-hot encoding
        n_layers: The number of fully-connected hidden layers
        n_hidden: The number of nodes per hidden layer
        dropout_rate: Dropout rate to apply to each of the hidden layers
        use_batch_norm: Whether to use batch norm in layers
        use_layer_norm: Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        add_regression_reconst: bool = False,
    ):
        super().__init__()
        self.add_regression_reconst = add_regression_reconst
        layers_dim = [n_input] + [n_hidden] * n_layers
        # px_decoder's last dimension will still be n_hidden
        self.px_decoder = fully_connected.create_fully_connected_network(
            layers_dim=layers_dim,
            add_batch_normalization=use_batch_norm,
            add_layer_normalization=use_layer_norm,
        )

        if self.add_regression_reconst:
            self.x_hat_decoder = nn.Linear(n_hidden, n_output)

        # px_scale_decoder can convert output of
        # px_decoder to reconstructed values
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )

        # dropout
        # px_dropout_decoder can convert output of
        # px_decoder to reconstructed values

        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
    ):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression

        Args:
            z: tensor with shape ``(n_input,)``
            library: library size
            cat_list:  list of category membership(s) for this sample

        Returns: 4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters
        # of the ZINB distribution
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        x_hat = None
        if self.add_regression_reconst:
            x_hat = self.x_hat_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to
        # avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = None
        return px_scale, px_r, px_rate, px_dropout, px, x_hat
