"""Module providing ModelParams class and OrionConfig dataclass

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

import inspect
from typing import Dict, Optional, List, Union, Any, Literal
import dataclasses

import torch

import numpy as np


@dataclasses.dataclass
class OrionConfig:
    """Stores key parameters of Orion:
    n_input: number of input features
    n_input_lib: number of input library features
    n_hidden: number of hidden units
    n_features: number of features
    num_epochs: number of epochs
    rng: random number generator
    random_seed: random seed
    dp: dropout rate
    l1: l1 regularization
    l2: l2 regularization
    grad_clip_max: gradient clipping maximum
    loss_scalers: list of loss scalers
    lr: learning rate
    num_lvs: number of latent variables
    n_layers: number of layers
    mixed_precision: whether to use mixed precision
    inject_lib: whether to inject library features
    inject_lib_method: How to inject the library features, as a multiplicative
        factor or as a concatenation to the latent space.
    mini_batch: mini batch size
    tm_rounds: number of rounds for tm
    weight_sample_loss: whether to weight the sample loss
    add_batchnorm_lib: adjust the last layer to use batchnorm - deprecated
    use_generative_sampling: whether to use generative sampling
    generative_samples_num: number of generative samples
    bag: whether to use bag of words
    num_classes: number of classes (supervised task)
    use_size_factor_key: whether to use size factor key
    use_triplet_loss: whether to use triplet loss
    use_augmentator: whether to use augmentator
    predict_classes: whether to predict cell types
    scaler: gradient scaler
    device: device to use
    use_best: whether to use the best model
    """

    n_input: int
    n_input_lib: int
    n_hidden: int
    n_features: int
    num_epochs: int
    rng: np.random.default_rng
    random_seed: int
    dp: float = 0.5
    l1: Optional[float] = None
    l2: float = 0.5
    grad_clip_max: float = 1
    # The following scalers are the default values for the loss scalers.
    # The order of the scalers is: reconstruction (both for MLE and kld_l),
    # kld_z, CE loss,
    # triplet margin loss, and regression based reconstruction loss
    loss_scalers: List[float] = dataclasses.field(
        default_factory=lambda: [10000, 1, 1, 1, 1]
    )  # List of loss scalers, cannot be initialized with a list.
    # It needs to be initialized with a lambda function. See:
    # https://stackoverflow.com/questions/52063759/
    # passing-default-list-argument-to-dataclasses
    lr: float = 0.001
    num_lvs: float = 10
    n_layers: int = 1
    mixed_precision: bool = False
    inject_lib: bool = True
    inject_lib_method: Literal["concat", "multiply"] = "multiply"
    mini_batch: int = 128
    tm_rounds: int = 4
    weight_sample_loss: bool = False
    add_batchnorm_lib: bool = False
    use_generative_sampling: bool = True
    generative_samples_num: int = 100
    bag: bool = False
    num_classes: int = 2
    use_size_factor_key: bool = False
    use_triplet_loss: bool = True
    use_augmentator: bool = False
    predict_classes: bool = True
    scaler: torch.cuda.amp.GradScaler = None
    device: torch.device = torch.device("cpu")
    use_best: bool = False
    add_regression_reconst: bool = False
    log_variational: bool = True
    gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb"

    def __getitem__(self, key: str):
        """Get an attribute by key."""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise AttributeError(f"OrionConfig has no attribute {key}")

    def __setitem__(self, key: str, value: Any):
        """Set an attribute by key."""
        return setattr(self, key, value)

    def get(self, key: str, default_value=None):
        """Get an attribute by key."""
        return getattr(self, key, default_value)

    def keys(self):
        """Get the keys of the dataclass."""
        return dataclasses.asdict(self).keys()

    def values(self):
        """Get the values of the dataclass."""
        return dataclasses.asdict(self).values()

    def items(self):
        """Get the items of the dataclass."""
        return dataclasses.asdict(self).items()


class ModelParams:
    """Stores key parameters of Orion and sets the attributes of the evaluator
    and trainer.

    Args:
        dict_params: Dictionary or Params dataclass of parameters.
        Defaults to None.

    Returns:
        ModelParams: ModelParams object
    """

    def __init__(
        self,
        dict_params: Optional[Union[Dict, OrionConfig]] = None,
        **params,
    ):
        if dict_params is None:
            dict_params = {}

        for key, val in params.items():
            setattr(self, key, val)
            dict_params[key] = val
        for key, val in dict_params.items():
            setattr(self, key, val)

    def print(self):
        attrs = inspect.getmembers(self)
        dict_params = {}
        for attr in attrs:
            if attr[0][0] not in ["<", "_"]:
                if str(attr[1])[0] not in ["<", "_"]:
                    dict_params[attr[0]] = attr[1]
        print(dict_params)
        return dict_params
