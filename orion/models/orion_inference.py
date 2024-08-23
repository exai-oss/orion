"""A library to create the orion inference model.

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

import json
import pathlib
from typing import Any, Optional, Literal

import torch
from orion.models import base_model, fully_connected
from torch import distributions, nn

# Filepath suffixes for saving and loading Orion inference model.
ORION_MODEL_CONFIG_FILEPATH_SUFFIX = "_model_config.json"
ORION_MODEL_STATE_DICT_FILEPATH_SUFFIX = "_model_state_dict.pt"


class VariationalEncoder(nn.Module):
    """Defines a variational encoder network.

    For a quick introduction to variational autoencoders, see:
    https://en.wikipedia.org/wiki/Variational_autoencoder

    For a more detailed introduction, see:
    https://arxiv.org/abs/1606.05908

    Args:
        input_dim: Input dimension.
        latent_dim: Latent space dimension.
        hidden_layer_num: Number of fully-connected hidden layers.
        hidden_dim: Hidden layer dimension.
        fully_connected_kwargs: Optional arguments for the fully connected
            network created with `create_fully_connected_network`. All these
            optional arguments have default values, so this is optional. Do not
            provide `layers_dim` here, as it is automatically set based on the
            above required arguments.

    Raises:
        ValueError: If `hidden_layer_num` is not positive.
    """

    _MIN_LOGVAR = -6
    _MAX_LOGVAR = 6

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_layer_num: int = 1,
        hidden_dim: int = 128,
        **fully_connected_optional_kwargs,
    ):
        super().__init__()

        if hidden_layer_num < 1:
            raise ValueError(f"{hidden_layer_num=} must be greater than 0.")

        # Note there is an additional output layer for the mean and variance,
        # so the last layer here is a hidden layer of the final encoder.
        layers_dim = [input_dim] + [hidden_dim] * hidden_layer_num
        self._shared_encoder = fully_connected.create_fully_connected_network(
            layers_dim=layers_dim,
            **fully_connected_optional_kwargs,
        )
        self._mean_encoder = nn.Linear(hidden_dim, latent_dim)
        self._logvar_encoder = nn.Linear(hidden_dim, latent_dim)

    def get_deterministic_encoder_network(self) -> nn.Sequential:
        """Returns a deterministic encoder network.

        The variational encoder provides a distribution over the latent space,
        but to get a single encoding for an input (i.e. the deterministic latent
        vector), the maximum likelihood is the mean of the distribution and we
        return the network corresponding to this mean.
        """
        return nn.Sequential(self._shared_encoder, self._mean_encoder)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The forward computation for a single sample.

        This encodes the input into the latent space by computing the
        mean and variance of the normal latent distribution. Then, applies the
        reparametrization trick and creates an actual sample of the
        distribution. See the references in the class docstring for more
        details.

        Args:
            x: Input data.

        Returns:
            A 3-tuple of tensors for the mean and variance of the normal
            distribution representing the latent space and an actual sample of
            this distribution.
        """
        # Calculate the mean and variance for the latent distribution, `q`.
        intermediate_encoding = self._shared_encoder(x)

        q_mean = self._mean_encoder(intermediate_encoding)

        logvar = self._logvar_encoder(intermediate_encoding)
        # When the weights of `logvar` encoder blow up, the log of variance can
        # be negative/positive infinity and the variance becomes zero or
        # infinity that can create issues with sampling in the next step.
        # To avoid this, we clamp it to a reasonable range.
        logvar = torch.clamp(logvar, min=self._MIN_LOGVAR, max=self._MAX_LOGVAR)
        q_var = torch.exp(logvar)

        # Sample from latent distribution.
        # Note the "r" in `rsample` stands for "reparameterized". The
        # parameterized random variable can be constructed via a parameterized
        # deterministic function of a parameter-free random variable. The
        # reparameterized sample therefore becomes differentiable.
        latent = distributions.Normal(q_mean, q_var.sqrt()).rsample()

        return q_mean, q_var, latent


class OrionModelConfig(base_model.ModelConfig):
    """Defines the configuration for the Orion model.

    The Orion model is a shell that combines the Orion inference sub-models and
    does not have any new parameters of its own. There are three sub-models:

    *   A variational encoder that converts the oncRNA expression values into an
        encoding.
    *   An optional variational encoder that converts a vector of sample
        features into a normalizing scalar. This scalar is multiplied to the
        sample oncRNA encoding to create a normalized encoding. This is to be
        robust against the overall oncRNA content variation. For example, in the
        first Orion model for the Sage project, this scalar is based on the
        miRNA content of a sample and the features are miRNA expression values.
    *   A fully connected network that predicts the cancer from the normalized
        encoding.

    We use a simple nested dictionary of built-in types for simplicity. To save
    and load the config we use the JSON language.

    Args:
        oncrna_encoder_args: Arguments for a variational encoder for the
            oncRNA data. This will be passed to `VariationalEncoder`.
        normalizing_scalar_args: Arguments for a variational encoder for the
            normalizing scalar. This will be passed to `VariationalEncoder`.
            Since this network encodes a scalar, the `latent_dim` argument must
            be 1. This parameter is optional and if it is `None`, there is no
            normalizing scalar.
        cancer_predictor_args: Arguments for a fully connected network that
            predicts cancer from (normalized) oncRNA encoding. This will be
            passed to `create_fully_connected_network`.
    """

    # Config dictionary keys.
    _ONCRNA_ENCODER_ARGS = "oncrna_encoder_args"
    _NORMALIZING_SCALAR_ARGS = "normalizing_scalar_args"
    _CANCER_PREDICTOR_ARGS = "cancer_predictor_args"

    def __init__(
        self,
        oncrna_encoder_args: dict[str, Any],
        normalizing_scalar_args: dict[str, Any],
        cancer_predictor_args: dict[str, Any],
    ):
        self._config = {
            self._ONCRNA_ENCODER_ARGS: oncrna_encoder_args,
            self._NORMALIZING_SCALAR_ARGS: normalizing_scalar_args,
            self._CANCER_PREDICTOR_ARGS: cancer_predictor_args,
        }
        super().__init__()

    def __eq__(self, value) -> bool:
        return self._config == value._config

    @property
    def oncrna_encoder_args(self) -> dict[str, Any]:
        return self._config[self._ONCRNA_ENCODER_ARGS]

    @property
    def normalizing_scalar_args(self) -> dict[str, Any]:
        return self._config[self._NORMALIZING_SCALAR_ARGS]

    @property
    def cancer_predictor_args(self) -> dict[str, Any]:
        return self._config[self._CANCER_PREDICTOR_ARGS]

    def validate(self) -> None:
        """Validate the Orion config has the expected structure and values.

        Raises:
            InvalidModelConfigException: If the config is not valid.
        """
        if (
            "latent_dim" not in self.normalizing_scalar_args
            or self.normalizing_scalar_args["latent_dim"] != 1
        ):
            raise base_model.InvalidModelConfigException(
                f"{self.normalizing_scalar_args=} must have `latent_dim`=1."
            )

    def save(self, config_filepath: pathlib.Path) -> None:
        """Saves the model config to a file."""
        with open(config_filepath, "w", encoding="utf-8") as f:
            json.dump(self._config, f)

    @classmethod
    def load(cls, config_filepath: pathlib.Path) -> "OrionModelConfig":
        """Loads the model config from a file."""
        with open(config_filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Note the init arguments are the same as the constant keys used in the
        # config.
        return cls(**config)


class OrionInferenceModel(base_model.Model, nn.Module):
    """Defines the Orion inference model.

    The Orion model is a shell that combines the Orion inference sub-models and
    does not have any new parameters of its own. Read the docstring of the
    :meth:`exai.models.orion_inference.OrionModelConfig` class for more details
    on the sub-models.

    Args:
        config: Configuration for the Orion inference model.
    """

    def __init__(
        self,
        config: OrionModelConfig,
    ):
        base_model.Model.__init__(self, config)
        nn.Module.__init__(self)

        self._oncrna_encoder = VariationalEncoder(
            **self._config.oncrna_encoder_args  # type: ignore
        )
        self._normalizing_scalar = VariationalEncoder(
            **self._config.normalizing_scalar_args  # type: ignore
        )
        self._cancer_predictor = fully_connected.create_fully_connected_network(
            **self._config.cancer_predictor_args  # type: ignore
        )

    def forward(
        self,
        x_oncrna: torch.Tensor,
        x_scalar: Optional[torch.Tensor] = None,
        inject_lib_method: Literal["concat", "multiply"] = "multiply",
    ) -> torch.Tensor:
        """The forward computation for a single sample.

        Args:
            x_oncrna: OncRNA expression values.
            x_scalar: Data for the normalizing scalar if there is one.
            inject_lib_method: Method for injecting the normalizing scalar to
                the oncRNA encoding. If `concat`, the scalar is concatenated to
                the oncRNA encoding. If `multiply`, the scalar is multiplied to
                the oncRNA encoding.

        Returns:
            The predicted cancer score.
        """
        oncrna_encoding = (
            self._oncrna_encoder.get_deterministic_encoder_network()(x_oncrna)
        )

        if x_scalar is not None:
            # Note x_scalar is a vector of library size vector
            # it can be concatenated or multiplied to the oncrna encoding
            normalizing_scale = (
                self._normalizing_scalar.get_deterministic_encoder_network()(
                    x_scalar
                )
            )
            if inject_lib_method == "concat":
                normalized_encoding = torch.cat(
                    (oncrna_encoding, normalizing_scale), dim=1
                )
            elif inject_lib_method == "multiply":
                normalized_encoding = torch.multiply(
                    oncrna_encoding, normalizing_scale
                )
        else:
            normalized_encoding = oncrna_encoding

        return self._cancer_predictor(normalized_encoding)

    def get_submodels(self) -> tuple[nn.Module, ...]:
        """Returns the sub-models of the Orion inference model.

        Returns:
            A 3-tuple of:
                - the variational encoder for the oncRNA data,
                - the variational encoder for the normalizing scalar,
                - the fully connected network that predicts cancer.
        """
        return (
            self._oncrna_encoder,
            self._normalizing_scalar,
            self._cancer_predictor,
        )

    def save(self, model_files_prefix: pathlib.Path):
        """Saves the model to files."""
        p = model_files_prefix
        config_filepath = p.parent / (
            p.name + ORION_MODEL_CONFIG_FILEPATH_SUFFIX
        )
        state_dict_filepath = p.parent / (
            p.name + ORION_MODEL_STATE_DICT_FILEPATH_SUFFIX
        )

        self._config.save(config_filepath)
        torch.save(self.state_dict(), state_dict_filepath)

    @classmethod
    def load(cls, model_files_prefix: pathlib.Path) -> "OrionInferenceModel":
        """Loads the model from files."""
        p = model_files_prefix
        config_filepath = p.parent / (
            p.name + ORION_MODEL_CONFIG_FILEPATH_SUFFIX
        )
        if not config_filepath.exists():
            raise FileNotFoundError(f"{config_filepath} does not exist.")
        state_dict_filepath = p.parent / (
            p.name + ORION_MODEL_STATE_DICT_FILEPATH_SUFFIX
        )
        if not state_dict_filepath.exists():
            raise FileNotFoundError(f"{state_dict_filepath} does not exist.")

        config = OrionModelConfig.load(config_filepath)
        model = cls(config=config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load(state_dict_filepath, map_location=device)
        )
        return model
