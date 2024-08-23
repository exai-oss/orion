"""Defines the base model and model config.

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

import abc
import pathlib


class InvalidModelConfigException(Exception):
    """Defines an exception for an invalid ML model config."""


class ModelConfig(abc.ABC):
    """Defines the base model config.

    To define a model, we need a configuration to determine the model
    architecture and the parameters to create the components of this
    architecture.

    """

    @abc.abstractmethod
    def validate(self) -> None:
        """Validate the config has the expected structure and values.

        Raises:
            InvalidModelConfigException: If the config is not valid.
        """
        raise InvalidModelConfigException("Validation method not implemented.")

    @abc.abstractmethod
    def save(self, config_filepath: pathlib.Path) -> None:
        """Saves the model config to a file."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, config_filepath: pathlib.Path) -> "ModelConfig":
        """Loads the model config from a file."""
        raise NotImplementedError


class Model(abc.ABC):
    """Defines the base ML model."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._config.validate()

    @abc.abstractmethod
    def save(self, model_files_prefix: pathlib.Path):
        """Saves the model to files."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, model_files_prefix: pathlib.Path) -> "Model":
        """Loads the model from files."""
        raise NotImplementedError
