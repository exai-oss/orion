"""Defines the base configs for use in ML modeling and experimentation.

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
from typing import T, Type

import dataclasses_json
import marshmallow


class InvalidConfigException(Exception):
    """Defines an exception for an invalid config."""


class BaseConfig(abc.ABC):
    """Defines a generic config."""

    @abc.abstractmethod
    def validate(self) -> None:
        """Validate the config has the expected structure and values.

        Raises:
            InvalidModelConfigException: If the config is not valid.
        """
        raise InvalidConfigException("Validation method not implemented.")

    @abc.abstractmethod
    def save(self, config_filepath: pathlib.Path) -> None:
        """Saves the config to a file."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, config_filepath: pathlib.Path) -> "BaseConfig":
        """Loads the config from a file."""
        raise NotImplementedError


class DataclassJsonConfig(BaseConfig, dataclasses_json.DataClassJsonMixin):
    """Defines an abstract config, which is a dataclass serializable to JSON.

    To use this class, define a dataclass with the 'dataclasses.dataclass'
    decorator in addition to inheriting from this class.

    Example:

        .. code-block:: python

            import dataclasses
            @dataclasses.dataclass
            class MyConfig(DataclassJsonConfig):
                my_field: str

            config = MyConfig(my_field="my_value")
            config.save("my_config.json")
            reloaded_config = MyConfig.load("my_config.json")

    """

    def save(self, config_filepath: pathlib.Path) -> None:
        """Saves the model config to a file."""
        with open(config_filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls: Type[T], config_filepath: pathlib.Path) -> T:
        """Loads the config from a file.

        When we load the config from a file, we validate the schema of the
        config based on the dataclass definition using the 'schema' method of
        `dataclasses_json` (which uses `marshmallow` library under the hood).
        """
        with open(config_filepath, "r", encoding="utf-8") as f:
            config_str = f.read()
        try:
            config = cls.schema().loads(config_str)
        except marshmallow.exceptions.ValidationError as validation_error:
            raise InvalidConfigException(
                f"Schema validation error for config: {config_str}"
                f"Error: {validation_error}"
            ) from validation_error
        return config
