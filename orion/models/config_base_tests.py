"""Tests for exai.models.config_base.

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

import dataclasses
import pathlib
import tempfile

from absl.testing import parameterized
from orion.models import config_base


@dataclasses.dataclass
class FakeConfig(config_base.DataclassJsonConfig):
    list_field: list[str]
    float_field: float = 0.0
    bool_field: bool = False

    def validate(self) -> None:
        pass


class DataclassJsonConfigTest(parameterized.TestCase):
    def test_saves_and_loads_properly(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = FakeConfig(list_field=["a", "b", "c"], float_field=0.5)
            config_filepath = pathlib.Path(tempdir) / "config.json"
            config.save(config_filepath)
            reloaded_config = FakeConfig.load(config_filepath)
            self.assertEqual(config, reloaded_config)

    @parameterized.named_parameters(
        {
            "testcase_name": "invalid_type_inside_list",
            "config_str": '{"list_field": ["a", 5]}',
        },
        {
            "testcase_name": "unknown_field",
            "config_str": '{"list_field": ["a", 5], "unknown_field": 10}',
        },
        {
            "testcase_name": "missing_field",
            "config_str": '{"float_field": 1.0, "bool_field": true}',
        },
    )
    def test_raises_invalid_config_error(self, config_str):
        with tempfile.TemporaryDirectory() as tempdir:
            config_filepath = pathlib.Path(tempdir) / "config.json"
            with open(config_filepath, "w", encoding="utf-8") as f:
                f.write(config_str)

            with self.assertRaises(config_base.InvalidConfigException):
                _ = FakeConfig.load(config_filepath)
