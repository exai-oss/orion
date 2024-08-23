"""Tests for exai.models.orion_inference.

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
import pathlib
import tempfile

import torch
from absl.testing import parameterized
from orion.models import base_model, orion_inference


def create_test_orion_config(
    x_oncrna_dim=100,
    oncrna_latent_dim=10,
    x_scalar_dim=50,
    scalar_latent_dim=1,
    num_classes=3,
    inject_lib_method="multiply",
):
    oncrna_encoder_args = {
        "input_dim": x_oncrna_dim,
        "latent_dim": oncrna_latent_dim,
        "hidden_layer_num": 1,
        "hidden_dim": 8,
        # fully_connected_optional_kwargs
        "dropout_rate": 0.1,
        "add_batch_normalization": True,
    }
    normalizing_scalar_args = {
        "input_dim": x_scalar_dim,
        "latent_dim": scalar_latent_dim,
        "hidden_layer_num": 1,
        "hidden_dim": 8,
    }

    if inject_lib_method == "concat":
        cancer_predictor_layer_dim_1 = oncrna_latent_dim + scalar_latent_dim
    elif inject_lib_method == "multiply":
        cancer_predictor_layer_dim_1 = oncrna_latent_dim
    else:
        raise ValueError(f"Invalid `inject_lib_method`: {inject_lib_method}.")
    cancer_predictor_args = {
        "layers_dim": [cancer_predictor_layer_dim_1, 8, num_classes],
        # fully_connected_optional_kwargs
        "activation": "LeakyReLU",
        "activation_args": {"negative_slope": 0.1},
        "no_activation_on_output_layer": True,
    }
    return orion_inference.OrionModelConfig(
        oncrna_encoder_args=oncrna_encoder_args,
        normalizing_scalar_args=normalizing_scalar_args,
        cancer_predictor_args=cancer_predictor_args,
    )


class VariationalEncoderTest(parameterized.TestCase):
    @parameterized.named_parameters(
        {
            "testcase_name": "without_fully_connected_keyword_args",
            "fully_connected_optional_kwargs": {},
        },
        {
            "testcase_name": "with_fully_connected_keyword_args",
            "fully_connected_optional_kwargs": {
                "add_batch_normalization": True
            },
        },
    )
    def test_creates_variational_encoder_properly(
        self, fully_connected_optional_kwargs
    ):
        variational_encoder = orion_inference.VariationalEncoder(
            input_dim=2,
            latent_dim=4,
            hidden_layer_num=1,
            hidden_dim=8,
            **fully_connected_optional_kwargs,
        )

        batch_size, input_dim, latent_dim = 3, 2, 4
        x = torch.randn(batch_size, input_dim)
        q_mean, q_var, latent = variational_encoder(x)
        self.assertEqual(q_mean.shape, (batch_size, latent_dim))
        self.assertEqual(q_var.shape, (batch_size, latent_dim))
        self.assertEqual(latent.shape, (batch_size, latent_dim))

    def test_raises_exception_with_no_hidden_layer(self):
        with self.assertRaisesWithLiteralMatch(
            ValueError, "hidden_layer_num=0 must be greater than 0."
        ):
            _ = orion_inference.VariationalEncoder(
                input_dim=2, latent_dim=4, hidden_layer_num=0
            )

    def test_gets_deterministic_encoder_properly(self):
        batch_size, input_dim = 3, 2

        variational_encoder = orion_inference.VariationalEncoder(
            input_dim=input_dim,
            latent_dim=4,
        )

        encoder = variational_encoder.get_deterministic_encoder_network()
        self.assertIsInstance(encoder, torch.nn.Module)

        x = torch.randn(batch_size, input_dim)
        q_mean, _, _ = variational_encoder(x)
        self.assertTrue(torch.equal(q_mean, encoder(x)))


class OrionModelConfigTest(parameterized.TestCase):
    def test_creates_orion_model_config_properly(self):
        config = create_test_orion_config()
        self.assertEqual(config.oncrna_encoder_args["input_dim"], 100)
        self.assertEqual(config.normalizing_scalar_args["latent_dim"], 1)
        self.assertEqual(config.cancer_predictor_args["layers_dim"], [10, 8, 3])

    def test_raises_config_invalid_error(self):
        with self.assertRaisesRegex(
            base_model.InvalidModelConfigException,
            ".* must have `latent_dim`=1.",
        ):
            config = create_test_orion_config(scalar_latent_dim=5)
            config.validate()

    def test_saves_and_loads_properly(self):
        config = create_test_orion_config()

        with tempfile.TemporaryDirectory() as config_dir:
            config_path = pathlib.Path(config_dir) / "config.json"

            config.save(config_path)
            loaded_config = orion_inference.OrionModelConfig.load(config_path)
            self.assertEqual(config, loaded_config)


class OrionInferenceModelTest(parameterized.TestCase):
    @parameterized.named_parameters(
        {
            "testcase_name": "with_normalizing_scalar",
            "add_scalar": False,
        },
        {
            "testcase_name": "without_normalizing_scalar",
            "add_scalar": True,
        },
        {
            "testcase_name": "with_normalizing_scalar_concatenated_injection",
            "add_scalar": True,
            "inject_lib_method": "concat",
        },
    )
    def test_creates_orion_inference_model_properly(
        self, add_scalar, inject_lib_method="multiply"
    ):
        x_oncrna_dim = 5
        oncrna_latent_dim = 6
        x_scalar_dim = 4
        scalar_latent_dim = 1
        num_classes = 3
        config = create_test_orion_config(
            x_oncrna_dim=x_oncrna_dim,
            oncrna_latent_dim=oncrna_latent_dim,
            x_scalar_dim=x_scalar_dim,
            scalar_latent_dim=scalar_latent_dim,
            num_classes=num_classes,
            inject_lib_method=inject_lib_method,
        )

        model = orion_inference.OrionInferenceModel(config=config)

        batch_size = 16
        x_oncrna = torch.randn(batch_size, x_oncrna_dim)
        x_scalar = torch.randn(batch_size, x_scalar_dim)
        if add_scalar:
            predictions = model(
                x_oncrna, x_scalar, inject_lib_method=inject_lib_method
            )
        else:
            predictions = model(x_oncrna)
        self.assertEqual(predictions.shape, (batch_size, num_classes))

    def test_saves_and_loads_properly(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_files_prefix = pathlib.Path(model_dir) / "test_model"

            config = create_test_orion_config()
            model = orion_inference.OrionInferenceModel(config=config)
            model.save(model_files_prefix)

            loaded_model = orion_inference.OrionInferenceModel.load(
                model_files_prefix
            )

            # `x_oncrna_dim` = 100 and `x_scalar_dim` = 50.
            batch_size = 16
            x_oncrna = torch.randn(batch_size, 100)
            x_scalar = torch.randn(batch_size, 50)

            # Set the models to eval mode to make the predictions deterministic.
            model.eval()
            loaded_model.eval()
            self.assertTrue(
                torch.equal(
                    model(x_oncrna, x_scalar), loaded_model(x_oncrna, x_scalar)
                )
            )
