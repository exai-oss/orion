"""Module providing VAE class.

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

from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Normal
from orion.models.encoder_decoder import Decoder
from orion.models import orion_inference


class VAE(nn.Module):
    """
    This Variational Auto Encoder is at the heart of the Orion model.
    The VAE supports a few different ways of encoding library size.
    As latent variable, it can be estimated from the input matrix
    or an additional matrix.
    The class allows adjusting for batch effect by encoding the
    one-hot encoding of batch into the z_encoder and decoder.

    Args:
        n_input (int): Number of input oncrnas.
        n_labels (int) Number of cell types to be used. For example,
            cancer vs. control (rename to n_classes?, renamed from n_celltypes)
        n_hidden (int): Number of hidden units of the hidden layer(s).
        n_latent (int): Dimensionality of the latent space.
        n_layers (int): Number of hidden layers used for encoder and
            decoder NNs.
        dropout_rate (float): Dropout rate for neural networks.
        log_variational (bool): Log the variational parameters.
        gene_likelihood (Literal["zinb", "nb", "poisson"]): One
            of the following
                * ``'zinb'`` - Zero-inflated negative binomial
                * ``'nb'`` - Negative binomial
                * ``'poisson'`` - Poisson
        latent_distribution (Literal["normal", "ln"]): One of the following
            * ``'normal'`` - Isotropic normal distribution
            * ``'ln'`` - Logistic normal (normal with a softmax
                non-linearity constrained to sum to one)
        n_input_lib (int): Number of input dimensions of the Q matrix
            or the endogenous RNA counts used for inferring the library size.
            This is an important parameter for the default Orion model.
            Example; number of miRNAs in the miRNA matrix provided for
            library size estimation
        use_batch_norm (Literal["encoder", "decoder", "none", "both"]):
            Whether to use batch norm in the encoder/decoder.
        use_layer_norm (Literal["encoder", "decoder", "none", "both"]):
            Whether to use layer norm in the encoder/decoder.
        inject_lib (bool): Whether to inject library size into the
            latent space.
        add_batchnorm_lib (bool): Whether to use batchnorm on the library
            size encoder.
        use_generative_sampling (bool): Whether to use generative sampling
            for the latent space.
        generative_samples_num (int): Number of generative samples to use
            for the latent space.
        add_regression_reconst (bool): Whether to use regression for the
            reconstruction.
    """

    def __init__(
        self,
        n_input: int,
        n_labels: int = 2,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson", "gaussian"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        n_input_lib: Optional[int] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        inject_lib: bool = False,
        inject_lib_method: Literal["concat", "multiply"] = "multiply",
        add_batchnorm_lib: bool = False,
        use_generative_sampling: bool = True,
        generative_samples_num: Optional[int] = 100,
        add_regression_reconst: Optional[bool] = False,
    ):
        if latent_distribution == "ln":
            raise ValueError(
                f"The value {latent_distribution=} is not supported at this "
                "time. Please contact ml-team@ for more information."
            )

        super().__init__()
        self.add_regression_reconst = add_regression_reconst
        self.dispersion = "gene"
        self.encode_covariates = False
        if n_input_lib is None:
            n_input_lib = n_input
        self.generative_samples_num = generative_samples_num
        self.use_generative_sampling = use_generative_sampling
        self.inject_lib = inject_lib
        self.inject_lib_method = inject_lib_method
        self.add_batchnorm_lib = add_batchnorm_lib
        self.n_labels = n_labels
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.latent_distribution = latent_distribution

        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        use_batch_norm_encoder = use_batch_norm in {"encoder", "both"}
        use_batch_norm_decoder = use_batch_norm in {"decoder", "both"}
        use_layer_norm_encoder = use_layer_norm in {"encoder", "both"}
        use_layer_norm_decoder = use_layer_norm in {"decoder", "both"}

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input
        n_input_encoder_l = n_input_lib

        oncrna_encoder_args = {
            "input_dim": n_input_encoder,
            "latent_dim": n_latent,
            "hidden_layer_num": n_layers,
            "hidden_dim": n_hidden,
            # fully_connected_optional_kwargs
            "dropout_rate": dropout_rate,
            "add_batch_normalization": use_batch_norm_encoder,
            "add_layer_normalization": use_layer_norm_encoder,
        }
        normalizing_scaler_args = {
            "input_dim": n_input_encoder_l,
            "latent_dim": 1,
            "hidden_layer_num": 1,
            "hidden_dim": n_hidden,
            # fully_connected_optional_kwargs
            "dropout_rate": dropout_rate,
            "add_batch_normalization": use_batch_norm_encoder,
            "add_layer_normalization": use_layer_norm_encoder,
        }
        if self.inject_lib_method == "concat":
            # be 1. the +1 is for the library size from the latent values for
            # library
            in_dim_ctpred = int((n_latent + 1))
        else:
            in_dim_ctpred = int(n_latent)
        cancer_predictor_args = {
            "layers_dim": [in_dim_ctpred, 2 * n_labels, n_labels],
            "activation": "LeakyReLU",
            "no_activation_on_output_layer": True,
            "predictor_batch_norm": add_batchnorm_lib
        }
        orion_inference_config = orion_inference.OrionModelConfig(
            oncrna_encoder_args, normalizing_scaler_args, cancer_predictor_args
        )
        self._inference_model = orion_inference.OrionInferenceModel(
            config=orion_inference_config
        )

        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = Decoder(
            n_input=n_input_decoder,
            n_output=n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            add_regression_reconst=self.add_regression_reconst,
        )

    @property
    def inference_model(self):
        return self._inference_model

    def get_latents(self, x) -> torch.Tensor:
        """
        Returns the result of ``sample_from_posterior_z`` inside a list.
        Args:
            x: tensor of values with shape ``(batch_size, inputsize)``

        Returns:
            one element list of tensor
        """
        return [self.sample_from_posterior_z(x)]

    def sample_from_posterior_z(
        self, x, give_mean=False, n_samples=0
    ) -> torch.Tensor:
        """
        Samples the tensor of latent values from the posterior.
        Args:
            x: tensor of values with shape ``(batch_size, inputsize)``
            give_mean: is True when we want the mean of the posterior
                distribution rather than sampling (Default value = False)
            n_samples: how many MC samples to average over for
                transformed mean (Default value = 5000)
        Returns:
            tensor of shape ``(batch_size, lvsize)``
        """
        if n_samples > 0:
            raise ValueError(
                f"The value {n_samples=} is not supported at this time. "
                "Please contact ml-team@ for more information."
            )

        z_encoder, _, _ = self._inference_model.get_submodels()
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, _, z = z_encoder(x)

        if give_mean:
            z = qz_m
            # if self.latent_distribution == "ln":
            #     samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
            #     z = self.z_encoder.z_transformation(samples)
            #     z = z.mean(dim=0)
        return z

    def sample_from_posterior_l(self, x, give_mean=True) -> torch.Tensor:
        """
        Samples the tensor of library sizes from the posterior.
        Args:
            x: tensor of values with shape ``(batch_size, inputsize)``
        Returns:
            tensor of shape ``(batch_size, 1)``
        """
        _, l_encoder, _ = self._inference_model.get_submodels()
        if self.log_variational:
            x = torch.log(1 + x)
        try:
            ql_m, ql_v, library = l_encoder(x)
        except Exception as e:
            out_str = f"Data types {x.dtype}"
            print(out_str)
            raise Exception("Data type issue?").with_traceback(e.__traceback__)
        if give_mean:
            library = torch.distributions.LogNormal(ql_m, ql_v.sqrt()).mean
        return library

    def get_sample_scale(
        self,
        x,
        generative_samples_num=1,
    ) -> torch.Tensor:
        """
        Returns the tensor of predicted frequencies of expression.
        Args:
            x: tensor of values with shape ``(batch_size, inputsize)``
            batch_index: array that indicates which batch the cells
                belong to with shape ``batch_size`` (Default value = None)
            generative_samples_num: number of samples (Default value = 1)
        Returns:
            tensor of predicted frequencies of expression
            with shape ``(batch_size, inputsize)``
        """
        return self.inference(
            x,
            generative_samples_num=generative_samples_num,
        )["px_scale"]

    def get_sample_rate(
        self,
        x,
        generative_samples_num=1,
    ) -> torch.Tensor:
        """
        Returns the tensor of means of the negative binomial distribution.
        Args:
            x: tensor of values with shape ``(batch_size, inputsize)``
            batch_index: array that indicates which batch the cells belong to
                with shape ``batch_size`` (Default value = None)
            generative_samples_num: number of samples (Default value = 1)
        Returns:
            tensor of means of the negative binomial distribution with
                shape ``(batch_size, inputsize)``
        """
        return self.inference(
            x,
            generative_samples_num=generative_samples_num,
        )["px_rate"]

    @staticmethod
    def sample_normal_distribution(
        sample_num: int, mean: torch.tensor, variance: torch.tensor
    ) -> torch.tensor:
        """
        Samples from a normal distribution.
        Args:
            sample_num: number of samples
            mean: mean of the normal distribution
            variance: variance of the normal distribution
        Returns:
            torch.tensor``
        """
        mean = mean.unsqueeze(0).expand(
            (sample_num, mean.size(0), mean.size(1))
        )
        variance = variance.unsqueeze(0).expand(
            (sample_num, variance.size(0), variance.size(1))
        )
        sample = Normal(mean, variance.sqrt()).rsample()

        return sample

    def inference(
        self,
        x: torch.Tensor,
        library_manual: torch.Tensor = None,
        generative_samples_num: int = 1,
        x_lib: torch.Tensor = None,
    ) -> dict:
        """Helper function used in forward pass.
        Args:
            x: tensor of values with shape ``(batch_size, inputsize)``
            library_manual: tensor of library sizes with shape
                ``(batch_size, 1)`` (Default value = None)
            generative_samples_num: number of generative samples used for
                the latent space (Default value = 1)
            x_lib: tensor of library sizes with shape ``(batch_size, 1)``
                (Default value = None)
        Returns:
            dictionary of sample scale, rate, and dropout values. Contains:
               px_scale: torch.Tensor
                    tensor of predicted frequencies of expression
                    with shape ``(batch_size, inputsize)``
                px_r: torch.Tensor
                    tensor of r of the negative binomial distribution
                    with shape ``(batch_size, inputsize)``
                px_rate: torch.Tensor
                    tensor of means of the negative binomial distribution
                    with shape ``(batch_size, inputsize)``
                px_dropout: torch.Tensor
                    tensor of the dropout probabilities of ZINB with shape
                    ``(batch_size, inputsize)``
                qz_m: torch.Tensor
                    tensor of mean of the (variational) posterior
                    distribution of ``z`` (from oncRNAs) with shape ``
                    (batch_size, n_latent)``
                qz_v: torch.Tensor
                    tensor of variance of the (variational) posterior
                    distribution of ``z`` (from oncRNAs) with shape
                    ``(batch_size, n_latent)``
                ql_m: torch.Tensor
                    tensor of mean of the (variational) posterior distribution
                    of ``l`` (from endogenous RNAs) with shape
                    ``(batch_size, 1)``
                ql_v: torch.Tensor
                    tensor of variance of the (variational) posterior
                    distribution of ``l`` (from endogenous RNAs) with shape
                    ``(batch_size, 1)``
                library: torch.Tensor
                    tensor of library sizes with shape ``(batch_size, 1)``
                px: torch.Tensor
                    tensor of probabilities of expression scaled by size
                    factor with shape ``(batch_size, inputsize)``
                ctpred: torch.Tensor
                    tensor of predicted cell type probabilities with shape
                    ``(batch_size, n_labels)``
        """
        (
            z_encoder,
            l_encoder,
            cancer_predictor,
        ) = self._inference_model.get_submodels()
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = z_encoder(x_)
        ql_m, ql_v, library = l_encoder(x_lib)
        # Sampling from k-mers

        # Predict celltypes using zctpred_in = qz_m

        if generative_samples_num > 1:
            untransformed_z = self.sample_normal_distribution(
                sample_num=generative_samples_num, mean=qz_m, variance=qz_v
            )
            untransformed_l = self.sample_normal_distribution(
                sample_num=generative_samples_num, mean=ql_m, variance=ql_v
            )
            # We do not need to give the z from gen sampling to the decoder
            generative_sampling_z = untransformed_z
            library = untransformed_l
            if self.inject_lib:
                l_sampled = untransformed_l
                if self.inject_lib_method == "concat":
                    # concat the library size to the latent space
                    generative_sampling_z = torch.concat(
                        (generative_sampling_z, l_sampled), dim=-1
                    )
                else:
                    # multiply the library size to the latent space
                    # This will happen if concat is not selected
                    generative_sampling_z = torch.multiply(
                        generative_sampling_z, l_sampled
                    )
            ctpred = cancer_predictor(
                generative_sampling_z.reshape(
                    generative_sampling_z.shape[0]
                    * generative_sampling_z.shape[1],
                    generative_sampling_z.shape[2],
                )
            )
            ctpred = ctpred.reshape(
                generative_sampling_z.shape[0],
                generative_sampling_z.shape[1],
                ctpred.shape[1],
            )

        else:
            if self.inject_lib:
                ctpred = self._inference_model(
                    x_, x_lib, self.inject_lib_method
                )
            else:
                ctpred = self._inference_model(x_)

        px_scale, px_r, px_rate, px_dropout, px, x_hat = self.decoder(
            z, library
        )

        px_r = torch.exp(self.px_r)

        return {
            "px_scale": px_scale,
            "px_r": px_r,
            "px_rate": px_rate,
            "px_dropout": px_dropout,
            "x_hat": x_hat,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "z": z,
            "ql_m": ql_m,
            "ql_v": ql_v,
            "library": library,
            "px": px,
            "ctpred": ctpred,
        }

    def generative(
        self,
        z,
        library,
        cont_covs=None,
    ):
        """Runs the generative model."""
        decoder_input = (
            z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        )

        px_scale, px_r, px_rate, px_dropout, _, x_hat = self.decoder(
            decoder_input, library
        )

        px_r = torch.exp(self.px_r)

        return {
            "px_scale": px_scale,
            "px_r": px_r,
            "px_rate": px_rate,
            "px_dropout": px_dropout,
            "x_hat": x_hat,
        }

    def forward(
        self,
        x: torch.tensor,
        evaluation_mode: Optional[bool] = False,
        x_lib: Optional[torch.tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        Args:
            x: tensor of values with shape ``(batch_size, inputsize)``
            evaluation_mode:  (Default value = False). When True, we don't
                want to use generative sampling
            x_lib: tensor of library sizes with shape ``(batch_size, 1)``
                (Default value = None)
            Returns:
            tuple of two tensors:
                * the first tensor is the output of the inference
                * the second tensor is the output of the generative
                    model
        """
        dict_inference = self.inference(x, x_lib=x_lib)
        # when evaluation_mode is True, we don't want to use generative sampling
        if self.use_generative_sampling and not evaluation_mode:
            dict_inference_ct = self.inference(
                x=x,
                x_lib=x_lib,
                generative_samples_num=self.generative_samples_num,
            )
            dict_inference["ctpred.generative"] = dict_inference_ct["ctpred"]
        library = dict_inference["library"]
        dict_generative = self.generative(dict_inference["z"], library)
        return dict_inference, dict_generative
