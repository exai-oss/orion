"""Module providing OrionLoss class.

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

from typing import Dict, Tuple, Literal, Optional


import torch
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torch import nn
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl


class OrionLoss(nn.Module):
    """OrionLoss Implementation."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion_class: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        weight_sample_loss: Optional[bool] = False,
        add_regression_reconst: Optional[bool] = False,
        criterion_regression: Optional[
            torch.nn.Module
        ] = torch.nn.SmoothL1Loss(),
        gene_likelihood: Literal["zinb", "nb", "poisson", "gaussian"] = "zinb",
    ):
        """Initialize with device.

        Args:
            model: PyTroch model to access model parameters from.
            device (torch.device): Device to use for loss calculation.
            criterion_class: Loss function to use for classification.
            weight_sample_loss: Whether to weight the CE loss
                by the sample_loss_scaler parameter of dataloader.
            add_regression_reconst: bool
                Whether to use regression loss for reconstruction.
            criterion_regression: torch.nn.Module
                Criterion torch.nn.Module class for regression loss.
            gene_likelihood: str
                Distribution to use for gene likelihood.

        Methods:
            forward: Calculates the loss by comparing the predicted outputs to
                the target outputs.

        Raises:
            ZeroDivisionError: Prediction had no elements.

        Returns:
            loss_1 through loss_4 and an overall kis: Losses for each of the
            four components of the Orion model.
        """
        super().__init__()
        self.model = model
        self.device = device
        self.criterion_class = criterion_class.to(self.device)
        self.tmloss = torch.nn.TripletMarginLoss().to(self.device)
        self.criterion_reg = torch.nn.MSELoss().to(self.device)
        self.weight_sample_loss = weight_sample_loss
        self.add_regression_reconst = add_regression_reconst
        self.criterion_regression = criterion_regression
        self.gene_likelihood = gene_likelihood

    def forward(
        self, prediction_dict: Dict, parameters_dict: Dict
    ) -> torch.Tensor:
        # pylint: disable=not-callable
        """Calculates the loss by comparing the predicted outputs to the
        target outputs.


        Args:
            prediction: Predicted outputs, the number of which is used to
                appropriately scale the calculated loss.

        Returns:
            loss_1 through loss_4: Losses for each of the four components of
                the Orion model.

        Raises:
            ZeroDivisionError: Prediction had no elements.
        """
        prediction_dict = prediction_dict.copy()
        prediction_dict.update(parameters_dict)
        if prediction_dict["input_onctensor"].size(0) == 0:
            raise ZeroDivisionError(
                "loss cannot be normalized by the number of samples if"
                "`prediction` is of size 0."
            )

        loss_1, loss_2, loss_5 = self._calculate_loss_per_cat(
            prediction_dict["qz_m"],
            prediction_dict["qz_v"],
            prediction_dict["input_onctensor"],
            prediction_dict["px_rate"],
            prediction_dict["px_r"],
            prediction_dict["px_dropout"],
            prediction_dict["ql_m"],
            prediction_dict["ql_v"],
            prediction_dict["local_l_mean"],
            prediction_dict["local_l_var"],
            x_hat=prediction_dict["x_hat"],
        )

        loss_1 = torch.mean(loss_1)
        loss_2 = torch.mean(loss_2)
        ct_pred = prediction_dict["ctpred"]
        if prediction_dict["use_generative_sampling"]:
            if "ctpred.generative" not in prediction_dict:
                raise ValueError(
                    "ctpred.generative is not in prediction_dict,"
                    " while generative sampling is enabled."
                )

        # If there are multiple cell types, use cross entropy loss
        # If there is only one cell type, use MSE loss
        if prediction_dict["predict_classes"]:
            if prediction_dict["num_classes"] == 1:
                # If there is only one cell type, use MSE loss
                criterion = self.criterion_reg
            else:
                # If there are multiple cell types, use cross entropy loss
                criterion = self.criterion_class
            if prediction_dict["use_generative_sampling"]:
                # if use_generative_sampling is enabled, we have multiple
                # predictions for each cell type. We need to calculate
                # the loss for each prediction and average them. We are
                # using the generative model to predict the cell type.

                # First, reshape the generative predictions so that they
                # are in the same shape as the input onehottensor. They were
                # in the shape of (num_samples, batch_size, num_classes) and
                # we need them to be in the shape of (batch_size x num_samples
                # , num_classes).
                ctpred_gen_reshaped = prediction_dict[
                    "ctpred.generative"
                ].reshape(-1, prediction_dict["ctpred.generative"].shape[-1])
                if self.weight_sample_loss:
                    loss_3 = self.sample_weighted_loss(
                        ctpred_gen_reshaped,
                        prediction_dict["input_onehottensor"].repeat(
                            prediction_dict["generative_samples_num"], 1
                        ),
                        criterion,
                        parameters_dict["sample_loss_scaler"].repeat(
                            prediction_dict["generative_samples_num"], 1
                        ),
                    )
                else:
                    loss_3 = criterion(
                        ctpred_gen_reshaped,
                        prediction_dict["input_onehottensor"].repeat(
                            prediction_dict["generative_samples_num"], 1
                        ),
                    )
            else:
                if self.weight_sample_loss:
                    loss_3 = self.sample_weighted_loss(
                        ct_pred,
                        prediction_dict["input_onehottensor"],
                        criterion,
                        parameters_dict["sample_loss_scaler"],
                    )
                else:
                    loss_3 = criterion(
                        ct_pred, prediction_dict["input_onehottensor"]
                    )
        else:
            loss_3 = 0
        pos_qz_m_len = len(prediction_dict["pos_qz_m"])
        if prediction_dict["use_triplet_loss"] and pos_qz_m_len > 0:
            loss_4 = torch.tensor([0.0], requires_grad=True).to(self.device)
            for pos_qz_m, neg_qz_m in zip(
                prediction_dict["pos_qz_m"], prediction_dict["neg_qz_m"]
            ):
                loss_4 = loss_4 + self.tmloss(
                    prediction_dict["qz_m"],
                    pos_qz_m,
                    neg_qz_m,
                )
            loss_4 = loss_4[0] / pos_qz_m_len
        else:
            loss_4 = torch.tensor([0.0], requires_grad=True)[0]

        loss = loss_1 * 0
        losses = [loss_1, loss_2, loss_3, loss_4, loss_5]
        for i, each_loss in enumerate(losses):
            if prediction_dict["loss_scalers"][i] > 0:
                loss = loss + (
                    each_loss / torch.tensor(prediction_dict["loss_scalers"][i])
                )
        # Regularize the network
        loss = self._regularize_net(
            self.model,
            loss,
            prediction_dict["l1"],
            prediction_dict["l2"],
        )
        return loss_1, loss_2, loss_3, loss_4, loss, loss_5

    @staticmethod
    def _get_params(model: torch.nn.Module):
        """Get the parameters for the data encoder, library encoder, and
        decoder and concatenate each into one vector.

        Args:
            model: The model to get the parameters from.

        Returns:
            oncrna_params: The parameters for the data encoder.
            l_params: The parameters for the library encoder.
            decoder_params: The parameters for the decoder.
        """
        submodels = model.inference_model.get_submodels()
        oncrna_params = torch.cat(
            [x.view(-1) for x in submodels[0].parameters()]
        )
        l_params = torch.cat([x.view(-1) for x in submodels[1].parameters()])
        decoder_params = torch.cat(
            [x.view(-1) for x in model.decoder.parameters()]
        )
        return oncrna_params, l_params, decoder_params

    def _regularize_net(self, model, loss, l1=None, l2=None):
        """Regularize the network with L1 and L2
        It adjusts the loss for inference_model components and decoder.

        Args:
            model: The model to regularize.
            loss: The loss to add the regularization to.
            l1: The L1 penalty. This is the sum of the absolute values of the
                weights.
            l2: The L2 penalty. This is square root of the sum of the squares
                of the weights.

        Returns:
            loss: The loss with the regularization added.
        """
        if l2 is not None or l1 is not None:
            # Get the parameters for the data encoder, library encoder, and
            # decoder and concatenate each into one vector.
            oncrna_params, l_params, decoder_params = self._get_params(model)
            if l1 is not None:
                l1_z = l1 * torch.norm(oncrna_params, 1)
                l1_l = l1 * torch.norm(l_params, 1)
                l1_decoder = l1 * torch.norm(decoder_params, 1)
                loss = loss + l1_z + l1_l + l1_decoder
            if l2 is not None:
                l2_z = l2 * torch.norm(oncrna_params, 2)
                l2_l = l2 * torch.norm(l_params, 2)
                l2_decoder = l2 * torch.norm(decoder_params, 2)
                loss = loss + l2_z + l2_l + l2_decoder
        return loss

    def _calculate_loss_per_cat(
        self,
        qz_m: torch.Tensor,
        qz_v: torch.Tensor,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        ql_m: torch.Tensor,
        ql_v: torch.Tensor,
        local_l_mean: torch.Tensor,
        local_l_var: torch.Tensor,
        x_hat: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate loss_1 and loss_2.
            Loss_1 is the reconstruction loss plus the KL divergence between
            the variational distribution of ``l`` and the prior distribution
            of ``l``. This is the loss for the library encoder.
            Loss_2 is the KL divergence between the variational distribution
            of ``z`` and the prior distribution of ``z``. This is for the loss
            for the data encoder.

        Args:
            qz_m: torch.Tensor
                    tensor of mean of the (variational) posterior
                    distribution of ``z`` (from oncRNAs) with shape ``
                    (batch_size, n_latent)``
            qz_v: torch.Tensor
                    tensor of variance of the (variational) posterior
                    distribution of ``z`` (from oncRNAs) with shape ``
                    (batch_size, n_latent)``
            x: torch.Tensor
                    tensor of counts with shape ``(batch_size, n_input)``
            px_rate: torch.Tensor
                    tensor of mean of the (zero-inflated negative binomial)
                    distribution of ``x`` with shape ``(batch_size, n_input)``
            px_r: torch.Tensor
                    tensor of count of the (zero-inflated negative binomial)
                    distribution of ``x`` with shape ``(batch_size, n_input)``
            px_dropout: torch.Tensor
                    tensor of dropout of the (zero-inflated negative binomial)
                    distribution of ``x`` with shape ``(batch_size, n_input)``
            ql_m: torch.Tensor
                    tensor of mean of the (variational) posterior
                    distribution of ``l`` (from endogenous RNAs) with shape ``
                    (batch_size, n_latent)``
            ql_v: torch.Tensor
                    tensor of variance of the (variational) posterior
                    distribution of ``l`` (from endogenous RNAs) with shape ``
                    (batch_size, n_latent)``
            local_l_mean: torch.Tensor
                    tensor of mean of the (variational) posterior
                    distribution of ``l`` (from oncRNAs) with shape ``
                    (batch_size, n_input)``
            local_l_var: torch.Tensor
                    tensor of variance of the (variational) posterior
                    distribution of ``l`` (from oncRNAs) with shape ``
                    (batch_size, n_input)``

        Returns:
            Tuple(loss_1: torch.Tensor, loss_2: torch.Tensor)

        """
        # Calculate the central values of the distributions, then calculate
        # the KL divergences between the variational distributions and the
        # prior distributions.
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        # First calculate the KL divergence between the variational
        # distribution of ``z`` and the prior distribution of ``z``.
        kl_divergence_z = kl(
            Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)
        ).sum(dim=1)
        # Next calculate the KL divergence between the variational
        # distribution of ``l`` and the prior distribution of ``l``.
        kl_divergence_l = kl(
            Normal(
                ql_m.to(self.device),
                torch.sqrt(ql_v).to(self.device)),
            Normal(
                local_l_mean.to(self.device),
                torch.sqrt(local_l_var).to(self.device)),
        ).sum(dim=1)
        # Calculate the reconstruction loss.
        reconst_loss, reg_reconst_loss = self._get_reconstruction_loss(
            x=x, px_rate=px_rate, px_r=px_r, px_dropout=px_dropout, x_hat=x_hat
        )
        return reconst_loss + kl_divergence_l, kl_divergence_z, reg_reconst_loss

    def _get_reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        gene_likelihood: Literal[
            "zinb", "nb", "poisson", "gaussian", None
        ] = "zinb",
        x_hat: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the reconstruction loss. This is the negative log
        likelihood of the data under one of ZINB, negative binomial, Poisson, or
        Gaussian distribution.

        Args:

            x: torch.Tensor
                    tensor of counts with shape ``(batch_size, n_input)``
            px_rate: torch.Tensor
                    tensor of mean of the (zero-inflated negative binomial)
                    distribution of ``x`` with shape ``(batch_size, n_input)``
            px_r: torch.Tensor
                    tensor of count of the (zero-inflated negative binomial)
                    distribution of ``x`` with shape ``(batch_size, n_input)``
            px_dropout: torch.Tensor
                    tensor of dropout of the (zero-inflated negative binomial)
                    distribution of ``x`` with shape ``(batch_size, n_input)``
            gene_likelihood: ["zinb", "nb", "poisson", "gaussian", None]
                    Distribution to use for gene likelihood.
            x_hat: If provided and self.add_regression_reconst is True, use this
                tensor for regression loss.

        Returns:

            torch.Tensor: MLE Reconstruction loss
            torch.Tensor: MSE Reconstruction loss

        """
        # initiate MSE loss placeholder if x_hat is None
        if x_hat is None:
            reg_reconst_loss = torch.Tensor([0.0]).to(self.device)
            reg_reconst_loss.requires_grad = True
            if self.add_regression_reconst:
                raise ValueError(
                    "x_hat cannot be None with " "add_regression_reconst = True"
                )
        # set gene_likelihood to allow portability of function
        # for use with k-mers with a different distribution
        if gene_likelihood is None:
            gene_likelihood = self.gene_likelihood
        # MSE-based reconstruction loss
        if self.add_regression_reconst:
            reg_reconst_loss = self.criterion_regression(x_hat, x)
        # MLE-based Reconstruction Loss, based on the specified distribution
        if self.gene_likelihood == "zinb":
            reconst_loss_mle = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss_mle = (
                -NegativeBinomial(mu=px_rate, theta=px_r)
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss_mle = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        elif self.gene_likelihood in ["Normal", "Gaussian"]:
            reconst_loss_mle = -Normal(px_rate, px_r).log_prob(x).sum(dim=-1)
        return reconst_loss_mle, reg_reconst_loss

    def sample_weighted_loss(
        self, ctpreds, cttargets, criterion, sample_loss_scaler
    ):
        """Weights a loss by the sample weight.
        Works with regression loss too.
        Args:
            ctpreds: predicted logits
            cttargets: target labels
            criterion: torch nn.criterion class for loss calculation
                Expects criterion object with reduction parameter
            sample_loss_scaler: tensor of weights for each sample
        Returns:
            loss: weighted closs
        """
        assert hasattr(
            criterion, "reduction"
        ), "only supports criterions with reduction"
        current_reduction = criterion.reduction
        criterion.reduction = "none"
        weighted_loss = criterion(ctpreds, cttargets) * sample_loss_scaler
        weighted_loss = weighted_loss.mean()
        criterion.reduction = current_reduction
        return weighted_loss
