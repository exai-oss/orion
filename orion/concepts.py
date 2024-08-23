""" Type aliases and small class definitions used to capture ubiquious
concepts/types. The intention is to provide additional meaning and readability
(along with light type checking) to commonly used data containers and function
signatures in model training when using Pytorch.

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

from __future__ import annotations

import dataclasses
import enum
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils import data

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# Args: prediction, target. Returns: Loss
# For regularization, passing the weights is not necessary as the loss class
# can be given a pointer to the weights upon initilization.

TrainDataset = data.Dataset
TestDataset = data.Dataset
TrainTestDatasetPair = tuple[TrainDataset, TestDataset]
# Quick aliases simply used to provide additional readabilty.
# Question: does this have any risks? There could be an explicit wrapper
# which adds an additional enum field with member enum 'test' or 'train'.
# And then in training / evaluator code can have additional checks that
# training only on a TrainDataset and evaluating / validating only on a
# TestDataset.


@dataclasses.dataclass(frozen=True)
class Batch:
    """The forward propagation inputs. A feature, label pair."""

    x: torch.Tensor
    y: torch.Tensor


@dataclasses.dataclass(frozen=True)
class BatchResult:
    """The forward propagation outputs. A prediction, loss pair."""

    prediction: torch.Tensor
    loss: torch.Tensor


class BatchMode(enum.Enum):
    FULLBATCH = 1
    MINIBATCH = 2


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    """Configuration for training."""

    batch_mode: BatchMode = BatchMode.FULLBATCH
    eval_freq: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class OrionTrainBatchResult:
    """The Orion forward propagation outputs, containing:
    loss_1: torch.Tensor: MLE based Reconstruction loss.
    loss_2: torch.Tensor: KL divergence loss.
    loss_3: torch.Tensor: Cross entropy loss.
    loss_4: torch.Tensor: Triplet margin loss.
    loss_5: torch.Tensor: Regression based reconstruction loss.
    loss: torch.Tensor: Total loss.
    adacc: torch.Tensor: Adversarial accuracy.
    """

    loss_1: torch.Tensor
    loss_2: torch.Tensor
    loss_3: torch.Tensor
    loss_4: torch.Tensor
    loss_5: torch.Tensor
    loss: torch.Tensor
    adacc: np.ndarray


@dataclasses.dataclass(frozen=True)
class OrionTuneBatchResult:
    """The Orion tuning outputs, containing:
    loss_1: torch.Tensor: MLE based Reconstruction loss.
    loss_2: torch.Tensor: KL divergence loss.
    loss_3: torch.Tensor: Cross entropy loss.
    loss_4: torch.Tensor: Triplet margin loss.
    loss_5: torch.Tensor: Regression based reconstruction loss.
    loss: torch.Tensor: Total loss.
    celltype_resps: np.ndarray: Celltype responses.
    celltype_preds: np.ndarray: Celltype predictions.
    postmat: torch.Tensor: Posterior matrix. Contains posterior probabilities
        of each cell belonging to each celltype.
    reconst: np.ndarray: Reconstructed data. Contains reconstructed values of
        each cell. Buffers from the tensor of predicted frequencies of
        expression with shape ``(batch_size, inputsize)``
    mumat: np.ndarray: Mean matrix. Computed from the tensor of mean of the
        (variational) posterior distribution of ``l`` (from endogenous RNAs)
        with shape ``(batch_size, 1)``
    sd2mat: np.ndarray: Variance matrix. Computed from the tensor of variance
        of the (variational) posterior distribution of ``z`` (from oncRNAs)
        with shape ``(batch_size, n_latent)``
    """

    loss_1: torch.Tensor
    loss_2: torch.Tensor
    loss_3: torch.Tensor
    loss_4: torch.Tensor
    loss_5: torch.Tensor
    loss: torch.Tensor
    celltype_resps: np.ndarray
    celltype_preds: np.ndarray
    postmat: torch.Tensor
    reconst: np.ndarray
    mumat: np.ndarray
    sd2mat: np.ndarray
