"""
Base implementation for classes designed to aggregate and perform calculations
on data output by a model.

Note: Currently all Analyzer classes are kept in this `base.py` as there are
      few.

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
import collections
from typing import Any

import numpy as np


class Analyzer(abc.ABC):
    """Abstraction for analysis of model outputs.
    There are two main types of analysis for model outputs that are of interest:
        1. Analysis over a period of time.
        2. Analysis given a single datapoint.
    Three methods are exposed for these two analyses of interest:
        `analyze`: Perform desired analysis given a datapoint.
        `buffer`: Store data from for later analysis.
        `analyze_buffer`: Perform desired analysis on stored data.
    The `_aggregate` private method must be implemented for to call
    `_analyze_buffer` to dictate how multiple datapoints are to be aggregated.

    Note: This three function `analyze`, `buffer`, `analyze_buffer` approach is
          not set in stone if more flexible interfaces are requested.
    """

    def __init__(self):
        """Create a buffer to collect data."""
        self._buffer = collections.defaultdict(list)

    @abc.abstractmethod
    def analyze(self, data: Any) -> Any:
        """Overwrite with desired analysis.

        Args:
            data: Data to analyze.

        Returns:
            Result of analysis on data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _aggregate(self, buffer: dict[str, list]) -> Any:
        """Overwrite with desired aggregation for buffered data.

        Args:
            buffer: Buffer(s) of data to aggregate organized in a dict.

        Returns:
            Aggregated data.
        """
        raise NotImplementedError

    def buffer(self, data: dict) -> None:
        """Buffer result from batch for later processing.

        Args:
            data: A dictionary containing data.
        """
        for key, value in data.items():
            self._buffer[key].append(value)

    def analyze_buffer(self) -> Any:
        """Analyze buffered data.
        Clears buffered data after analysis.

        Returns:
            Result of analysis on buffer.
        """
        result = self.analyze(self._aggregate(self._buffer))
        self._reset()
        return result

    def _reset(self) -> None:
        """Reset interal data."""
        self._buffer = collections.defaultdict(list)


class DefaultAnalyzer(Analyzer):
    """No-op class."""

    def analyze(self, data: Any) -> Any:
        """No-op."""
        pass

    def _aggregate(self, buffer: dict) -> Any:
        """No-op."""
        pass


class LossAnalyzer(Analyzer):
    """Loss analysis.

    Loss is typically calculated per-epoch, so the `_aggregate` function sums
    the values obtained over the iterations.
    """

    def analyze(self, data: Any) -> Any:
        return data

    def _aggregate(self, buffer: dict) -> Any:
        return sum(self._buffer["loss"])


class OrionLossAnalyzer(Analyzer):
    """Orion Loss analysis.

    Loss is typically calculated per-epoch, so the `_aggregate` function
    performs aggregations for Orion loss.
    """

    def analyze(self, data: Any) -> Any:
        """No-op."""
        pass

    def analyze_buffer(self, loss_scalers, num_iters) -> Any:
        """Analyze buffered data.
        Clears buffered data after analysis.

        Returns:
            Result of analysis on buffer.
        """
        result = self._aggregate(self._buffer, loss_scalers, num_iters)
        self._reset()
        return result

    def _aggregate(
        self,
        buffer: dict[str, list],
        loss_scalers,
        num_iters,
    ) -> Any:
        """Aggregate data from buffer.

        Args:
            buffer: Buffer(s) of data to aggregate organized in a dict.
                Contains the following keys:
                    adacc: Accuracy value.
                    loss_1: MLE Reconstruction loss.
                    loss_2: KLD loss.
                    loss_3: Cross entropy loss.
                    loss_4: Triple margin loss.
                    loss_5: Regression based reconstruction loss.
                    loss: Total loss.
                    idx: Celltype indices.
                    celltype_resps: Celltypes.
                    celltype_preds: Celltype predictions.
                    postmat: Posterior matrix. Contains posterior probabilities
                        of each cell belonging to each celltype.
                    reconst: Reconstructed matrix. Contains reconstructed
                        values of each cell. Buffers from the tensor of
                        predicted frequencies of expression with shape
                        ``(batch_size, inputsize)``
                    mumat: Mu matrix. Contains mu values of each cell.
                        Computed from the tensor of mean of the (variational)
                        posterior distribution of ``l`` (from endogenous RNAs)
                        with shape ``(batch_size, 1)``
                    sd2mat: SD2 matrix. Contains sd2 values of each cell.
                        Computed from the tensor of variance of the
                        (variational) posterior distribution of ``z`` (from
                        oncRNAs) with shape ``(batch_size, n_latent)``
            loss_scalers: List of loss scalers.
            num_iters: Number of iterations.

        Returns:
            Aggregated data. (dict) with keys:
                accval: Accuracy value.
                cur_loss_reconst: Reconstruction loss.
                cur_kld: KLD loss.
                cur_ce: Cross entropy loss.
                cur_loss: Total loss.
                cur_tm: Triple margin loss.
                celltype_resps: Celltypes.
                celltype_preds: Celltype predictions.
                postmat: Posterior matrix. Contains posterior probabilities of
                    each cell belonging to each celltype.
                reconst: Reconstructed matrix. Contains reconstructed values of
                    each cell. Aggregated from the tensor of predicted
                    frequencies of expression with shape
                    ``(batch_size, inputsize)``
                mumat: Mu matrix. Aggregated from the tensor of mean of the
                    (variational) posterior distribution of ``l`` (from
                    endogenous RNAs) with shape ``(batch_size, 1)``
                sd2mat: SD2 matrix. Contains sd2 values of each cell.
                Aggregated from the tensor of variance of the (variational)
                posterior distribution of ``z`` (from oncRNAs) with shape
                ``(batch_size, n_latent)``

        """
        aggresult = {}
        # Accuracy value, calculated as the average of the accuracy values
        aggresult["accval"] = sum(buffer["adacc"]) / num_iters
        # Different loss values are calculated per-iteration, so the
        # `_aggregate` function sums the values obtained over the iterations.
        aggresult["cur_loss_reconst"] = (
            sum(buffer["loss_1"]) / loss_scalers[0] / num_iters
        )
        aggresult["cur_kld"] = (
            sum(buffer["loss_2"]) / loss_scalers[1] / num_iters
        )
        aggresult["cur_ce"] = (
            sum(buffer["loss_3"]) / loss_scalers[2] / num_iters
        )
        aggresult["cur_loss"] = sum(buffer["loss"])
        aggresult["cur_tm"] = (
            sum(buffer["loss_4"]) / loss_scalers[3] / num_iters
        )
        if buffer["loss_5"] is not None:
            aggresult["cur_reg_recon_loss"] = (
                sum(buffer["loss_5"]) / loss_scalers[4]
            ) / num_iters
        else:
            aggresult["cur_reg_recon_loss"] = None
        # Celltype indices are concatenated and the maximum value is added by 1
        # to get the length of the first dimension of the aggregated arrays.
        # This is done to avoid errors when concatenating arrays of different
        # lengths.
        if "idx" in buffer:
            # idxs are not necessarily sorted. In addition, max(idxs) could be
            # more than the length of idxs. Therefore, we need to find the
            # maximum value of idxs and add 1 to get the length of the first
            # dimension of the aggregated arrays.
            if len(buffer["idx"]) > 1:
                # Concatenate idxs from different batches
                idxs = np.concatenate(buffer["idx"])
            else:
                idxs = buffer["idx"][0]
            # dim_one_len is the length of the first dimension of the
            # aggregated arrays.
            dim_one_len = max(idxs) + 1
            # Celltype predictions and responses are concatenated.
            keys = ["celltype_resps", "celltype_preds"]  # 1d arrays
            for key in keys:
                if key in buffer:
                    temp = np.concatenate(buffer[key])
                    # Initialize the aggregated array with zeros. These
                    # aggregated arrays will be used to calculate the
                    # confusion matrix.
                    aggresult[key] = np.zeros((dim_one_len))
                    for i in range(len(idxs)):
                        aggresult[key][idxs[i]] = temp[i]
            # Posterior matrix, reconstructed matrix, mu matrix, and sd2 matrix
            # are concatenated.
            keys = ["postmat", "reconst", "mumat", "sd2mat"]  # 2d arrays
            for key in keys:
                if key in buffer:
                    temp = np.concatenate(buffer[key])
                    # Initialize the aggregated array with zeros. These arrays
                    # will be used to calculate the posterior matrix,
                    # reconstructed matrix, mu matrix, and sd2 matrix.
                    aggresult[key] = np.zeros((dim_one_len, temp.shape[1]))
                    for i in range(len(idxs)):
                        aggresult[key][i, :] = temp[i]

        return aggresult
