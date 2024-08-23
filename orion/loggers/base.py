"""
Interface for classes designed to log information.

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
from typing import Any, Dict, Optional

import torch


class Logger(abc.ABC):
    """Interface for a generic logger.
    Each concrete derived class must implement the log method, whether logging
    to stdout, file, in-memory arrays, or wrapping around existing library
    function such as `mlflow.log()`.
    """

    @abc.abstractmethod
    def log(self, data: Any) -> None:
        """Generic log function."""
        raise NotImplementedError


class DefaultLogger(Logger):
    """No-op class."""

    def log(self, data: Any) -> None:
        """No-op."""
        pass


class TensorLogger(Logger):
    """
    # NOTE: hardcoded to use torch and floats at the moment.
    # No error checking or extension atm either.
    """

    def __init__(self, tensor: torch.Tensor):
        self._log = tensor
        self._cur_idx = 0

    def log(self, data: torch.float) -> None:
        self._log[self._cur_idx] = data
        self._cur_idx += 1


class OrionLogger(Logger):
    """Orion logger class. Logs training information to a dictionary.
    This logger is designed to be used with the `Orion` class. It is not
    intended to be used as a standalone logger.

    It is intended to provide a uniform interface for logging training and
    validation information. The `Orion` wrapper will initiate the `log` method
    with the appropriate dictionary.

    The logger will log the following information:
        - epoch
        - reconstruction loss
        - KL divergence loss
        - cross entropy loss
        - triplet margin loss/ adversarial batch loss
        - accuracy

    The dictionary keys do not need to be the same as the above list. However,
    we recommend that the input dictionary be created using an `OrderedDict` to
    ensure that the order of the keys is consistent across all models.

    Since the `Orion` wrapper will call the `log` method, the user should not
    call this method directly. Instead, the user should use it when creating
    their own `Orion` wrapper.
    """

    def __init__(self, dict_log: Dict):
        self._dict_log = dict_log
        self.keys = list(dict_log.keys())

    def get_log(self):
        """Return the log dictionary."""
        return self._dict_log

    def log(
        self,
        epoch: int,
        cur_loss_reconst: torch.Tensor,
        cur_kld: torch.Tensor,
        cur_ce: torch.Tensor,
        tmloss: torch.Tensor,
        accval: torch.Tensor,
        cur_reg_recon_loss: Optional[torch.Tensor] = None,
    ):
        """Log training information to the dictionary.

        Args:
            epoch: Current epoch.
            cur_loss_reconst: Current reconstruction loss.
            cur_kld: Current KL divergence loss.
            cur_ce: Current cross entropy loss.
            tmloss: Current total loss.
            accval: Current accuracy.
            cur_reg_recon_loss: Current regression reconstruction loss.
        """
        if cur_reg_recon_loss is None:
            cur_reg_recon_loss = torch.Tensor([0])
        fourth_loss = "Triplet.Margin.Loss"
        if fourth_loss not in self._dict_log.keys():
            fourth_loss = "Adversarial.Batch.Loss"
        vals = [epoch]
        # Here, we are converting the tensors to a list of floats.
        for each in [cur_loss_reconst, cur_kld, cur_ce, tmloss]:
            if torch.is_tensor(each):
                vals.append(each.item())
            else:
                vals.append(each)
        vals.extend([accval])
        vals.append(cur_reg_recon_loss.item())
        # Here, we are are reading the lists from the dictionary and appending
        # the new values to the end of the list. We then update the dictionary
        # with the new list.
        for i in range(len(vals)):
            curlist = self._dict_log[self.keys[i]]
            try:
                curlist.append(round(vals[i], 3))
            except (RuntimeError, TypeError, NameError, ValueError):
                curlist.append(vals[i])
            self._dict_log[self.keys[i]] = curlist
