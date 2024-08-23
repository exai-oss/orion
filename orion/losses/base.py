""" Basic loss concepts and functionality which are not specific to a particular
loss module.

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

from collections import abc

import torch
from orion import concepts


def combine_losses(loss_functions: abc.Iterable[concepts.LossFunction]):
    def combined_loss(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        loss = 0.0
        for loss_function in loss_functions:
            loss += loss_function(prediction, target)
        return loss

    return combined_loss
