"""
Base evaluator class implementation.

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

from typing import Any

import torch
from orion import analyzers, concepts, loggers
from torch import nn
from torch.utils import data


class Evaluator:
    """
    A module with a simple `eval()` function interface which will evaluate the
    the model it has been given using the loss function and dataloader it has
    been configured with. Analyzers and loggers have basic support as well.

    Note: This class may be modified / extended to change approach for execution
          e.g. perhaps whether to evaluate single-node vs distributed. It could
          also potentially be extended to support different literal methods,
          changing the operations performed on the test dataset during
          evaluation. As of yet it is unclear which way it may skew more.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_func: concepts.LossFunction,
        eval_dataloader: data.DataLoader,
        analyzer: analyzers.Analyzer = analyzers.DefaultAnalyzer(),
        logger: loggers.Logger = loggers.DefaultLogger(),
    ):
        self._model = model
        self._loss_func = loss_func
        self._eval_dataloader = eval_dataloader
        self._analyzer = analyzer
        self._logger = logger

    def eval(self) -> Any:
        """Evaluate model on test dataset.

        Configure model to evaluation mode, disable gradient calculation, and
        evaluate model performance on entire test dataset. Evaluation is
        implemented by the Analyzer of choice, e.g. a LossAnalyzer, the result
        of which is both logged and returned.

        Returns:
            Output of the analyzer referenced by this Evaluator instance, e.g.
            loss summed across batches when using LossAnalyzer.
        """
        self._model.eval()
        with torch.no_grad():
            for batch in self._eval_dataloader:
                batch_results = self._compute_batch(concepts.Batch(*batch))
                self._analyzer.buffer(batch_results.__dict__)
        analyzed_data = self._analyzer.analyze_buffer()
        self._logger.log(analyzed_data)
        return analyzed_data

    def _compute_batch(self, batch: concepts.Batch) -> concepts.BatchResult:
        """Execute forward computation for a batch.
        Perform the forward  propagation for a batch, given the model and loss
        function passed to the Evaluator instance.

        Args:
            batch: The forward propagation inputs.

        Returns:
            The forward and propagation outputs.
        """
        # Forward propogation.
        pred = self._model(batch.x)
        loss = self._loss_func(pred, batch.y)
        return concepts.BatchResult(pred, loss)
