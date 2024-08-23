"""
Base trainer class implementation.

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

from typing import Optional

from orion import analyzers, concepts, evaluators, loggers
from torch import nn, optim
from torch.utils import data


class Trainer:
    """A highly reusable and easily overloaded / extended class for model
    training, made so by strict reliance on only modular interfaces for the
    model, loss function, optimizer, data, logging, and analysis components.

    I.e. If a different logger or loss function is desired, the trainer can
    simply be instatiated with a different logger or loss function, and the
    training loop / infrastructure never has to be rewritten across multiple
    experimental notebooks.

    Note: This class may be heavily extended (or divided). Potentially needed
    extensions include: devices (if want to parallelize the model with
    DistributedDataParallel, or will subclass a parallel version), logging
    infrastructure, more optimization details (optimizer, learning rate
    scheduler), period evaluation on validation / test data during training,
    etc., verbose/stdout options for tqdm printout, different classes / methods
    for full vs. minibatch training.

    Thougts: This class organization is not finalized, and may be reworked with
    hooks/callbacks. This is because of the following questions: Is the current
    method of analyzing / logging only at the end of every epoch sufficient?
    Post every epoch is the most important, but if want to analyze / log
    something else internally. Could this be elegantly done with some kind of
    hooks oris that asking too much and supporting that level of arbitrary
    introspection should never be needed, if some other introspection is really
    needed then that just means specific, separate features should be added
    instead?
    """

    def __init__(
        self,
        model: nn.Module,
        loss_func: concepts.LossFunction,
        optimizer: optim.Optimizer,
        train_dataloader: data.DataLoader,
        cfg: concepts.TrainerConfig = concepts.TrainerConfig(),
        analyzer: analyzers.Analyzer = analyzers.DefaultAnalyzer(),
        logger: loggers.Logger = loggers.DefaultLogger(),
        evaluator: Optional[evaluators.Evaluator] = None,
    ):
        self._model = model
        self._loss_func = loss_func
        self._optimizer = optimizer
        self._train_dataloader = train_dataloader
        self._analyzer = analyzer
        self._logger = logger
        self._cfg = cfg
        self._evaluator = evaluator

    def train(self, epochs: int) -> None:
        """Training loop.

        The model passed upon instatiation will be trained for the desired
        number of epochs with the given dataloader, trainer config, and if
        specified in the trainer config, additionally automatically evaluated
        periodically during training.

        Args:
            epochs: Number of epochs to train.
        Returns:
            None.
        Raises:
            ValueError: Invalid BatchMode configured.
        """
        self._model.train()
        train_impl = None
        if self._cfg.batch_mode == concepts.BatchMode.FULLBATCH:
            train_impl = self._train_fullbatch
        elif self._cfg.batch_mode == concepts.BatchMode.MINIBATCH:
            train_impl = self._train_minibatch
        else:
            raise ValueError(
                "BatchMode not supported by current Trainer implementation."
            )
        if self._evaluator is not None and self._cfg.eval_freq is not None:
            for epoch in range(epochs):
                train_impl()
                # Evaluate with desired frequency. Add 1 to eval with desired
                # frequency taking into account 0-indexing.
                if (epoch > 0) and ((epoch + 1) % self._cfg.eval_freq == 0):
                    self._evaluator.eval()
        else:
            for _ in range(epochs):
                train_impl()

    def _train_fullbatch(self) -> None:
        """Full batch training.

        Update the weights after all batches. One epoch of training. Batch
        results are automattically buffered into the analyzer."""
        self._optimizer.zero_grad()
        for batch in self._train_dataloader:
            batch_results = self._compute_batch(concepts.Batch(*batch))
            self._analyzer.buffer(batch_results.__dict__)
        self._update_weights()

    def _train_minibatch(self) -> None:
        """Mini-batch training.

        Update the weights after each batch. One epoch of training. Batch
        results are automattically buffered into the analyzer."""
        for batch in self._train_dataloader:
            self._optimizer.zero_grad()
            batch_results = self._compute_batch(concepts.Batch(*batch))
            self._analyzer.buffer(batch_results.__dict__)
            self._update_weights()

    def _update_weights(self) -> None:
        """Update model weights.

        Each time weights are updated the analyzer is automatically triggered
        and analysis results logged."""
        self._optimizer.step()
        logdata = self._analyzer.analyze_buffer()
        self._logger.log(logdata)

    def _compute_batch(self, batch: concepts.Batch) -> concepts.BatchResult:
        """Execute all computation for a batch.
        Perform the forward and backward propagation for a batch,
        given the model and loss function passed to the Trainer instance.

        Args:
            batch: The forward propagation inputs.

        Returns:
            The forward and propagation outputs.
        """
        # Forward propogation.
        pred = self._model(batch.x)
        loss = self._loss_func(pred, batch.y)
        # Backward propagation.
        loss.backward()
        return concepts.BatchResult(pred, loss)
