"""Module providing ModelTrainer class and related functions

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
import copy

import numpy as np
import pandas as pd
import torch
from orion import analyzers, concepts, evaluators, loggers
from orion.utils.modelparams import ModelParams
from orion.utils.preprocessing import adjust_dimensions
from orion.utils.utils import compute_log_lib_params
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from torch import optim
from torch.utils import data


class ModelTrainer(ModelParams):
    """Orion Model Trainer"""

    def __init__(
        self,
        model,
        loss_func: concepts.LossFunction,
        optimizer: optim.Optimizer,
        train_dataloader: data.DataLoader,
        logger: loggers.Logger,
        dict_params=None,
        evaluator: Optional[evaluators.Evaluator] = None,
        analyzer: analyzers.Analyzer = analyzers.OrionLossAnalyzer(),
    ):
        """Orion Model Trainer object
        Args:
            model: PyTorch model
            loss_func: Loss function
            optimizer: Optimizer
            train_dataloader: PyTorch dataloader for training data
            logger: Logger
            dict_params: Dictionary of model parameters
            evaluator: Evaluator
            analyzer: Analyzer
        Methods:
            train: Trains the model
        """
        # loading the parameters
        if dict_params is None:
            dict_params = {}
        # setting the parameters using init
        ModelParams.__init__(self, dict_params)
        self.add_regression_reconst = dict_params.get(
            "add_regression_reconst", False
        )
        self.mixed_precision = dict_params.get("mixed_precision", False)
        self.weight_sample_loss = dict_params.get("weight_sample_loss", False)
        self.use_generative_sampling = dict_params.get(
            "use_generative_sampling", True
        )
        self.generative_samples_num = dict_params.get(
            "generative_samples_num", 100
        )
        # loading the datasets and the model
        oncrna_ar = train_dataloader.dataset.oncmat

        self.dict_params = self.print()
        self.model = model
        self.train_dataloader = train_dataloader

        self.anchor_obj = self.train_dataloader.anchor_obj
        self.rng = dict_params.get("rng", np.random.default_rng(42))

        self.tm_rounds = dict_params.get("tm_rounds", 4)
        self._loss_func = loss_func
        # This can be removed and just set to TripletMarginLoss

        self.batch_loss_name = "Triplet.Margin.Loss"
        self._analyzer = analyzer

        self._optimizer = optimizer
        self.sampled_idxs = self.rng.choice(
            np.arange(oncrna_ar.shape[0]), oncrna_ar.shape[0], replace=False
        )
        self.totbatchidx = int(oncrna_ar.shape[0] / self.mini_batch) + 1
        self._evaluator = evaluator

        self._logger = logger

    def train(self, num_epochs=None):
        """Trains the model
        Args:
            num_epochs: Number of epochs to train
        """
        # setting the cur_tune_loss to a random high value
        cur_tune_loss = 1000
        if num_epochs is None:
            num_epochs = self.num_epochs
        for epoch in range(num_epochs):
            # training minibatch
            self._train_minibatch(epoch)
            # logging the results
            training_dict_log = self._logger.get_log()
            training_logdf = pd.DataFrame(training_dict_log)
            print(
                f"Epoch {epoch} {list(training_logdf.iloc[-1, :])} - Training"
            )
            if self._evaluator is not None:
                # evaluating the model if evaluator is provided
                tuning_dict_log = self._evaluator.logger.get_log()
                tuning_logdf = pd.DataFrame(tuning_dict_log)
                new_tune_loss = tuning_dict_log["CE.Loss"][-1]
                if new_tune_loss < cur_tune_loss:
                    cur_tune_loss = new_tune_loss
                    print(f"Best tuning loss at epoch {epoch}")
                    self.best_state_dict = copy.deepcopy(
                        self.model.state_dict()
                    )
                print(
                    f"Epoch {epoch} {list(tuning_logdf.iloc[-1, :])} - Tuning"
                )

    def _train_minibatch(self, epoch):
        """Trains the model for one minibatch
        Args:
            epoch: Epoch number
        """
        self.model.train()
        # TRAINING LOOP
        for dict_inputs in self.train_dataloader:
            # regular backprop
            self._optimizer.zero_grad()  # train
            # computing the batch results
            batch_results = self._compute_batch(dict_inputs, epoch)
            if self.grad_clip_max is not None:
                self._clip_gradients()
            # buffering the results
            self._analyzer.buffer(batch_results.__dict__)
            # updating the weights
            self._update_weights()
        # aggeregate the results from buffer
        aggresult = self._analyzer.analyze_buffer(
            self.loss_scalers, len(self.train_dataloader)
        )
        # logging the aggregates
        self._logger.log(
            epoch,
            aggresult["cur_loss_reconst"],
            aggresult["cur_kld"],
            aggresult["cur_ce"],
            aggresult["cur_tm"],
            aggresult["accval"],
            aggresult["cur_reg_recon_loss"],
        )
        if self._evaluator is not None:
            _ = self._evaluator.eval(epoch=epoch)

    def _update_weights(self) -> None:
        """Update model weights.

        Each time weights are updated the analyzer is automatically triggered
        and analysis results logged."""
        if torch.cuda.is_available() and self.mixed_precision:
            with torch.cuda.amp.autocast():
                self.scaler.step(self._optimizer)
                self.scaler.update()
        else:
            self._optimizer.step()

    def _compute_batch(
        self,
        dict_inputs: dict,
        epoch: int,
    ):
        """Computes the loss for a batch of data
        Args:
            dict_inputs: Dictionary of inputs
            epoch: Epoch number
        Returns:
            batch_results: Batch results
        """
        # reshape if necessary
        dict_inputs = adjust_dimensions(dict_inputs)
        idxs = dict_inputs["idx"].tolist()

        # library_manual
        mir_ar = dict_inputs["smtensor"].detach().cpu().numpy()
        # calculate the local mean of the library for this minibatch
        local_l_mean, local_l_var = compute_log_lib_params(mir_ar, self.rng)

        # not ideal but more readable. This creates a local variable to log
        oncrna_tensor = dict_inputs["onctensor"]
        mir_tensor = dict_inputs["smtensor"]

        # train
        # setting the outdict
        outdict, _ = self.model(
            oncrna_tensor,
            x_lib=mir_tensor,
        )
        outdict_pos_list = []
        outdict_neg_list = []
        if self.use_triplet_loss:
            variables = list(self.anchor_obj.patient_anchor_dict.keys())
            for variable in variables:
                for _ in range(self.tm_rounds):
                    pos_idxs, neg_idxs = self.anchor_obj.get_anchors(
                        idxs, variable
                    )
                    (
                        dict_poses,
                        dict_negs,
                    ) = self.train_dataloader.dataset.load_from_dataloader(
                        pos_idxs, neg_idxs
                    )
                    outdict_pos, _ = self.model(
                        dict_poses["onctensor"],
                        x_lib=dict_poses["smtensor"],
                    )
                    outdict_neg, _ = self.model(
                        dict_negs["onctensor"],
                        x_lib=dict_negs["smtensor"],
                    )
                    outdict_pos_list.append(outdict_pos["qz_m"])
                    outdict_neg_list.append(outdict_neg["qz_m"])

        # obtain k-mer loss and add it to loss_1. These need to be fed to the
        # loss function in addition to the other parameters
        parameters_dict = {
            "local_l_mean": local_l_mean,
            "local_l_var": local_l_var,
            "input_onctensor": dict_inputs["onctensor"],
            "predict_classes": self.predict_classes,
            "num_classes": self.num_classes,
            "use_triplet_loss": self.use_triplet_loss,
            "input_onehottensor": dict_inputs["onehottensor"],
            "pos_qz_m": outdict_pos_list,
            "neg_qz_m": outdict_neg_list,
            "input_batchtensor_train": dict_inputs["batchtensor_train"],
            "l1": self.l1,
            "l2": self.l2,
            "grad_clip_max": self.grad_clip_max,
            "loss_scalers": self.loss_scalers,
            "sample_loss_scaler": dict_inputs.get("sample_loss_scaler", None),
            "use_generative_sampling": self.use_generative_sampling,
            "generative_samples_num": self.generative_samples_num,
            "add_regression_reconst": self.add_regression_reconst,
        }
        # calculate the loss
        loss_1, loss_2, loss_3, loss_4, loss, loss_5 = self._loss_func(
            outdict, parameters_dict
        )
        # cell type prediction
        ct_pred = outdict["ctpred"]
        if self.predict_classes and self.num_classes > 1:
            one_hot_resp = (
                torch.max(dict_inputs["onehottensor"], 1)[1]
                .to(self.device)
                .long()
            )
            one_hot_pred = torch.max(ct_pred, 1)[1]
            if self.use_generative_sampling:
                # if we are using generative sampling, we need to calculate
                # the accuracy for each sample and then average them out
                # to get the final accuracy.
                ctpred_multi = outdict["ctpred.generative"]
                adacc = 0
                for j in range(ctpred_multi.shape[0]):
                    one_hot_pred = torch.max(ctpred_multi[j], 1)[1]
                    curacc = accuracy_score(
                        one_hot_pred.detach().cpu().numpy().reshape(-1),
                        one_hot_resp.detach().cpu().numpy().reshape(-1),
                    )
                    adacc += curacc
                adacc = adacc / max(len(ctpred_multi), 1)
            else:
                adacc = accuracy_score(
                    one_hot_resp.detach().cpu().numpy(),
                    one_hot_pred.detach().cpu().numpy(),
                )
            # calculate accuracy
        elif self.predict_classes and self.num_classes == 1:
            one_hot_resp = dict_inputs["onehottensor"]
            one_hot_pred = ct_pred
            if self.use_generative_sampling:
                # if we are using generative sampling, we need to calculate
                # the accuracy for each sample and then average them out
                # to get the final accuracy.
                ctpred_multi = outdict["ctpred.generative"]
                adacc = 0
                for j in range(ctpred_multi.shape[0]):
                    if j == 0:
                        print("Using z sampling for CE loss")
                    one_hot_pred = ctpred_multi[j]
                    curacc, _ = pearsonr(
                        one_hot_pred.detach().cpu().numpy().reshape(-1),
                        one_hot_resp.detach().cpu().numpy().reshape(-1),
                    )
                    adacc += curacc
                adacc = adacc / max(len(ctpred_multi), 1)
            else:
                adacc, _ = pearsonr(
                    one_hot_pred.detach().cpu().numpy().reshape(-1),
                    one_hot_resp.detach().cpu().numpy().reshape(-1),
                )

        del outdict_pos_list
        del outdict_neg_list
        del dict_inputs
        del parameters_dict
        del local_l_mean, local_l_var
        if torch.cuda.is_available() and self.mixed_precision:
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast():
                self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return concepts.OrionTrainBatchResult(
            loss_1, loss_2, loss_3, loss_4, loss_5, loss, adacc
        )

    def _clip_gradients(self) -> None:
        """Clip gradients.

        Clip the gradients of the model parameters by the maximum gradient norm
        specified in `self.grad_clip_max`.
        """
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_max
        )
