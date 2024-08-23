""" This module contains the evaluator class for the Orion model. 

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

import numpy as np
import pandas as pd
import torch
from orion import analyzers, concepts, loggers
from orion.utils.modelparams import ModelParams
from orion.utils.preprocessing import adjust_dimensions
from orion.utils.utils import compute_log_lib_params
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.metrics import accuracy_score, average_precision_score
from torch import nn
from torch.utils import data


class OrionEvaluator(ModelParams):
    """Evaluator for the Orion model. This class is responsible for evaluating
    the model on a given dataset. It also contains the logic for computing
    metrics and logging them.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_func: concepts.LossFunction,
        eval_dataloader: data.DataLoader,
        logger: Optional[loggers.Logger] = None,
        dict_params: Optional[dict] = None,
        analyzer: analyzers.Analyzer = analyzers.OrionLossAnalyzer(),
    ):
        """Orion Model Trainer object
        Args:
            model(): PyTorch model
            loss_func: Loss function
            eval_dataloader: PyTorch dataloader for evaluation
            logger: Logger object
            dict_params: Dictionary of model parameters. It gets added to the
            attributes by `ModelParams.__init__`
            analyzer: Analyzer object. It is used to buffer metrics and
            aggregate them.
        Methods:
            eval: Applies the model to obtain predictions
            and computes the loss and metrics"""
        if dict_params is None:
            dict_params = {}
        ModelParams.__init__(self, dict_params)
        self._eval_dataloader = eval_dataloader
        self.anchor_obj = self._eval_dataloader.anchor_obj
        self.rng = dict_params.get("rng", np.random.default_rng(42))
        self.mixed_precision = dict_params.get("mixed_precision", False)
        self.add_regression_reconst = dict_params.get(
            "add_regression_reconst", False
        )
        # During evaluation, we do not use generative sampling
        # So, use_generative_sampling should be set to False
        self.use_generative_sampling = False
        self.generative_samples_num = 0
        # Number of rounds for the triplet margin
        self.tm_rounds = dict_params.get("tm_rounds", 4)

        self.model = model

        self._loss_func = loss_func
        self.batch_loss_name = "Triplet.Margin.Loss"
        self._analyzer = analyzer
        self.logger = logger

    def eval(
        self,
        epoch: Optional[int] = None,
        skip_setting_to_eval_mode: bool = False,
    ):
        """Applies the model to obtain predictions and computes the loss and
        metrics.
        Args:
            epoch: Current epoch number. It is used for logging.
            skip_setting_to_eval_mode: If True, the model is not set to
            evaluation mode. This is useful when we want to evaluate the model
            and add uncertainty to the predictions.
        Returns:
            perfdf: Pandas dataframe containing the metrics.
        """
        # We set the model to evaluation mode, which turns off dropout and
        # batch normalization. In addition, it also turns off the gradient
        # calculation.
        if not skip_setting_to_eval_mode:
            self.model.eval()
        # iterate over batches
        for dict_inputs in self._eval_dataloader:
            # Calculate the metrics for the minibatch
            batch_results = self._compute_batch(dict_inputs)
            # Adjusting the batch results to be a dictionary and adding the
            # index
            batch_results = batch_results.__dict__
            batch_results["idx"] = dict_inputs["idx"].reshape(-1)
            # Buffer the results
            self._analyzer.buffer(batch_results)
        # aggregate the results across minibatches
        aggresult = self._analyzer.analyze_buffer(
            self.loss_scalers, len(self._eval_dataloader)
        )
        # compute metrics based on the aggregated results
        # First, making the postmat into a dataframe
        perfdf = pd.DataFrame(aggresult["postmat"])
        # Depending on how many classes there are, we adjust the columns
        # and compute the metrics
        if len(perfdf.columns) == 2:
            perfdf.columns = ["Normal.Prob", "Cancer.Prob"]
            select_col = perfdf.columns[1]
            perfdf["Prediction"] = aggresult["celltype_preds"]
            perfdf["Label"] = aggresult["celltype_resps"]
            apscore = average_precision_score(
                perfdf["Label"], perfdf[select_col]
            )
            accscore = accuracy_score(perfdf["Label"], perfdf["Prediction"])
        elif len(perfdf.columns) > 2:
            # This is for the case where there are more than 2 classes, such
            # as in the case of the multiple cancer types
            num_classes = perfdf.shape[1]
            perfdf.columns = [
                f"Class.{each}.Prob" for each in range(num_classes)
            ]
            perfdf["Prediction"] = aggresult["celltype_preds"]
            perfdf["Label"] = aggresult["celltype_resps"]
            apscores = []
            accscores = []
            for j, each in enumerate(pd.unique(aggresult["celltype_resps"])):
                select_col = perfdf.columns[j]
                apscore = average_precision_score(
                    perfdf["Label"] == each, perfdf[select_col]
                )
                accscore = accuracy_score(perfdf["Label"], perfdf["Prediction"])
                apscores.append(apscore)
                accscores.append(accscore)
            apscore = np.mean(apscores)
            accscore = np.mean(accscore)
        else:
            # If the number of classes is 1, then we only have the predicted
            # probability
            perfdf.columns = ["Predicted"]
            perfdf["Response"] = aggresult["celltype_resps"]
            apscore = metrics.r2_score(perfdf["Response"], perfdf["Predicted"])
            accscore, _ = pearsonr(perfdf["Predicted"], perfdf["Response"])
        # Creating the output dictionary
        dict_out = {
            "Reconstruction.loss": aggresult["cur_loss_reconst"],
            "KLD": aggresult["cur_kld"],
            "BCE": aggresult["cur_ce"],
            self.batch_loss_name: aggresult["cur_tm"],
            "Accuracy": accscore,
            "AP.score": apscore,
            "Regression.Reconstruction.Loss": aggresult["cur_reg_recon_loss"],
        }
        # If logger is available, log the results
        if self.logger is not None:
            self.logger.log(
                epoch,
                dict_out["Reconstruction.loss"],
                dict_out["KLD"],
                dict_out["BCE"],
                dict_out[self.batch_loss_name],
                dict_out["Accuracy"],
                dict_out["Regression.Reconstruction.Loss"],
            )
        return (
            aggresult["reconst"],
            aggresult["mumat"],
            aggresult["sd2mat"],
            dict_out,
            perfdf,
        )

    def _compute_batch(self, dict_inputs, other_exp=None, other_mir=None):
        # reshape if necessary
        dict_inputs = adjust_dimensions(dict_inputs)
        idxs = dict_inputs["idx"].tolist()

        # library_manual
        mir_ar = dict_inputs["smtensor"].detach().cpu().numpy()
        local_l_mean, local_l_var = compute_log_lib_params(mir_ar, self.rng)

        oncrna_tensor = dict_inputs["onctensor"]
        mir_tensor = dict_inputs["smtensor"]
        if other_mir is not None:
            mir_tensor = torch.from_numpy(other_mir).float().to(self.device)

        if other_exp is not None:
            oncrna_tensor = torch.from_numpy(other_exp).float().to(self.device)
        # train
        # setting the outdict
        outdict, _ = self.model(
            oncrna_tensor,
            x_lib=mir_tensor,
            evaluation_mode=True,
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
                    ) = self._eval_dataloader.dataset.load_from_dataloader(
                        pos_idxs, neg_idxs
                    )
                    outdict_pos, _ = self.model(
                        dict_poses["onctensor"],
                        x_lib=dict_poses["smtensor"],
                        evaluation_mode=True,
                    )
                    outdict_neg, _ = self.model(
                        dict_negs["onctensor"],
                        x_lib=dict_negs["smtensor"],
                        evaluation_mode=True,
                    )
                    outdict_pos_list.append(outdict_pos["qz_m"])
                    outdict_neg_list.append(outdict_neg["qz_m"])
        # obtain k-mer loss and add it to loss_1. These need to be fed to the
        # loss function in addition to the other parameters
        # Add the extra parameters to calculate the loss. We are keeping
        # the parameters in a dictionary to make it easier to pass to the
        # loss function. These parameters are not included in the forward
        # step output dictionaries. We could revise the forward step to
        # include these parameters, but it would make the forward step
        # output dictionaries unnecessarily large.
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
            # sample_loss_scaler used for calculating weighted loss
            "sample_loss_scaler": dict_inputs.get("sample_loss_scaler", None),
            "use_generative_sampling": self.use_generative_sampling,
            "generative_samples_num": self.generative_samples_num,
        }
        # Calculate the loss
        # Loss_5 (regression based reconstruction loss) comes after the loss_1,
        # loss_2, loss_3, loss_4, loss because it does not always need to be
        # calculated
        loss_1, loss_2, loss_3, loss_4, loss, loss_5 = self._loss_func(
            outdict, parameters_dict
        )
        # Calculate the posterior mean and variance and matrix of
        # cell type predictions
        ct_pred = outdict["ctpred"]
        if self.predict_classes and self.num_classes > 1:
            postmat = torch.softmax(ct_pred, dim=1).detach().cpu().numpy()
            one_hot_resp = (
                torch.max(dict_inputs["onehottensor"], 1)[1]
                .to(self.device)
                .long()
            )
            one_hot_pred = torch.max(ct_pred, 1)[1]
            celltype_resps = one_hot_resp.detach().cpu().numpy()
            celltype_preds = one_hot_pred.detach().cpu().numpy()
        elif self.predict_classes and self.num_classes == 1:
            postmat = ct_pred.detach().cpu().numpy()
            one_hot_resp = dict_inputs["onehottensor"]
            one_hot_pred = ct_pred
            celltype_resps = one_hot_resp.detach().cpu().numpy()
            celltype_preds = one_hot_pred.detach().cpu().numpy()

        del outdict_pos_list
        del outdict_neg_list
        del dict_inputs
        del parameters_dict
        del local_l_mean, local_l_var
        # Return the results
        return concepts.OrionTuneBatchResult(
            loss_1,
            loss_2,
            loss_3,
            loss_4,
            loss_5,
            loss,
            celltype_resps.reshape(-1),
            celltype_preds.reshape(-1),
            postmat,
            outdict["px_scale"].cpu().detach().numpy(),
            outdict["qz_m"].cpu().detach().numpy(),
            outdict["qz_v"].cpu().detach().numpy(),
        )
