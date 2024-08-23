"""Module providing Functions for processing/ training

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

from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from orion.utils.xgboost_functions import perform_xgb
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


def find_sens_cutoff(df_in, fpr):
    """Find the cutoff for a given fpr
    Args:
        df_in: A pd.DataFrame with at least two columns:
               'score' (continuous) and 'label' (binary).
        fpr: A float, desired False Positive Rate.
    Returns:
        cutoff: A float
    """

    assert "label" in df_in.columns, "label column not found"
    assert "score" in df_in.columns, "score column not found"

    fprs, tprs, cutoffs = roc_curve(df_in["label"], df_in["score"])

    diff_vals = np.abs(fprs - fpr)
    idx_best = np.where(diff_vals == np.min(diff_vals))[0]

    print(
        f"""Cutoff {cutoffs[idx_best - 1]}: specificity of
        {1 - fprs[idx_best - 1]} and sensitivity of {tprs[idx_best - 1]}"""
    )
    # outcutoffs = cutoffs[idx_best]
    return cutoffs[idx_best[-1]]


def remove_batch_features(
    cpmdf, y_array, train_index, min_auc=0.7, max_rounds=10
):
    """
    Find the top features using xgboost.
    Args:
        cpmdf (pd.DataFrame): dataframe of cpm values
        y_array (np.array): array of labels
        train_index (np.array): array of training indices
        min_auc (float): minimum auc to be reached before stopping
        max_rounds (int): number of maximum xgboost runs
    Return:
        list_features (list): list of batchy features
    """
    train_index_1, tune_index_1 = train_test_split(train_index, test_size=0.35)
    cur_auc = 1.0
    j = 1
    list_removed_features = []
    while cur_auc > min_auc:
        cpmdf_select_t = cpmdf.T.loc[
            np.setdiff1d(cpmdf.columns, list_removed_features)
        ].T
        _, scoredf, auc_score = perform_xgb(
            cpmdf_select_t, y_array, train_index_1, tune_index_1
        )
        cur_auc = auc_score
        selected_features = list(
            scoredf["oncRNA"][scoredf["feature.importance"] > 0]
        )
        list_removed_features = np.union1d(
            selected_features, list_removed_features
        )
        print(
            f"""Round {j} of xgboost removing {len(list_removed_features)}
            features AUC = {auc_score}"""
        )
        if j >= max_rounds:
            print(f"Stopping after {j} rounds")
            print(f"Reached AUC of {cur_auc}")
            cur_auc = 0
        j += 1
    return list_removed_features


def add_zero_inflation(
    oncrna_ar: np.ndarray,
    frac_dropout: float = 0.75,
    rng: np.random.default_rng = np.random.default_rng(42),
):
    newar = np.zeros(oncrna_ar.shape)
    for i in range(oncrna_ar.shape[0]):
        idx_add = rng.choice(
            np.where(oncrna_ar[i, :] != 0)[0],
            int(oncrna_ar.shape[0] * (1 - frac_dropout)),
            replace=False,
        )
        newar[i, idx_add] = np.round(abs(oncrna_ar[i, idx_add]))
    return newar


def load_pytorch_model(net, outpath):
    state_dict = torch.load(outpath)
    try:
        net.load_state_dict(state_dict)
    except (
        RuntimeError,
        TypeError,
        NameError,
        ValueError,
        AttributeError,
    ):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        try:
            net.load_state_dict(new_state_dict)
        except (
            RuntimeError,
            TypeError,
            NameError,
            ValueError,
            AttributeError,
        ):
            try:
                net.load_state_dict(state_dict, strict=True)
            except (
                RuntimeError,
                TypeError,
                NameError,
                ValueError,
                AttributeError,
            ):
                net.load_state_dict(new_state_dict, strict=True)
    return net


def order_df_by_features(countdf, oncrna_names):
    """
    Order a dataframe by a list of features.
    Will also add 0 for missing oncrna_names.

    Args:
        countdf (pd.DataFrame): dataframe to order
        oncrna_names (list): list of features to order by
    Returns:
        pd.DataFrame: ordered dataframe
    """
    idx_present = np.isin(oncrna_names, countdf.columns)
    if np.sum(idx_present) != len(oncrna_names):
        print(np.setdiff1d(oncrna_names, countdf.columns))
        assert np.all(
            np.isin(oncrna_names, countdf.columns)), "Some features not found"
    out_df = countdf[oncrna_names]
    return out_df


def save_pytorch_model(net, outpath, optimizer=None):
    if torch.cuda.is_available():
        checkpoint = {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "amp": amp.state_dict()
        }
        torch.save(checkpoint, outpath)
    else:
        torch.save(net, outpath)


def enforce_nonzero(
    in_tensor: torch.tensor,
    mean_noise: float = 5,
    sd_noise: float = 1,
    rng: np.random.default_rng = np.random.default_rng(42),
):
    """
    Enforces non-zero values in a tensor by adding a small amount of noise

    Args:
        in_tensor: torch tensor
        mean_noise: mean of the noise
        sd_noise: standard deviation of the noise
    Returns:
        in_tensor: torch tensor with non-zero values
    """
    # in_tensor_cpu
    in_tensor_cpu = in_tensor.detach().cpu()
    # index of non-zero entries
    idx_nonzero = np.where(in_tensor_cpu == 0)
    # random small noise
    epsilon_dist = np.abs(
        rng.normal(loc=mean_noise, scale=sd_noise, size=in_tensor_cpu.shape)
    )
    epsilon_dist = torch.from_numpy(epsilon_dist)
    epsilon_dist = epsilon_dist.type(in_tensor_cpu.dtype)
    in_tensor_cpu[idx_nonzero] = epsilon_dist[idx_nonzero]
    in_tensor = in_tensor_cpu.to(in_tensor.device)
    return in_tensor


def compute_log_lib_params(mir_ar: np.ndarray, rng: np.random.default_rng):
    """Computes the library parameters for the model.

    Args:
        mir_ar: mir_ar count matrix of shape (num_samples, num_features)

    Returns:
        log_l_mean_tensor: torch.Tensor of shape (num_samples, 1)
        log_l_var_tensor: torch.Tensor of shape (num_samples, 1)
    """
    # Prior of log library per mini batch
    log_onc_content = np.log(np.sum(mir_ar, axis=1))
    log_l_mean = np.mean(log_onc_content)
    log_l_var = np.var(log_onc_content)

    # convert to array to have one value per sample
    log_l_mean_ar = np.repeat(log_l_mean, mir_ar.shape[0]).reshape(-1, 1)
    log_l_var_ar = np.repeat(log_l_var, mir_ar.shape[0]).reshape(-1, 1)

    # moving to torch
    log_l_mean_tensor = torch.from_numpy(log_l_mean_ar).float()
    log_l_mean_tensor = enforce_nonzero(log_l_mean_tensor, rng=rng)
    log_l_var_tensor = torch.from_numpy(log_l_var_ar).float()
    log_l_var_tensor = enforce_nonzero(log_l_var_tensor, rng=rng)
    return log_l_mean_tensor, log_l_var_tensor
