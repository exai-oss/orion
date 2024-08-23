"""Module providing Functions for XGboost

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

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score


def find_top_features_xgb_multiclass(
    cpmdf, y_array, train_idx, tune_idx, top_k=1000
):
    """
    Find the top features using xgboost.

    Args:
        cpmdf (pd.DataFrame): dataframe of cpm values
        y_array (np.array): array of labels
        train_idx (np.array): array of training indices
        tune_idx (np.array): array of tuning indices
        top_k (int): number of top features to return
    Return:
        top_features (list): list of top features
        scoredf (pd.DataFrame): dataframe of scores
        probs (pd.DataFrame): dataframe of probabilities
    """
    traindf = cpmdf.iloc[train_idx, :]
    traindf = traindf.iloc[
        :, np.where(np.apply_along_axis(np.sum, arr=traindf > 0, axis=0) > 0)[0]
    ]
    print(traindf.shape)
    y_train = y_array[train_idx]
    model = xgboost.XGBClassifier(n_estimators=100, n_jobs=-1)
    model.fit(traindf, y_train)
    scoredf = pd.DataFrame(
        {
            "feature.importance": model.feature_importances_,
            "oncRNA": traindf.columns,
        }
    )
    scoredf.sort_values("feature.importance", inplace=True, ascending=False)
    scoredf["rank"] = np.arange(scoredf.shape[0])
    top_features = list(scoredf["oncRNA"])[:top_k]
    # xgboost model make predictions
    print("Making xgboost predictions")
    tunedf = cpmdf.T.loc[traindf.columns].T.iloc[tune_idx, :]
    y_tune = y_array[tune_idx]
    probs = pd.DataFrame(model.predict_proba(tunedf))
    probs["Model"] = "XGBoost"
    probs.index = tunedf.index
    probs["label"] = y_tune
    return top_features, scoredf, probs


def feature_selection_xgb(cpmdf, train_idx, tune_idx, y_array, n_times=2):
    """
    Perform feature selection using xgboost.

    Args:
        cpmdf (pd.DataFrame): dataframe of cpm values
        train_idx (np.array): array of training indices
        tune_idx (np.array): array of tuning indices
        y_array (np.array): array of labels
        n_times (int): number of times to perform feature selection
    Return:
        out_features (list): list of selected features
    """
    out_features = []
    for j in np.arange(n_times):
        print(f"Round {j + 1} of feature selection")
        _, scoredf, _ = find_top_features_xgb_multiclass(
            cpmdf.T.loc[np.setdiff1d(cpmdf.columns, out_features)].T,
            y_array,
            train_idx,
            tune_idx,
            top_k=5000,
        )
        selected_features = list(
            scoredf["oncRNA"][scoredf["feature.importance"] > 0]
        )
        out_features = np.union1d(out_features, selected_features)
    return out_features


def perform_xgb(cpmdf, y_array, train_idx, tune_idx):
    """
    Find the top features using xgboost.
    Args:
        cpmdf (pd.DataFrame): dataframe of cpm values
        y_array (np.array): array of labels
        train_idx (np.array): array of training indices
        tune_idx (np.array): array of tuning indices
    Return:
        model (xgboost): xgboost model
        scoredf (pd.DataFrame): dataframe of scores
        auc_score (float): tuning set auc
    """
    traindf = cpmdf.iloc[train_idx, :]
    traindf = traindf.iloc[
        :, np.where(np.apply_along_axis(np.sum, arr=traindf > 0, axis=0) > 0)[0]
    ]
    print(traindf.shape)
    y_train = y_array[train_idx]
    model = xgboost.XGBClassifier(n_estimators=100, n_jobs=-1)
    model.fit(traindf, y_train)
    scoredf = pd.DataFrame(
        {
            "feature.importance": model.feature_importances_,
            "oncRNA": traindf.columns,
        }
    )
    scoredf.sort_values("feature.importance", inplace=True, ascending=False)
    scoredf["rank"] = np.arange(scoredf.shape[0])
    # xgboost model make predictions
    print("Makeing xgboost predictions")
    tunedf = cpmdf.T.loc[traindf.columns].T.iloc[tune_idx, :]
    y_tune = y_array[tune_idx]
    probs = pd.DataFrame(model.predict_proba(tunedf))
    probs.index = tunedf.index
    auc_scores = []
    for j in range(probs.shape[1]):
        auc_scores.append(roc_auc_score(y_tune == j, probs[j]))
    auc_score = np.mean(auc_scores)
    probs["Model"] = "XGBoost"
    probs["label"] = y_tune
    # calculate auc
    return model, scoredf, auc_score
