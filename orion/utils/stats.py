"""Module providing Functions for stats

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
from scipy.stats import chi2_contingency


def get_fisher_motor_fast(vals: np.array) -> float:
    """
    get the p-value for a 2x2 contingency table.

    Args:
        vals (np.array): 4x1 contingency table
    Return:
        p (float): p-value
    """
    # if there are any zeros, chi-square fails
    if np.sum(vals == 0) > 0:
        return 1.0
    else:
        tabdf = np.zeros((2, 2))
        tabdf[0, :] = vals[:2]
        tabdf[1, :] = vals[2:]
        # try:
        _, p, _, _ = chi2_contingency(tabdf)
        return p


def get_fisher_pval_fast(statdf: pd.DataFrame, cancer_names: list) -> np.array:
    """
    get the p-value for a 2x2 contingency table.

    Args:
        statdf (pd.DataFrame): dataframe of statistics
        cancer_names (list): list of cancer names
    Return:
        pvals (np.array): p-values
    """
    countmat = np.zeros((len(cancer_names), statdf.shape[0], 4))
    for i, cancer_name in enumerate(cancer_names):
        countmat[i, :, 0] = np.array(statdf[f"num_{cancer_name}_has"])
        countmat[i, :, 2] = np.array(statdf["num_control_has"])
        countmat[i, :, 1] = np.array(statdf[f"num_{cancer_name}_no"])
        countmat[i, :, 3] = np.array(statdf["num_control_no"])
    pvals = np.apply_along_axis(get_fisher_motor_fast, arr=countmat, axis=2)
    return pvals
