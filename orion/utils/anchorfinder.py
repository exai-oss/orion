"""Module providing AnchorFinder class

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

import warnings
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from scipy.spatial import distance_matrix


class AnchorFinder:
    """Class to find anchors for triplet margin loss. This class is used to
    find anchors for triplet margin loss. It annotates which sample indices
    can be used as positive and negative anchor for each individual sample.
    Example: if a sample is cancer from batch 1, its positive anchors indicate
    all other samples which are cancer but from any batch other than 1,
    and its negative anchors annotate all non-cancer samples from any batch.

    Example:
        {"supplier": {0: {"positive": [idx1p, idx2p, ...],
                      "negative": [idx1n, idx2n, ...]},}},
         "patient_id": {0: {
            "positive": [idx1p_id, idx2p_id, ...],
            "negative": [idx1n_id, idx2n_id, ...]},}},
        ...}, ...}

    Args:
        dict_train (dict): Dictionary of training data
        select_vars (list): List of variables to use for
            anchor finding. Defaults to None.
        seed (int): Random seed. Defaults to 42.
    Returns:
        dict_anchors (dict): Dictionary of
            Variable > sample > anchors
    """

    def __init__(
        self,
        dict_train,
        select_vars=None,
        rng: Optional[np.random.default_rng] = np.random.default_rng(42),
    ):
        if select_vars is None:
            select_vars = ["batch_list", "study_name"]
        self.oncrna_ar = dict_train["oncrna_ar"]
        self.select_vars = select_vars
        for each_var in select_vars:
            setattr(self, each_var, dict_train[each_var])
        self.onehotar = self._fix_onehotar(dict_train["onehotar"])
        self.rng = rng
        self.patient_anchor_dict = self.make_patient_anchor_dict(select_vars)
        self.test_all_indices_exist()

    @staticmethod
    def _fix_onehotar(onehotar: npt.ArrayLike) -> npt.ArrayLike:
        """Fix onehotar if it is not one-hot encoded.
        This is done by dichotomizing the response.
        It will set the top 50% of the response to 1 and the bottom 50% to 0.
        Then it will one-hot encode the response.

        Args:
            onehotar (npt.ArrayLike): one-hot encoded array

        Returns:
            onehotar (npt.ArrayLike): one-hot encoded array
        """
        if len(onehotar.shape) == 1 or onehotar.shape[1] == 1:
            print("Dichotomizing response for anchors")
            # dichotomize samples
            idx_top = np.where(onehotar > np.quantile(onehotar, 0.5))[0]
            idx_bottom = np.setdiff1d(np.arange(onehotar.shape[0]), idx_top)
            onehotar = np.zeros((onehotar.shape[0], 2))
            onehotar[idx_bottom, 0] = 1
            onehotar[idx_top, 1] = 1
        return onehotar

    @staticmethod
    def _find_similarities(
        cur_ar: npt.ArrayLike, idxs: npt.ArrayLike, rng
    ) -> Dict:
        """Find positive and negative anchors for each sample in idxs

        Args:
            cur_ar (npt.ArrayLike): array of samples to find anchors for.
            idxs  (npt.ArrayLike): indices of samples to find anchors for.

        Returns:
            dict_indices (dict): dictionary with keys as indices of samples
        """
        dist_mat = distance_matrix(cur_ar, cur_ar)
        dict_indices = {}
        for i, cur_idx in enumerate(idxs):
            vals = dist_mat[i, :]
            vals_pos = vals[vals > 0]
            idx_sorted_vals = np.argsort(vals_pos)
            idx_distant = idx_sorted_vals[-1]
            if len(idx_sorted_vals) > 5:
                idx_distant = rng.choice(idx_sorted_vals[5:], 1)[0]
            min_vals = min(vals_pos)
            max_vals = max(vals_pos)
            idx_pos = np.where(vals == min_vals)[0]
            idx_neg = np.where(vals == max_vals)[0]
            dict_indices[cur_idx] = {
                # most similar
                "Positive": idxs[idx_pos][0],
                # least similar
                "Negative": idxs[idx_neg][0],
                # least similar ordered sampled top 5
                "Negative_random": idxs[idx_distant],
            }
        return dict_indices

    def get_dict_variables(self, select_vars):
        dict_variables = {}
        for each_var in select_vars:
            values = getattr(self, each_var)
            if len(np.unique(values)) == 1:
                warnings.warn(
                    f"""Expected more than 1 batch for {each_var}
                \nRandomly assigning a second batch"""
                )
                # randomly assign a second batch
                values = self.rng.choice(
                    np.arange(2), len(values), replace=True
                )
                setattr(self, each_var, values)
            dict_variables[each_var] = {"values": getattr(self, each_var)}
        return dict_variables

    def test_all_indices_exist(self):
        for variable, curdict in self.patient_anchor_dict.items():
            list_keys = list(curdict.keys())
            for i in np.arange(self.onehotar.shape[0]):
                if i not in list_keys:
                    print(f"For {variable}, missing index {i}")
                    raise ValueError("Missing index")
        print("All indices exist for all samples")

    def make_patient_anchor_dict(
        self,
        select_vars,
    ) -> Dict:
        """Make dictionary of patient anchors

        Args:
            select_vars (List): List of variables to use for anchors

        Returns:
            Dict: Dictionary with keys as patient indices and values as
            dictionaries with keys "Positive" and "Negative" and values as
            indices of positive and negative anchors
        """
        dict_variables = self.get_dict_variables(select_vars)
        outdict = {}
        for variable, vardict in dict_variables.items():
            # for each variable, find the positive and negative anchors by label
            print(f"Adding anchors for {variable}")
            addict = {}
            values = vardict["values"]
            unique_values = np.unique(values)
            if len(unique_values) == 1:
                print(f"Only 1 value for {variable}\n\nSkipping!")
                continue
            for each_batch in unique_values:
                # looking into each unique value of variable
                print(f"Batch {each_batch}")
                dict_unmatched = {}
                batch_idxs = np.where(values == each_batch)[0]
                other_batch_idxs = np.where(values != each_batch)[0]
                assert (
                    len(other_batch_idxs) > 0
                ), "Must have > 1 batch for TM loss"
                for each_label in range(self.onehotar.shape[1]):
                    # positive anchor must be from a different batch
                    label_idxs = np.where(self.onehotar[:, each_label] == 1)[0]
                    idxs_use = np.intersect1d(
                        batch_idxs,
                        label_idxs,
                    )
                    if len(idxs_use) > 0:
                        idxs_matched = np.intersect1d(
                            other_batch_idxs,
                            label_idxs,
                        )
                        if len(idxs_matched) > 0:
                            shuffled_idxs = self.rng.choice(
                                idxs_matched, idxs_use.shape[0], replace=True
                            )
                        else:
                            print(
                                f"""For class {each_label}, batch {each_batch},
                                no matches"""
                            )
                            # generate similarity matrix of samples in
                            # batch_idxs
                            dict_unmatched = self._find_similarities(
                                self.oncrna_ar[batch_idxs,],
                                batch_idxs,
                                rng=self.rng,
                            )
                            # pick the sample from the same batch with the most
                            # distance
                            shuffled_idxs = [
                                dict_unmatched[each]["Negative_random"]
                                for each in batch_idxs
                            ]
                        # negative anchor
                        idxs_unmatched = np.intersect1d(
                            other_batch_idxs,
                            np.where(self.onehotar[:, each_label] != 1)[0],
                        )
                        if len(idxs_unmatched) > 0:
                            shuffled_idxs_neg = self.rng.choice(
                                idxs_unmatched, idxs_use.shape[0], replace=True
                            )
                        else:
                            dict_unmatched = self._find_similarities(
                                self.oncrna_ar[batch_idxs,],
                                batch_idxs,
                                rng=self.rng,
                            )
                            shuffled_idxs_neg = [
                                dict_unmatched[each]["Negative_random"]
                                for each in batch_idxs
                            ]
                        for i in range(idxs_use.shape[0]):
                            curdict = {}
                            curdict["Positive"] = self.rng.choice(
                                shuffled_idxs, 10, replace=True
                            )
                            curdict["Negative"] = self.rng.choice(
                                shuffled_idxs_neg, 10, replace=True
                            )
                            if idxs_use[i] in addict:
                                print(
                                    """
                                    Probably due to one batch with only one 
                                    group
                                    """
                                )
                                print(
                                    f"""
                                    Error at label {each_label} batch
                                    {each_batch}
                                    """
                                )
                                raise ValueError("Unexpected index overlap")
                            addict[idxs_use[i]] = curdict
            outdict[variable] = addict
        return outdict

    def get_anchors(self, patient_idxs, variable_name="batch_list"):
        """Each time get_anchors is called,
        it should return different indices.
        """
        if isinstance(patient_idxs, int):
            patient_idxs = [patient_idxs]
        pos_idxs = np.zeros(len(patient_idxs), dtype=int)
        neg_idxs = np.zeros(len(patient_idxs), dtype=int)
        for i, each in enumerate(patient_idxs):
            if each not in self.patient_anchor_dict[variable_name].keys():
                print(
                    f"""Maximum patient value: 
                    {max(self.patient_anchor_dict[variable_name].keys())}"""
                )
                raise ValueError("Patient idx not in anchor dict")
            poses = self.patient_anchor_dict[variable_name][each]["Positive"]
            self.rng.shuffle(poses)
            pos_idxs[i] = poses[0]
            poses = self.patient_anchor_dict[variable_name][each]["Negative"]
            self.rng.shuffle(poses)
            neg_idxs[i] = poses[0]
        return pos_idxs, neg_idxs
