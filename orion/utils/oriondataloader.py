"""Module providing OrionDataLoader class

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

from typing import List, Optional

import numpy as np
import numpy.typing as npt
import torch
from orion.utils.anchorfinder import AnchorFinder
from orion.utils.preprocessing import adjust_dimensions
from orion.utils.utils import enforce_nonzero
from torch.utils.data import Dataset


class OrionDataLoader(Dataset):
    """PyTorch data loader for Orion"""

    def __init__(
        self,
        oncmat: npt.ArrayLike,
        smmat: npt.ArrayLike,
        batchmat: npt.ArrayLike,
        onehotmat: npt.ArrayLike,
        oncmat_loss: npt.ArrayLike,
        batchmat_train: npt.ArrayLike = None,
        device: torch.device = None,
        study_name: Optional[List] = None,
        patient_id: Optional[List] = None,
        sample_loss_scaler: Optional[npt.ArrayLike] = None,
        rng: np.random.default_rng = np.random.default_rng(42),
    ):
        """Initialize the OrionDataLoader class.
        Args:
            oncmat: numpy array (samples, oncRNAs), dtype float
            smmat: numpy array (samples, smRNA), dtype float
            batchmat: numpy array (samples, batch identity), dtype int
            onehotmat: numpy array (samples, num_labels), dtype float
            oncmat_loss: numpy array
            batchmat_trai: numpy array
            device: torch device
            study_name: list of study names
            patient_id: list of patient ids
            sample_loss_scaler: numpy array of integers
             as sample loss scalers
            rng: random number generator

        Returns:
            dict_data: dictionary of the dataset
            Contains the following keys:
                onctensor: torch tensor of oncRNA expression counts
                smtensor: torch tensor of smRNA expression counts
                batchtensor: torch tensor of batch identity
                onehottensor: torch tensor of one-hot encoded labels
                lib_pos: torch tensor of log library size of positive anchors
                lib_neg: torch tensor of log library size of negative anchors
                cur_xlib_pos: torch tensor of smRNA expression of positive
                    anchors
                cur_xlib_neg: torch tensor of smRNA expression of negative
                    anchors
                local_l_mean: torch tensor of mean of oncRNA expression of
                    training samples
                local_l_var: torch tensor of variance of oncRNA expression of
                    training samples
                oncrna_tensor_pos: torch tensor of oncRNA expression of positive
                    anchors
                oncrna_tensor_neg: torch tensor of oncRNA expression of negative
                    anchors
                onctensor_loss: torch tensor of oncRNA expression only for
                    loss calculation (if different than onctensor)
                batchtensor_train: torch tensor of batch identity only for
                    loss calculation (if different than batchtensor)
                idx: index of the samples in the output
                sample_loss_scaler: torch tensor of sample loss scalers
            Attributes:
            anchor_obj: instance of AnchorFinder

        """

        self.oncmat = oncmat
        self.smmat = smmat
        self.batchmat = batchmat
        self.batchmat_train = batchmat_train
        if batchmat_train is None:
            self.batchmat_train = self.batchmat
        self.oncmat_loss = oncmat_loss
        self.onehotmat = onehotmat
        self.patient_id = patient_id
        self.study_name = study_name
        if sample_loss_scaler is None:
            # initiate weights as ones to have no impact
            self.sample_loss_scaler = np.ones(oncmat.shape[0])
        else:
            assert (
                sample_loss_scaler.shape[0] == oncmat.shape[0]
            ), "sample_loss_scaler must have same length as oncmat"
            self.sample_loss_scaler = sample_loss_scaler
        self.dict_data = {
            "oncrna_ar": oncmat,
            "smmat": smmat,
            "batch_list": batchmat,
            "onehotar": onehotmat,
            "study_name": study_name,
            "patient_id": patient_id,
        }
        select_vars = []
        for each in ["batch_list", "study_name", "patient_id"]:
            if self.dict_data[each] is not None:
                select_vars.append(each)
        print(f"Anchor finder will use {select_vars}")
        self.anchor_obj = AnchorFinder(self.dict_data, select_vars, rng)
        self.rng = rng
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

    def __len__(self):
        return self.oncmat.shape[0]

    def __getitem__(self, idx):
        """
        Returns a dictionary of the dataset.
        Args:
            idx: index of the sample
        Returns:
            outdict: dictionary of the dataset
                Contains the following keys:
                    onctensor: torch tensor of oncRNA expression counts
                    smtensor: torch tensor of smRNA expression counts
                    libtensor: torch tensor of library size
                    batchtensor: torch tensor of batch identity
                    onehottensor: torch tensor of one-hot encoded labels
                    local_l_mean: torch tensor of mean of oncRNA expression of
                    training samples
                    local_l_var: torch tensor of variance of oncRNA expression
                    of training samples
                    onctensor_loss: torch tensor of oncRNA expression only for
                    loss calculation (if different than onctensor)
                    batchtensor_train: torch tensor of batch identity only for
                    loss calculation (if different than batchtensor)
                    idx: index of the samples in the output
                    sample_loss_scaler: torch tensor of sample loss scalers
        """
        if isinstance(idx, int):
            idx = np.array([idx])
        # main indices
        onctensor_loss = torch.from_numpy(self.oncmat_loss[idx]).float()
        onctensor = torch.from_numpy(self.oncmat[idx]).float()
        smtensor = torch.from_numpy(self.smmat[idx]).float()
        batchtensor = torch.from_numpy(self.batchmat[idx]).long()
        batchtensor_train = torch.from_numpy(self.batchmat_train[idx]).long()
        onehottensor = torch.from_numpy(self.onehotmat[idx]).float()
        sample_loss_scaler = torch.from_numpy(
            self.sample_loss_scaler[idx]
        ).float()
        # library_manual
        local_l_mean = np.mean(
            np.apply_along_axis(np.mean, 1, self.oncmat[idx, :])
        ).reshape(1, -1)
        local_l_mean = torch.from_numpy(local_l_mean).float()
        local_l_mean = enforce_nonzero(local_l_mean, rng=self.rng)
        local_l_var = np.var(
            np.apply_along_axis(np.sum, 1, self.oncmat[idx, :])
        ).reshape(1, -1)
        local_l_var = torch.from_numpy(local_l_var).float()
        local_l_var = enforce_nonzero(local_l_var, rng=self.rng)
        # output
        outdict = {}
        keys = [
            "onctensor",
            "smtensor",
            "batchtensor",
            "onehottensor",
            "local_l_mean",
            "local_l_var",
            "onctensor_loss",
            "batchtensor_train",
            "idx",
            "sample_loss_scaler",
        ]
        vals = [
            onctensor,
            smtensor,
            batchtensor,
            onehottensor,
            local_l_mean,
            local_l_var,
            onctensor_loss,
            batchtensor_train,
            idx,
            sample_loss_scaler,
        ]
        for i, key in enumerate(keys):
            if torch.is_tensor(vals[i]):
                outdict[key] = vals[i].to(self.device)
            else:
                outdict[key] = vals[i]
        return outdict

    def load_from_dataloader(self, pos_idxs, neg_idxs):
        dict_pos = adjust_dimensions(self[pos_idxs])
        dict_neg = adjust_dimensions(self[neg_idxs])
        return dict_pos, dict_neg
