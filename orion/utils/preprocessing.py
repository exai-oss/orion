"""Module providing Functions for preprocessing the data

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

from typing import Dict, List, Optional, Tuple

import joblib

import numpy as np
import numpy.typing as npt

import pandas as pd
import torch
from orion.utils.stats import get_fisher_pval_fast
from sklearn import preprocessing as p
from sklearn.preprocessing import StandardScaler


def frac_pos(vals: npt.ArrayLike) -> float:
    """
    Compute the fraction of positive values in an array
    Args:
        vals (npt.ArrayLike): array of values
    Returns:
        frac_pos (float): fraction of positive values
    """
    num_pos = len(np.where(vals > 0)[0])
    return num_pos / len(vals)


def one_hot(
    index: torch.Tensor, n_cat: int, dtype=torch.float32
) -> torch.Tensor:
    """
    Convert an index to a one-hot encoding
    Args:
        index (torch.Tensor): index to convert
        n_cat (int): number of categories
        dtype (torch.dtype): data type
    Returns:
        onehot (torch.Tensor): one-hot encoded index
    """
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(dtype)


def identity(tensor: torch.Tensor) -> torch.Tensor:
    """
    Identity function
    Args:
        tensor (torch.Tensor): input tensor
    Returns:
        tensor (torch.Tensor): input tensor
    """
    return tensor


def to_onehot(array: npt.ArrayLike) -> npt.ArrayLike:
    """
    Convert an array of values to one-hot encoding.
    Args:
        array (npt.ArrayLike): array of values
    Returns:
        onehotar (npt.ArrayLike): one-hot encoded array
    """
    num_vals = len(np.unique(array))
    onehotar = np.zeros((array.shape[0], num_vals))
    for j, each in enumerate(np.unique(array)):
        i = np.where(array == each)[0]
        onehotar[i, j] = 1
    return onehotar


def get_or_table(
    cpmdf: pd.DataFrame,
    y_array: npt.ArrayLike,
    train_idx: npt.ArrayLike,
    epsilon: Optional[float] = 0.001,
    cur_cancer: Optional[str] = "Cancer",
) -> pd.DataFrame:
    """
    Compute the odds ratio table for a given dataset
    Args:
        cpmdf (pd.DataFrame): dataframe of CPM values
        y_array (npt.ArrayLike): array of labels
        train_idx (npt.ArrayLike): array of training indices
        epsilon (Optional, float): epsilon value for smoothing. Default: 0.001
        cur_cancer (Optional, str): name of cancer. Default: "Cancer"
    Returns:
        statdf (pd.DataFrame): odds ratio table
    """
    if not pd.Series(cpmdf.columns).is_unique:
        raise ValueError("Columns are not unique")

    cpm_ar = np.array(cpmdf)

    # count of controls
    idx_controls = np.intersect1d(np.where(y_array == 0)[0], train_idx)
    statdf = pd.DataFrame(
        {
            "Gene": cpmdf.columns,
            "Control.counts": np.apply_along_axis(
                np.sum, 0, cpm_ar[idx_controls] > 0
            ),
        }
    )
    statdf.index = statdf["Gene"]
    statdf["Control.frac"] = statdf["Control.counts"] / len(idx_controls)

    # count of cancers
    idx_cases = np.intersect1d(np.where(y_array == 1)[0], train_idx)
    cur_idxs = np.union1d(idx_cases, idx_controls)
    statdf[f"Case.{cur_cancer}"] = np.apply_along_axis(
        np.sum, 0, cpm_ar[idx_cases] > 0
    )
    statdf[f"Case.{cur_cancer}.frac"] = statdf[f"Case.{cur_cancer}"] / len(
        idx_cases
    )
    statdf[f"Case.{cur_cancer}.OR"] = (
        statdf[f"Case.{cur_cancer}.frac"] + epsilon
    ) / (statdf["Control.frac"] + epsilon)
    statdf[f"Case.{cur_cancer}.log.OR"] = np.log2(
        statdf[f"Case.{cur_cancer}.OR"]
    )
    # OR
    statdf[f"Case.{cur_cancer}"] = np.apply_along_axis(
        np.sum, 0, cpm_ar[idx_cases] > 0
    )
    statdf[f"Case.{cur_cancer}.frac"] = statdf[f"Case.{cur_cancer}"] / len(
        idx_cases
    )
    statdf[f"Case.{cur_cancer}.OR"] = (
        statdf[f"Case.{cur_cancer}.frac"] + epsilon
    ) / (statdf["Control.frac"] + epsilon)
    statdf[f"Case.{cur_cancer}.log.OR"] = np.log2(
        statdf[f"Case.{cur_cancer}.OR"]
    )
    # Generate counts for chisquarred test
    statdf["num_control_has"] = np.apply_along_axis(
        np.sum,
        axis=0,
        arr=cpmdf.T.loc[statdf["Gene"]].T.iloc[
            np.intersect1d(cur_idxs, np.where(y_array == 0)[0]), :
        ]
        > 0,
    )
    statdf["num_control_no"] = np.apply_along_axis(
        np.sum,
        axis=0,
        arr=cpmdf.T.loc[statdf["Gene"]].T.iloc[
            np.intersect1d(cur_idxs, np.where(y_array == 0)[0]), :
        ]
        == 0,
    )
    statdf[f"num_{cur_cancer}_has"] = np.apply_along_axis(
        np.sum,
        axis=0,
        arr=cpmdf.T.loc[statdf["Gene"]].T.iloc[
            np.intersect1d(cur_idxs, np.where(y_array == 1)[0]), :
        ]
        > 0,
    )
    statdf[f"num_{cur_cancer}_no"] = np.apply_along_axis(
        np.sum,
        axis=0,
        arr=cpmdf.T.loc[statdf["Gene"]].T.iloc[
            np.intersect1d(cur_idxs, np.where(y_array == 1)[0]), :
        ]
        == 0,
    )
    # now add p-values for this subset
    print(f"Adding p-values for {statdf.shape[0]} oncRNAs")
    cancer_names = [cur_cancer]
    pval_mat = get_fisher_pval_fast(statdf, cancer_names)
    pvaldf = pd.DataFrame(pval_mat).T
    pvaldf.columns = cancer_names
    pvaldf.index = statdf.index
    for each_col in cancer_names:
        statdf[f"{each_col}.pValue"] = np.array(pvaldf[each_col])
    return statdf


def scale_train_tune(x_train_input: pd.DataFrame, x_tune_input: pd.DataFrame):
    """
    Scale the training and tuning data using the training data.

    Args:
        x_train_input (pd.DataFrame): Training data.
        x_tune_input (pd.DataFrame): Tuning data.
    Returns:
        x_train (pd.DataFrame): Scaled training data.
        x_tune (pd.DataFrame): Scaled tuning data.
        scaler (sklearn.preprocessing.StandardScaler): Scaler used to scale the
    """
    scaler = StandardScaler().fit(x_train_input)
    x_train = pd.DataFrame(scaler.transform(x_train_input))
    x_tune = pd.DataFrame(scaler.transform(x_tune_input))
    x_train.index = x_train_input.index
    x_tune.index = x_tune_input.index
    x_train.columns = x_train_input.columns
    x_tune.columns = x_tune_input.columns
    return x_train, x_tune, scaler


def split_dict_with_idxs(
    datadict: Dict, idxs: List, names: Optional[List] = None
) -> Dict:
    """
    Split a dictionary of data into two dictionaries based on indices.

    Args:
        datadict (Dict): Dictionary of data.
        idxs (List): List of indices to split the data into.
        names (Optional, List): List of names for the new dictionaries. Default
        is ["Training", "Tuning"].
    Returns:
        outdict (Dict): Dictionary of dictionaries.
    """
    if names is None:
        names = ["Training", "Tuning"]

    keys_leaveout = ["oncrna_names", "smrna_names"]
    num_regs = datadict["oncrna_ar"].shape[0]
    unused_idxs = np.arange(num_regs)
    outdict = {}
    unused_idxs = np.arange(num_regs)
    num_regs = len(unused_idxs)
    for i, cur_idxs in enumerate(idxs):
        outdict[names[i]] = {}
        print(f"Using {cur_idxs.shape[0]} indices for {names[i]}")
        for each_key, each_ar in datadict.items():
            if each_key == "metadf":
                outdict[names[i]][each_key] = each_ar.iloc[cur_idxs, :]
            elif each_key not in keys_leaveout:
                outdict[names[i]][each_key] = np.array(each_ar)[cur_idxs]
            else:
                outdict[names[i]][each_key] = np.array(each_ar)
                print(f"{each_key} is not subsetted")
    return outdict


def filter_by_columns(
    oncrna_ar: npt.ArrayLike,
    all_oncrnas: npt.ArrayLike,
    select_oncrnas: npt.ArrayLike,
) -> npt.ArrayLike:
    """
    Filter the expression array by the oncrnas in select_oncrnas.

    Args:
        oncrna_ar (npt.ArrayLike): Expression array.
        all_oncrnas (npt.ArrayLike): All oncrnas in the expression array.
        select_oncrnas (npt.ArrayLike): oncrnas to select.
    Returns:
        newar (npt.ArrayLike): Filtered expression array.
        shared_oncrnas (npt.ArrayLike): oncrnas that were shared between all_oncrnas
            and select_oncrnas.
    """
    shared_oncrnas, idxs_1, _ = np.intersect1d(
        all_oncrnas, select_oncrnas, return_indices=True
    )
    newar = oncrna_ar[:, idxs_1]
    return newar, shared_oncrnas


def split_and_find_features(
    merged_dict_data: Dict,
    train_idxs: npt.ArrayLike,
    tune_idxs: npt.ArrayLike,
    feature_names: Optional[List] = None,
) -> Tuple[Dict, Dict]:
    """
    Split the data into training and tuning and then filter the features.

    Args:
        merged_dict_data (Dict): Dictionary of data.
        train_idxs (npt.ArrayLike): Indices for training data.
        tune_idxs (npt.ArrayLike): Indices for tuning data.
        feature_names (list): List of feature names.
    Returns:
        dict_train (Dict): Dictionary of training data.
        dict_tune (Dict): Dictionary of tuning data.
    """
    if feature_names is None:
        feature_names = merged_dict_data["oncrna_names"]
    data_dict_splitted = split_dict_with_idxs(
        merged_dict_data, [train_idxs, tune_idxs]
    )
    print(np.array(feature_names).shape)
    for each_key, tempdict in data_dict_splitted.items():
        all_oncrnas = tempdict["oncrna_names"].reshape(-1)
        newar, shared_oncrnas = filter_by_columns(
            tempdict["oncrna_ar"], all_oncrnas, np.array(feature_names)
        )
        tempdict["oncrna_ar"] = newar
        tempdict["oncrna_names"] = shared_oncrnas
        data_dict_splitted[each_key] = tempdict
    dict_train, dict_tune = (
        data_dict_splitted["Training"],
        data_dict_splitted["Tuning"],
    )
    return dict_train, dict_tune


def scale_maxes_mats(mat1: npt.ArrayLike, mat2: npt.ArrayLike) -> npt.ArrayLike:
    """Scales mat1 (microRNA) to maximum value in mat2 (oncRNA)
    Args:
        mat1 (npt.ArrayLike): a numpy array with (usually) microRNA data
        mat2 (npt.ArrayLike): a numpy array with (usually) oncRNA data
    Returns:
        outmat (npt.ArrayLike): a numpy array (transformed mat1)
    """
    maxval_2 = np.max(mat2)
    min_max_scaler = p.MinMaxScaler()
    idx_nonzero_2 = np.where(mat2 > 0)
    min_max_scaler.fit(mat2[idx_nonzero_2].reshape(-1, 1))
    outmat = np.zeros(mat1.shape)
    idx_nonzeros = np.where(mat1 > 0)
    curvals = mat1[idx_nonzeros].reshape(-1, 1)
    if np.max(curvals) > maxval_2:
        newvals = min_max_scaler.transform(curvals).reshape(-1)
        outmat[idx_nonzeros] = newvals
    else:
        outmat = mat1
    outmat = np.ceil(outmat)
    print(f"Max of mat1 changed from {np.max(mat1)} to {np.max(outmat)}")
    return outmat


def extract_features_tumors(
    data_dict_splitted: Dict, ratio: float = 0.5, minimum_frac: float = 0.02
) -> Dict:
    """
    This function is used to extract features from the data. It selects the
    features that are present in at least minimum_frac% of the patients
    and at least ratio% of the log2(tumors/non-tumors).

    Args:
        data_dict_splitted (Dict): Dictionary of data.
        ratio (float): Ratio of tumors to non-tumors to select.
        minimum_frac (float): Minimum fraction of tumors to select.
    Returns:
        data_dict_splitted (Dict): Dictionary of data.
    """
    idx_tumors = np.where(data_dict_splitted["Training"]["onehotar"][:, 1])[0]
    idx_nontumors = np.where(data_dict_splitted["Training"]["onehotar"][:, 0])[
        0
    ]
    data_gene = pd.DataFrame.from_dict(
        {
            "Gene": data_dict_splitted["Training"]["oncrna_names"].reshape(-1),
            "Fraction.Patients": np.apply_along_axis(
                frac_pos, 0, data_dict_splitted["Training"]["oncrna_ar"]
            ),
        }
    )
    data_gene["Fraction.Tumors"] = np.apply_along_axis(
        frac_pos, 0, data_dict_splitted["Training"]["oncrna_ar"][idx_tumors, :]
    )
    data_gene["Fraction.NonTumors"] = np.apply_along_axis(
        frac_pos,
        0,
        data_dict_splitted["Training"]["oncrna_ar"][idx_nontumors, :],
    )
    print(data_gene.head())
    data_gene["Use"] = data_gene["Fraction.Patients"] > minimum_frac
    data_gene["Use"] = np.logical_and(
        data_gene["Fraction.Patients"] > minimum_frac,
        np.log2(
            (data_gene["Fraction.Tumors"] + 0.001)
            / (data_gene["Fraction.NonTumors"] + 0.001)
        )
        > ratio,
    )
    print(f"Will use {len(np.where(data_gene['Use'])[0])} features")

    select_oncrnas = np.array(data_gene[data_gene["Use"]]["Gene"])
    print(
        f"""Using {len(select_oncrnas)}/{data_gene.shape[0]} of oncRNAs expressed
                    in at least {minimum_frac} of patients"""
    )
    print(select_oncrnas.shape)
    for each_key in data_dict_splitted.keys():
        tempdict = data_dict_splitted[each_key]
        oncrna_ar = tempdict["oncrna_ar"]
        all_oncrnas = tempdict["oncrna_names"].reshape(-1)
        _, idxs_1, _ = np.intersect1d(
            all_oncrnas, select_oncrnas, return_indices=True
        )
        newar = oncrna_ar[:, idxs_1]
        print(newar.shape)
        tempdict["oncrna_ar"] = newar
        tempdict["oncrna_names"] = all_oncrnas[idxs_1]
        data_dict_splitted[each_key] = tempdict
    return data_dict_splitted


def convert_logdf_to_plotdf(logdf: pd.DataFrame) -> pd.DataFrame:
    """Converts the logdf to a plotdf
    Args:
        logdf (pd.DataFrame): the logdf
    Returns:
        plotdf (pd.DataFrame): the plotdf
    """
    name_last_loss = "Triplet.Margin.Loss"
    if name_last_loss not in logdf.columns:
        name_last_loss = "Adversarial.Batch.Loss"
    losses = (
        list(logdf["Reconstruction.Loss"])
        + list(logdf["KLD"])
        + list(logdf["CE.Loss"])
        + list(logdf[name_last_loss])
        + list(logdf["Accuracy"])
    )
    losstypes = (
        list(["Reconstruction loss"] * logdf.shape[0])
        + list(["KLD"] * logdf.shape[0])
        + list(["CE.Loss"] * logdf.shape[0])
        + list([name_last_loss] * logdf.shape[0])
        + list(["Accuracy"] * logdf.shape[0])
    )
    return pd.DataFrame(
        {
            "Epoch": list(logdf["Epoch"]) * 5,
            "Values": losses,
            "Dataset": list(logdf["Cohort"]) * 5,
            "Data type": losstypes,
            "Models": list(logdf["Model"]) * 5,
        }
    )


def merge_dict_data(dict_data: Dict) -> Dict:
    """Merges the data from different datasets into one dictionary
    Args:
        dict_data (Dict): Dictionary of data. Each key is the name of the
            dataset and the value is a dictionary with the following keys:
            oncrna_ar (npt.ArrayLike): Expression array.
            x_lib (npt.ArrayLike): smRNA expression array.
            onehotar (npt.ArrayLike): One-hot array.
            library_manual (npt.ArrayLike): Manual library size array.
            oncrna_names (npt.ArrayLike): oncRNA names.
            smrna_names (npt.ArrayLike): Small RNA names.
            metadf (pd.DataFrame): Metadata dataframe (optional).
    Returns:
        outdict (Dict): Dictionary of combined data with the following keys:
            oncrna_ar (npt.ArrayLike): Expression array.
            x_lib (npt.ArrayLike): smRNA expression array.
            onehotar (npt.ArrayLike): One-hot array.
            library_manual (npt.ArrayLike): Manual library size array.
            patient_names (npt.ArrayLike): Patient names.
            oncrna_names (npt.ArrayLike): oncRNA names.
            smrna_names (npt.ArrayLike): Small RNA names.
            batch_list (npt.ArrayLike): Batch list.
            batch_onehot (npt.ArrayLike): One-hot batch list.
            datasetname (npt.ArrayLike): Dataset names.
            metadf (pd.DataFrame): Metadata dataframe (optional).
    """
    # make batch information
    # num_batches = len(dict_data.keys())
    num_samples = 0
    batch_codes = []
    feature_names = []
    feature_names_smrna = []
    patient_names = []
    patient_names_smrna = []
    list_metadfs = []
    list_datanames = []
    batch_id = 0
    for dataname, datadict in dict_data.items():
        print(f"Loading {dataname}")
        oncrna_ar = datadict["oncrna_ar"]
        list_datanames.extend([dataname] * oncrna_ar.shape[0])
        if "metadf" in list(datadict.keys()):
            metadf = datadict["metadf"]
            metadf["Dataset"] = dataname
            list_metadfs.append(metadf)
        patient_names.extend(list(oncrna_ar.index))
        patient_names_smrna.extend(list(datadict["x_lib"].index))
        num_samples += oncrna_ar.shape[0]
        set_new = set(np.array(datadict["oncrna_names"]).reshape(-1))
        adnames = list(set_new - set(feature_names))
        print(f"Adding {len(adnames)} new oncRNAs")
        feature_names.extend(adnames)
        set_new = set(np.array(datadict["smrna_names"]).reshape(-1))
        adnames_smrna = list(set_new - set(feature_names_smrna))
        print(f"Adding {len(adnames_smrna)} new smRNAs")
        feature_names_smrna.extend(adnames_smrna)
        batch_codes.extend([batch_id] * oncrna_ar.shape[0])
        batch_id += 1
    # now merge into one
    oncrna_ar = np.zeros((num_samples, len(feature_names)))
    smrna_ar = np.zeros((num_samples, len(feature_names_smrna)))
    lib_manual = np.zeros((num_samples, 1))
    onehotar_label = np.zeros((num_samples, 2))
    idx_st = 0
    idx_end = 0
    for dataname, datadict in dict_data.items():
        print(f"Merging {dataname}")
        oncrna_ar_temp = datadict["oncrna_ar"]
        # find shared_columns
        _, idxs_1, idxs_2 = np.intersect1d(
            np.array(feature_names),
            np.array(oncrna_ar_temp.columns),
            return_indices=True,
        )
        idx_end = idx_st + oncrna_ar_temp.shape[0]
        oncrna_ar[idx_st:idx_end, idxs_1] = np.array(oncrna_ar_temp)[:, idxs_2]
        smrna_ar_temp = datadict["x_lib"]
        _, idxs_1, idxs_2 = np.intersect1d(
            np.array(feature_names_smrna),
            np.array(smrna_ar_temp.columns),
            return_indices=True,
        )
        smrna_ar[idx_st:idx_end, idxs_1] = np.array(smrna_ar_temp)[:, idxs_2]
        onehotar_label[idx_st:idx_end] = datadict["onehotar"][
            : oncrna_ar_temp.shape[0]
        ]
        lib_manual[idx_st:idx_end] = datadict["library_manual"]
        idx_st = idx_end
    # sort smrna patients by oncRNA patients
    expdf_smrna = pd.DataFrame(smrna_ar)
    expdf_smrna.index = patient_names_smrna
    expdf_smrna = expdf_smrna.loc[patient_names]
    smrna_ar = np.array(expdf_smrna)
    out_metadf = pd.concat(list_metadfs)
    outdict = {
        "oncrna_ar": oncrna_ar,
        "x_lib": smrna_ar,
        "onehotar": onehotar_label,
        "library_manual": lib_manual,
        "patient_names": patient_names,
        "oncrna_names": feature_names,
        "smrna_names": feature_names_smrna,
        "batch_list": npt.ArrayLike(batch_codes),
        "batch_onehot": np.eye(batch_id + 1)[batch_codes],
        "datasetname": npt.ArrayLike(list_datanames),
        "metadf": out_metadf,
    }
    return outdict


def load_multiple_data(dict_datapaths: Dict) -> Dict:
    """Load multiple data from a dictionary of paths and merge them.

    Args:
        dict_datapaths (Dict): Dictionary of paths.

    Returns:
        dict_out (Dict): Dictionary of data.
    """
    dict_data = {}
    for dataname, datapath in dict_datapaths.items():
        print(f"Loading {dataname} from {datapath}")
        dict_data[dataname] = joblib.load(datapath)
    dict_out = merge_dict_data(dict_data)
    return dict_out


def split_dict(
    datadict: Dict,
    ratios: Optional[List] = None,
    names: Optional[List] = None,
    rng: np.random.default_rng = np.random.default_rng(42),
):
    """Split a dictionary of data into multiple dictionaries. Useful for
    splitting data into training and tuning sets.

    Args:
        datadict (Dict): Dictionary of data.
        ratios (List, optional): List of ratios. Defaults to [0.8, 0.2].
        names (List, optional): List of names. Defaults to ["Training",
            "Tuning"].
        rng (Optional, np.random.default_rng(), optional): Random number
            generator.
        Defaults to np.random.default_rng(42).
    Returns:
        outdict (Dict): Dictionary of dictionaries.
    """
    if ratios is None:
        ratios = [0.8, 0.2]
    if names is None:
        names = ["Training", "Tuning"]
    if sum(ratios) > 1:
        raise ValueError("Sum of ratios must be less than or equal to 1")
    outdict = {}
    keys_leaveout = ["oncrna_names", "smrna_names"]
    num_regs = datadict["oncrna_ar"].shape[0]
    unused_idxs = np.arange(num_regs)
    for i in range(len(ratios)):
        outdict[names[i]] = {}
        cur_idxs = rng.choice(
            unused_idxs, int(ratios[i] * num_regs), replace=False
        )
        for each_key, each_ar in datadict.items():
            print(each_key)
            if each_key not in keys_leaveout:
                outdict[names[i]][each_key] = np.array(each_ar)[cur_idxs]
            elif each_key == "metadf":
                outdict[names[i]][each_key] = each_ar.iloc[cur_idxs, :]
            else:
                outdict[names[i]][each_key] = np.array(each_ar)
        unused_idxs = np.setdiff1d(unused_idxs, cur_idxs)
    return outdict


def make_labels_ccle(
    metapath: str, oncrna_ar: npt.ArrayLike, barcodes: npt.ArrayLike
) -> Tuple[npt.ArrayLike, pd.DataFrame, npt.ArrayLike, torch.tensor]:
    """Make labels for CCLE data.

    Args:
        metapath (str): Path to metadata.
        oncrna_ar (npt.ArrayLike): Expression array.
        barcodes (npt.ArrayLike): Barcodes.

    Returns:
        outar (npt.ArrayLike): Expression array.
        outdf (pd.DataFrame): Metadata.
        out_barcodes (npt.ArrayLike): Barcodes.
        one_hot (torch.tensor): One-hot encoded labels.
    """
    metadf = pd.read_csv(metapath, sep="\t", index_col=0)
    if "Site_Primary" in metadf.columns:
        metadf["CellType"] = metadf["Site_Primary"]
        metadf["Barcode"] = metadf.index
    classes = np.unique(list(metadf["CellType"]))
    classes = np.array([each for each in classes if "nan" not in each])
    metadf = metadf[metadf["CellType"].isin(classes)]
    metadf = metadf[metadf["Barcode"].isin(barcodes)]
    _, idx_1, idx_2 = np.intersect1d(
        barcodes, np.array(metadf["Barcode"]), return_indices=True
    )
    outar = oncrna_ar[idx_1, :]
    outdf = metadf.iloc[idx_2, :]
    out_barcodes = np.array(barcodes, dtype="|U64")[idx_1]
    one_hot_df = pd.get_dummies(outdf["CellType"])
    one_hot_tensor = torch.from_numpy(np.array(one_hot_df))
    return outar, outdf, out_barcodes, one_hot_tensor


def adjust_dimensions(dict_inputs: Dict) -> Dict:
    """
    Order a dictionary of OrionDataLoader output
    by batch.
    Makes sure the first dimension is the batch except in some cases.
    Args:
        dict_inputs (Dict): Dictionary of inputs.

    Returns:
        dict_inputs (Dict): Transfomed dictionary of inputs.
    """
    batch_shape = dict_inputs["onctensor"].shape[0]
    for key, val in dict_inputs.items():
        if val is None:
            dict_inputs[key] = val
        elif key == "idx":
            dict_inputs[key] = val.reshape(-1)
        elif val.shape[0] == 1:
            dict_inputs[key] = val
        else:
            dict_inputs[key] = val.reshape(batch_shape, -1)
    return dict_inputs
