""" Functions for VAE model.

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

import collections
from typing import Optional

import joblib
import random
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from orion import evaluators, loggers, losses
from orion.models.vae import VAE
from orion.trainers.orion_trainer import ModelTrainer
from orion.utils.modelparams import OrionConfig
from orion.utils.oriondataloader import OrionDataLoader
from orion.utils.preprocessing import split_and_find_features, to_onehot
from orion.utils.utils import load_pytorch_model, order_df_by_features


def process_orion_config_dict_into_dataclass(
    dict_train: dict,
    dict_params: Optional[dict] = None,
    num_epochs: int = 300,
) -> OrionConfig:
    """
    Process the parameters dictionary into a dataclass.
    In addition, sets the random seeds throughout the code.

    Args:
        dict_train (dict): dictionary of training data
        dict_params (dict): dictionary of parameters
        num_epochs (int): number of epochs
    Return:
        OrionConfig (dataclass): dataclass of parameters
    """
    if dict_params is None:
        dict_params = {}
    # oncrna_ar is the expression matrix. xlib is the library matrix
    # their shapes are used to set the default values
    oncrna_ar = dict_train["oncrna_ar"]
    xlib = dict_train["smrnamat"]
    # n_hidden is the number of hidden units
    # setting default n_hidden to min([1500, int(oncrna_ar.shape[1] / 2)])
    n_hidden = min([1500, int(oncrna_ar.shape[1] / 2)])
    # setting the random seeds
    random_seed = dict_params.get("random_seed", 42)
    rng = dict_params.get("rng", np.random.default_rng(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # setting the device
    if "device" not in dict_params.keys():
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            warnings.warn("CUDA is not available. Using CPU. ")
    else:
        if torch.cuda.is_available():
            device = dict_params.get("device", torch.device("cuda:0"))
        else:
            device = torch.device("cpu")
            warnings.warn("CUDA is not available. Using CPU. ")
    if device == torch.device("cpu"):
        scaler = dict_params.get("scaler", None)
    else:
        scaler = dict_params.get("scaler", torch.cuda.amp.GradScaler())
    default_dict_params = {
        "n_input": dict_params.get("n_input", oncrna_ar.shape[1]),
        "n_input_lib": dict_params.get("n_input_lib", xlib.shape[1]),
        "n_hidden": dict_params.get("n_hidden", n_hidden),
        "n_features": dict_params.get("n_features", oncrna_ar.shape[1]),
        "num_epochs": dict_params.get("num_epochs", num_epochs),
        "rng": rng,
        "device": device,
        "scaler": scaler,
        "random_seed": random_seed,
    }
    default_dict_params.update(dict_params)
    # asserting that the loss scalers is a list
    assert isinstance(default_dict_params["loss_scalers"], list)
    # making sure there are 5 loss scalers
    if len(default_dict_params["loss_scalers"]) < 5:
        for _ in range(5 - len(default_dict_params["loss_scalers"])):
            default_dict_params["loss_scalers"].append(1)
    orion_config = OrionConfig(**default_dict_params)
    for this_key in orion_config.keys():
        if this_key not in dict_params.keys() and this_key != "rng":
            print(f"Using default value of {this_key}")
        elif (
            this_key == "rng"
            and dict_params.get("random_seed", None) is None
            and dict_params.get("rng", None) is None
        ):
            print(f"Using random number generator {random_seed}")
    return orion_config


def get_net(
    dict_train: dict,
    orion_config: OrionConfig,
    pretrained_net: Optional[nn.Module] = None,
):
    """
    Get the VAE network.

    Args:
        dict_train (dict): dictionary of training data
        orion_config (OrionConfig): dictionary of parameters
        loss_scalers (list): list of loss scalers
        pretrained_net (nn.Module): pre-trained network (default None)
    Return:
        net (VAE): VAE network
    """
    device = orion_config.get(
        "device",
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    )
    if pretrained_net is not None:
        assert isinstance(
            pretrained_net, nn.Module
        ), "pretrained_model must be nn.Module"
        net = pretrained_net
        print("Using pre-trained model")
    else:
        net = VAE(
            n_input=orion_config["n_input"],
            n_labels=dict_train["onehot_ar"].shape[1],
            n_hidden=int(orion_config["n_hidden"]),
            n_layers=orion_config["n_layers"],
            n_latent=int(orion_config["num_lvs"]),
            dropout_rate=orion_config["dp"],
            n_input_lib=orion_config["n_input_lib"],
            inject_lib=orion_config.get("inject_lib", True),
            inject_lib_method=orion_config.get("inject_lib_method", "multiply"),
            add_batchnorm_lib=orion_config.get("add_batchnorm_lib", False),
            use_generative_sampling=orion_config.get(
                "use_generative_sampling", True
            ),
            generative_samples_num=orion_config.get(
                "generative_samples_num", 100
            ),
            add_regression_reconst=orion_config.get(
                "add_regression_reconst", False
            ),
            log_variational=orion_config.get("log_variational", True),
        )
    net.to(device)
    return net


def add_prediction_variance(
    perfdf, model_train_obj, evaluator, dict_tune, num_exps=25
):
    """
    Adds variance of prediction to the output.

    Args:
        perfdf (pd.DataFrame): performance dataframe
        model_train_obj (ModelTrainer): model trainer object
        evaluator (Evaluator): evaluator object
        dict_tune (dict): dictionary of tuning data
        num_exps (int): number of experiments
    Return:
        perfdf (pd.DataFrame): performance dataframe with uncertainty
    """
    # we need to set the model to train mode to get uncertainty
    initial_state = model_train_obj.model.training
    model_train_obj.model.train()
    pred_ar = np.zeros(
        (num_exps, perfdf.shape[0], dict_tune["onehot_ar"].shape[1])
    )
    for i in range(num_exps):
        # we need to set the model to train mode to get uncertainty, so
        # skip_setting_to_eval_mode is set to True
        _, _, _, _, perfdf_temp = evaluator.eval(skip_setting_to_eval_mode=True)
        for j in np.arange(pred_ar.shape[2]):
            pred_ar[i, :, j] = perfdf_temp.iloc[:, j]
    var_preds = np.var(pred_ar, axis=0)
    for j in np.arange(pred_ar.shape[2]):
        perfdf[f"Class.{j}.predictionVariance"] = var_preds[:, j]
    perfdf["Max.Prediction.Variance"] = np.max(var_preds, axis=1)
    model_train_obj.model.train(mode=initial_state)
    return perfdf


class VaePredictor(nn.Module):
    """
    VAE predictor for use in conjunction wih SHAP score calculation.
    """

    def __init__(self, net, idx_onc, idx_sm):
        """
        Variational autoencoder predictor
        Assumes 0:idx_onc are oncRNA matrix
        Assumes idx_onc:idx_sm are smRNA matrix
        assumes idx_sm: are batch
        Args:
            net (nn.Module): VAE model
            idx_onc (int): index of oncRNA
            idx_sm (int): index of smRNA
        """
        super().__init__()
        self.idx_onc = idx_onc
        self.idx_sm = idx_sm
        self.net = net
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_onc = x[:, : self.idx_onc]
        x_sm = x[:, self.idx_onc : self.idx_sm]
        outdict, _ = self.net(
            x_onc, x_lib=x_sm, evaluation_mode=True, library_manual=None
        )
        ct_pred = self.softmax(outdict["ctpred"])
        return ct_pred


def prepare_data_dict(
        oncmat: pd.DataFrame,
        mirdf: pd.DataFrame,
        metadf: pd.DataFrame,
        modeldict: dict,
        batch_column: str = "supplier",
        label_column: str = "cohort"
):
    """
    Prepare data dictionary for training or prediction.

    Args:
        oncmat (pd.DataFrame): OncRNA expression matrix
        mirdf (pd.DataFrame): miRNA expression matrix
        metadf (pd.DataFrame): Metadata dataframe
        modeldict (dict): Model dictionary
        batch_column (str): Batch column
        label_column (str): Label column
    
    Returns:
        dict_data (dict): Dictionary of data objects
    """
    required_keys = [
        "name_oncrna_features",
        "name_mirna_features",
    ]
    for key in required_keys:
        if key not in modeldict.keys():
            raise ValueError(f"{key} is required in modeldict")
    oncmat = oncmat[modeldict["name_oncrna_features"]]
    mirdf = mirdf[modeldict["name_mirna_features"]]
    dict_data = {
        "oncrna_ar": np.array(oncmat),
        "smrnamat": np.array(mirdf),
        "batch_list": pd.factorize(metadf[batch_column])[0],
        "patient_names": np.array(oncmat.index),
        "oncrna_names": oncmat.columns,
        "smrna_names": mirdf.columns,
        "onehot_ar": to_onehot(pd.factorize(metadf[label_column])[0]),
    }
    return dict_data



def make_predictions(
    modelpath: str,
    parampath: str,
    metadf: pd.DataFrame,
    mirdf: pd.DataFrame,
    oncmat: pd.DataFrame,
    dict_params: Optional[dict] = None,
    mini_batch: Optional[int] = None,
    label_column: Optional[str] = "cohort",
    report_shap: Optional[bool] = True,
    batch_column: Optional[str] = "supplier"
):
    """
    Make predictions on a test set using a trained model.

    Args:
        modelpath (str): Path to the parameter file (.joblib file with
            dict_train and dict_tune)
        parampath (str): Path to the model file (.pt file)
        metadf (pd.DataFrame): Metadata dataframe
        mirdf (pd.DataFrame): miRNA expression dataframe (count)
        oncmat (np.ndarray): OncRNA expression matrix (count)
        dict_params (dict): Dictionary of model parameters
        report_shap (bool): Whether to compute and report the shap dictionary
    Returns:
        perfdf (pd.DataFrame): Performance dataframe
        model_trainer_obj (OrionTrainer): OrionTrainer object
        dict_outputs (Dictionary):
            SHAP: Dictionary containing SHAP information
                summarized by class (shapdf), and available for each sample
                (Shap.values)
                Shap.values is a tuple of [class x sample x oncRNA]
            Embeddings: Dataframe of embeddings
    """
    if dict_params is None:
        dict_params = {}
    oncmat = oncmat.loc[metadf.index]
    mirdf = mirdf.loc[metadf.index]
    modeldict = joblib.load(modelpath)
    dict_data = prepare_data_dict(
        oncmat, mirdf, metadf, modeldict,
        batch_column=batch_column, label_column=label_column
    )
    orion_config = process_orion_config_dict_into_dataclass(
        dict_train=dict_data,
        dict_params=dict_params,
    )
    if mini_batch is None:
        mini_batch = orion_config.get("mini_batch", 128)

    net = get_net(
        dict_data,
        orion_config=orion_config,
    )
    rng = orion_config.get("rng", np.random.default_rng(42))
    net = load_pytorch_model(net, parampath)
    # net.load_state_dict(torch.load(parampath))
    net.eval()
    print("Successfully loaded model and parameters.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # if add_regression_reconst
    add_regression_reconst = orion_config.get("add_regression_reconst", False)
    criterion_regression = torch.nn.SmoothL1Loss().to(device)

    valid_dataloader, _ = get_dataloaders(
        dict_data, dict_data, mini_batch=mini_batch, device=device, rng=rng
    )
    # assign loss weight
    weight_list = []
    for j in np.arange(dict_data["onehot_ar"].shape[1]):
        num_samples = np.sum(dict_data["onehot_ar"][:, j])
        total_samples = dict_data["onehot_ar"].shape[0]
        weight_list.append(num_samples / total_samples)
        print(f"Class {j}: {num_samples}/{total_samples}: {weight_list[-1]}")
    weight_tensor = torch.from_numpy(np.log2(1 / np.array(weight_list))).to(
        device
    )
    weight_tensor = weight_tensor / torch.max(weight_tensor)
    batch_loss_name = "Triplet.Margin.Loss"
    training_dict_log = get_logger_dictionary(batch_loss_name)
    training_logger = loggers.OrionLogger(training_dict_log)
    tuning_dict_log = get_logger_dictionary(batch_loss_name)

    tuning_logger = loggers.OrionLogger(tuning_dict_log)
    loss_func = losses.OrionLoss(
        model=net,
        device=device,
        criterion_class=torch.nn.CrossEntropyLoss(weight=weight_tensor),
        weight_sample_loss=orion_config.get("weight_sample_loss", False),
        add_regression_reconst=add_regression_reconst,
        criterion_regression=criterion_regression,
        gene_likelihood=orion_config.get("gene_likelihood", "zinb"),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    evaluator = evaluators.OrionEvaluator(
        model=net,
        loss_func=loss_func,
        eval_dataloader=valid_dataloader,
        logger=tuning_logger,
        dict_params=orion_config,
    )

    model_trainer_obj = ModelTrainer(
        model=net,
        loss_func=loss_func,
        optimizer=optimizer,
        train_dataloader=valid_dataloader,
        logger=training_logger,
        dict_params=orion_config,
        evaluator=evaluator,
    )
    model_trainer_obj.model.eval()

    data_obj_valid = OrionDataLoader(
        dict_data["oncrna_ar"],
        np.array(dict_data["smrnamat"]),
        dict_data["batch_list"].reshape(-1, 1),
        dict_data["onehot_ar"],
        dict_data["oncrna_ar"],
        device=device,
        rng=rng,
    )
    valid_dataloader = DataLoader(
        data_obj_valid, batch_size=mini_batch, shuffle=False
    )
    valid_dataloader.anchor_obj = data_obj_valid.anchor_obj
    valid_dataloader.oncmat = dict_data["oncrna_ar"]
    evaluator_valid = evaluators.OrionEvaluator(
        model=net,
        loss_func=loss_func,
        eval_dataloader=valid_dataloader,
        dict_params=orion_config,
    )
    if orion_config.get("use_best", False):
        evaluator_valid.model.load_state_dict(model_trainer_obj.best_state_dict)
        model_trainer_obj.model.load_state_dict(
            model_trainer_obj.best_state_dict
        )
    _, embed_mat, _, _, perfdf = evaluator_valid.eval()
    # convert embed_mat to a data frame
    # columns are latent variables; name as LV.x
    embed_df = pd.DataFrame(
        embed_mat,
        index=dict_data["patient_names"],
        columns=[f"LV.{each + 1}" for each in np.arange(embed_mat.shape[1])],
    )
    perfdf.index = dict_data["patient_names"]
    perfdf = add_prediction_variance(
        perfdf, model_trainer_obj, evaluator_valid, dict_data
    )

    # obtain SHAP
    dict_shap = {}
    if report_shap:
        dict_shap = compute_shap(dict_data, net, device=device)

    dict_outputs = {"SHAP": dict_shap, "Embeddings": embed_df}
    return perfdf, model_trainer_obj, dict_outputs


def compute_shap(
    dict_valid: dict,
    net: nn.Module,
    device: torch.device = torch.device("cuda:0"),
):
    """
    Compute SHAP values.

    Args:
        dict_valid (dict): dictionary of validation data
    Return:
        dict_shap (dict): dictionary of SHAP values
    """
    import shap

    dict_shap = {}
    # check if all the variables are present
    variables_used_in_shap = [
        "oncrna_names",
        "oncrna_ar",
        "smrnamat",
        "patient_names",
    ]
    for each in variables_used_in_shap:
        if each not in dict_valid.keys():
            raise ValueError(f"{each} is required to calculate SHAP values")
    # extract data from dict_valid
    dict_shap["feature_names"] = dict_valid["oncrna_names"]
    oncrna_ar = dict_valid["oncrna_ar"]
    xlib = dict_valid["smrnamat"]
    oncrna_ar_all = np.concatenate(
        (dict_valid["oncrna_ar"], dict_valid["smrnamat"]), axis=1
    )
    dict_shap["sample_names"] = dict_valid["patient_names"]
    # conatenate data for VaePredictor
    tensor_oncrna_ar_all = torch.from_numpy(oncrna_ar_all).to(device).float()
    idx_onc = oncrna_ar.shape[1]
    idx_sm = idx_onc + xlib.shape[1]
    # compure SHAP scores
    model_for_shap = VaePredictor(net, idx_onc, idx_sm)
    shap_explainer = shap.DeepExplainer(model_for_shap, tensor_oncrna_ar_all)
    shap_values = shap_explainer.shap_values(tensor_oncrna_ar_all)
    # store SHAP scores in dict_shap
    dict_shap["Shap.values"] = shap_values
    dict_shap["Input"] = oncrna_ar_all
    dict_shap["shap_sums"] = np.sum(np.abs(shap_values[1]), axis=0)
    shap_values = dict_shap["Shap.values"]
    shapdf = pd.DataFrame({"oncRNAs": dict_valid["oncrna_names"]})
    shapdf.index = shapdf["oncRNAs"]
    # Name SHAP values of each class
    for j in range(len(shap_values)):
        shapdf[f"Shap.Class.{j}"] = np.sum(
            np.abs(shap_values[j][:, : (shapdf.shape[0])]), axis=0
        )
    dict_shap["shapdf"] = shapdf
    return dict_shap


def make_dict_tune():
    oncrna_ar = torch.rand(400, 2200).numpy()
    xlib = oncrna_ar.copy()
    xlib[xlib > 0.8] = 1
    xlib[xlib <= 0.8] = 0
    oncrna_ar[oncrna_ar > 0.6] = 1
    oncrna_ar[oncrna_ar <= 0.6] = 0
    ct_id = torch.empty(oncrna_ar.shape[0], dtype=torch.long).random_(2).numpy()
    onehot_ct = torch.nn.functional.one_hot(torch.from_numpy(ct_id))
    batch_id = (
        torch.empty(oncrna_ar.shape[0], dtype=torch.long).random_(5).numpy()
    )
    lib_manual = np.apply_along_axis(np.sum, arr=oncrna_ar, axis=1)
    dict_tune = {
        "oncrna_ar": oncrna_ar,
        "x_lib": torch.from_numpy(xlib),
        "library_manual": lib_manual.reshape(-1, 1),
        "onehot_ar": onehot_ct,
        "batch_id": batch_id.reshape(-1, 1),
    }
    return dict_tune


def get_dataloaders(
    dict_train,
    dict_tune,
    mini_batch=128,
    device=None,
    shuffle_train_dataloader=True,
    verbose=True,
    rng: Optional[np.random.Generator] = np.random.default_rng(42),
):
    """
    Obtain dataloaders for training and tuning

    Args:
        dict_train (dict): Dictionary containing training data
        dict_tune (dict): Dictionary containing tuning data
        mini_batch (int): Mini batch size
        device (torch.device): Device to use for training
        shuffle_train_dataloader (bool): Whether to shuffle training dataloader
        verbose (bool): Whether to print verbose output
    Returns:
        train_data_object (Dataset): Training data object
        tune_data_object (Dataset): Tuning data object
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data_object = OrionDataLoader(
        dict_train["oncrna_ar"],
        dict_train["smrnamat"],
        dict_train["batch_list"].reshape(-1, 1),
        dict_train["onehot_ar"],
        dict_train["oncrna_ar"],
        dict_train["batch_list"],
        device=device,
        study_name=dict_train.get("study_name", None),
        patient_id=dict_train.get("patient_id", None),
        sample_loss_scaler=dict_train.get("sample_loss_scaler", None),
        rng=rng,
    )
    tune_data_object = OrionDataLoader(
        dict_tune["oncrna_ar"],
        np.array(dict_tune["smrnamat"]),
        dict_tune["batch_list"].reshape(-1, 1),
        dict_tune["onehot_ar"],
        dict_tune["oncrna_ar"],
        device=device,
        study_name=dict_tune.get("study_name", None),
        patient_id=dict_tune.get("patient_id", None),
        rng=rng,
    )
    if verbose:
        print(dict_train["oncrna_ar"].shape)
        print(dict_tune["oncrna_ar"].shape)
    train_dataloader = DataLoader(
        train_data_object,
        batch_size=mini_batch,
        shuffle=shuffle_train_dataloader,
    )
    tune_dataloader = DataLoader(
        tune_data_object, batch_size=mini_batch, shuffle=False
    )
    train_dataloader.anchor_obj = train_data_object.anchor_obj
    tune_dataloader.anchor_obj = tune_data_object.anchor_obj
    return train_dataloader, tune_dataloader


def train_orion_model(
    merged_dict_data: dict,
    train_idxs: npt.ArrayLike,
    tune_idxs: npt.ArrayLike,
    feature_names: npt.ArrayLike,
    dict_params: Optional[dict] = None,
    pretrained_net: Optional[nn.Module] = None,
    **other_params,
):
    """
    Train a model on the training set and report on the tune set.

    Args:
        merged_dict_data (dict): Dictionary of data
            Required keys include:
                "oncrna_ar": A numpy array of expression data
                "oncrna_names": A list of gene names
                "smrnamat": A numpy array of smrna data
                "smrna_names": A list of smrna gene names
                "onehot_ar": A numpy array of one-hot encoded labels
                "batch_list": A list of batch labels
                "patient_names": A list of patient names
        train_idxs (np.array): Indices of training set
        tune_idxs (np.array): Indices of tune set
        feature_names (list): List of feature names
        dict_params (dict): Dictionary of parameters (see OrionConfig)
        pretrained_net (nn.Module): Pre-trained network (default None)
        **other_params: Other parameters to pass to dict_params dictionary
    Returns:
        dict: Dictionary of results
    """

    if dict_params is None:
        dict_params = {}
    for key, val in other_params.items():
        dict_params[key] = val
    assert (
        len(np.intersect1d(train_idxs, tune_idxs)) == 0
    ), "Train/tune indices shouldn't ovelap"

    # We need to set the rng here to ensure reproducibility throughout
    # the function
    if "rng" in dict_params.keys():
        # if rng is provided, we will use the rng
        rng = dict_params["rng"]
    elif "random_seed" in dict_params.keys():
        # if random_seed is provided, we will use the random_seed
        rng = np.random.default_rng(dict_params["random_seed"])
        dict_params["rng"] = rng
    else:
        # if neither is provided, we will use the default rng~42
        rng = np.random.default_rng(42)
        dict_params["rng"] = rng
    if dict_params.get("bag", False):
        print("Sampling with replacement from training set")
        train_idxs = rng.choice(train_idxs, train_idxs.shape[0], replace=True)
    dict_train, dict_tune = split_and_find_features(
        merged_dict_data, train_idxs, tune_idxs, feature_names
    )

    orion_config = process_orion_config_dict_into_dataclass(
        dict_train=dict_train,
        dict_params=dict_params,
    )

    weight_list = []
    for j in np.arange(dict_train["onehot_ar"].shape[1]):
        num_samples = np.sum(dict_train["onehot_ar"][:, j])
        total_samples = dict_train["onehot_ar"].shape[0]
        weight_list.append(num_samples / total_samples)
        print(f"Class {j}: {num_samples}/{total_samples}: {weight_list[-1]}")

    device = orion_config.device

    criterion_regression = torch.nn.SmoothL1Loss().to(device)

    net = get_net(
        dict_train,
        orion_config,
        pretrained_net=pretrained_net,
    )
    print(net)
    train_dataloader, tune_dataloader = get_dataloaders(
        dict_train,
        dict_tune,
        orion_config["mini_batch"],
        device=device,
        rng=rng,
    )
    weight_tensor = torch.from_numpy(np.log2(1 / np.array(weight_list))).to(
        device
    )
    weight_tensor = weight_tensor / torch.max(weight_tensor)
    # weight_tensor[0] = torch.max(weight_tensor) * 1.2
    print(f"Using {weight_tensor}")
    print("Creating ModelTrain object")
    batch_loss_name = "Triplet.Margin.Loss"
    training_dict_log = get_logger_dictionary(batch_loss_name)
    training_logger = loggers.OrionLogger(training_dict_log)
    tuning_dict_log = get_logger_dictionary(batch_loss_name)

    tuning_logger = loggers.OrionLogger(tuning_dict_log)
    loss_func = losses.OrionLoss(
        model=net,
        device=device,
        criterion_class=torch.nn.CrossEntropyLoss(weight=weight_tensor),
        weight_sample_loss=orion_config.get("weight_sample_loss", False),
        add_regression_reconst=orion_config.get(
            "add_regression_reconst", False
        ),
        criterion_regression=criterion_regression,
        gene_likelihood=orion_config.get("gene_likelihood", "zinb"),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    evaluator = evaluators.OrionEvaluator(
        model=net,
        loss_func=loss_func,
        eval_dataloader=tune_dataloader,
        logger=tuning_logger,
        dict_params=orion_config,
    )

    model_train_obj = ModelTrainer(
        model=net,
        loss_func=loss_func,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        logger=training_logger,
        dict_params=orion_config,
        evaluator=evaluator,
    )
    model_train_obj.train()
    # recreating the tune and train dataloader to generate the mudf
    # This time, we do not want to shuffle the train dataloader
    # These will be used to generate plots
    train_dataloader, tune_dataloader = get_dataloaders(
        dict_train,
        dict_tune,
        orion_config["mini_batch"],
        device=device,
        shuffle_train_dataloader=False,
        verbose=False,
        rng=rng,
    )
    evaluator_train = evaluators.OrionEvaluator(
        model=net,
        loss_func=loss_func,
        eval_dataloader=train_dataloader,
        dict_params=orion_config,
    )
    evaluator_valid = evaluators.OrionEvaluator(
        model=net,
        loss_func=loss_func,
        eval_dataloader=tune_dataloader,
        dict_params=orion_config,
    )
    if orion_config.get("use_best", False):
        evaluator_train.model.load_state_dict(model_train_obj.best_state_dict)
        evaluator_valid.model.load_state_dict(model_train_obj.best_state_dict)
        model_train_obj.model.load_state_dict(model_train_obj.best_state_dict)
    (
        _,
        mumat_train,  # only keeping the mumat for train
        _,
        _,
        _,
    ) = evaluator_train.eval()
    (
        reconst_tune,
        mumat_tune,
        sd2mat_tune,
        dict_perf_tune,
        perfdf_tune,
    ) = evaluator_valid.eval()
    mudf_train = pd.DataFrame(mumat_train)
    mudf_train.index = dict_train["patient_names"]
    mudf_train.columns = [
        f"LV.{each}" for each in np.arange(mudf_train.shape[1])
    ]

    perfdf_tune.index = dict_tune["patient_names"]
    perfdf_tune["label"] = perfdf_tune["Label"]
    perfdf_tune = add_prediction_variance(
        perfdf_tune, model_train_obj, evaluator_valid, dict_tune
    )
    mudf_tune = pd.DataFrame(mumat_tune)
    mudf_tune.index = dict_tune["patient_names"]
    mudf_tune.columns = [f"LV.{each}" for each in np.arange(mudf_tune.shape[1])]
    outdict = dict(
        zip(
            [
                "mudf_train",
                "perfdf_tune",
                "reconst_tune",
                "mudf_tune",
                "sd2mat_tune",
                "dict_perf_tune",
                "dict_train",
                "dict_tune",
                "model_train_obj",
                "train_dataloader",
                "tune_dataloader",
                "training_logdf",
                "tuning_logdf",
            ],
            [
                mudf_train,
                perfdf_tune,
                reconst_tune,
                mudf_tune,
                sd2mat_tune,
                dict_perf_tune,
                dict_train,
                dict_tune,
                model_train_obj,
                train_dataloader,
                tune_dataloader,
                training_dict_log,
                tuning_dict_log,
            ],
        )
    )
    return outdict


def get_logger_dictionary(batch_loss_name):
    """Get logger dictionary"""
    return collections.OrderedDict(
        [
            ("Epoch", []),
            ("Reconstruction.Loss", []),
            ("KLD", []),
            ("CE.Loss", []),
            (batch_loss_name, []),
            ("Accuracy", []),
            ("Regression.Reconstruction.Loss", []),
        ]
    )
