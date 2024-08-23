# Orion code base

This package provides an interface to all of the Orion capabilities discussed in the manuscript entitled [Deep generative AI models analyzing circulating orphan non-coding RNAs enable accurate detection of early-stage non-small cell lung cancer
](https://doi.org/10.1101/2024.04.09.24304531)


## Requirements

This package and notebooks were run on Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz machine with Ubuntu 22.04.4 LTS and the following python v3.10.12 packages.
For installation of scvi-tools on M1-chip Mac machines, please visit: [scvi-tools](https://docs.scvi-tools.org/en/stable/installation.html)

```
Ubuntu                         22.04.4 LTS
python                         3.10.12
absl-py                        1.0.0
numpy                          1.26.4
numpyro                        0.15.0
pandas                         1.5.3
pytorch-lightning              2.2.5
scipy                          1.10.0
scvi-tools                     1.1.2
seaborn                        0.12.2
shap                           0.43.0
torch                          2.0.1
torchmetrics                   1.4.0.post0
tqdm                           4.64.1
```

Instructions for installation with conda (runtime: 5 minutes):

```bash
conda create -n orion python=3.10.12
conda activate orion
conda install pytorch=2.0.1 numpy=1.26.4 pandas=1.5.3
pip install scvi-tools==1.1.2
```

## Quick start

Please see the following [notebook](notebooks/orion-with-simulated-data.ipynb) which shows a basic usage of Orion with simulated datasets (runtime: 4 minutes).

Essentially, Orion requires a dictionary of the data with the following keys:

```
"oncrna_ar": A numpy array of [samples, oncRNA] features count data
"oncrna_names": Name of `oncrna_ar` columns
"patient_names": Names of `oncrna_ar` rows
"onehot_ar": One-hot encoded class labels to train/predict
"smrnamat": A numpy array of [samples, small RNAs] features counts data for learning RNA content
"smrna_names": Name of `smarnamat` columns
"batch_list": A list of integers indicating batches to consider for triplet margin loss anchors
```

In addition, Orion requires a dictionary of model hyperparameters with some most relevant ones shown below:
```
"n_input": Number of oncRNAs
"n_input_lib": Number of small RNAs
"dp": Dropout
"loss_scalers": A list of scalers corresponding to NLL, KLD_Z, CE, and TML losses
"lr": Learning rate
"n_hidden": Number of hidden units
"num_lvs": Number of latent variables
"n_layers": Number of layers
"num_epochs": Number of epochs
"mini_batch": Number of samples in each mini batch
"tm_rounds": Number of rounds to sample anchors and compute TMLoss per mini batch
"num_classes": Number of classes
"weight_sample_loss": Weight to assign for each sample for classification task
"use_generative_sampling": Whether to use generative sampling for training the classifier
```

With these two dictionaries, Orion can be trained as:

```python
trained_model_dict = train_orion_model(
    data_dictionary,
    train_idx, # Array of sample indices to be used for training
    tune_idx, # Array of sample indices to be used for reporting metrics
    select_features, # a subset or all of features in `onc_mat` and `oncrna_names` keys of data dictionary
    dict_params=dict_params,
)
```

## Datasets

The datasets required for running the [orion-generate-predictions.ipynb](notebooks/orion-generate-predictions.ipynb) are available on [Zenodo](https://doi.org/10.5281/zenodo.12809652).


## Maintenance and support

Please use the GitHub issues for requesting assistance with the Orion package.