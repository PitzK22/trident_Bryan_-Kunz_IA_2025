"""
Training script for MLP/DeepSet models on TRIDENT data.

This script provides:
    - Configuration loading and validation for training runs.
    - Data loading, preprocessing, and scaling (with support for per-column log/scaler transforms).
    - Model construction (MLP or DeepSet) based on configuration.
    - Weighted loss computation and training with early stopping and checkpointing.
    - Saving of model weights and scalers.
    - Generation of training/validation loss plots and regression metric plots.

Author: Kunz Bryan
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import logging
import os
import shutil
import dacite
import yaml
import simple_parsing
import torch as T
import numpy as np
import joblib

from MLP_Data_Network import fit
from MLP_Data_Network.Prepare_data import DataLoading, SplitDataLoader, DataScaler
from MLP_Data_Network.Network import build_model
from MLP_Data_Network.Network.model import ConfigMLP


@dataclass
class TrainingCli:
    run_name: str

@dataclass
class TrainingConfig:
    features_name : str
    epoch_count: int
    weight_decay: float
    learning_rate: float
    validation_split: float
    evaluation_split: float
    batch_size: int
    early_stop: bool
    early_stopping_patience: int
    early_stopping_min_delta: float

    model: ConfigMLP



### configuration ###
# Configuration loading and modification
def load_config(path: Path) -> TrainingConfig:
    with open(path) as config_file:
        config_dict = yaml.safe_load(config_file)
        training_dict = config_dict["training"]
        evaluation_dict = config_dict.get("evaluation", {})
        # Merge evaluation_split into training_dict if present
        if "evaluation_split" in evaluation_dict:
            training_dict["evaluation_split"] = evaluation_dict["evaluation_split"]
        return dacite.from_dict(TrainingConfig, training_dict)

def prepare_config(
    output_path: Path, default_path: Path, run_name: str) -> TrainingConfig:
    os.makedirs(output_path.parent, exist_ok=True)
    print(f"copying {default_path} to {output_path}")
    shutil.copy(default_path, output_path)
    _ = input(
        f"please edit the config in outputs/{run_name}/config.yaml"
        " to set the parameters for this run\n"
        "afterwards, please press enter to continue...")
    return load_config(output_path)

def save_hyperparameters(path: Path, config: ConfigMLP):
    with open(path, "w") as hyperparameter_cache:
        yaml.dump(asdict(config, dict_factory=dict), hyperparameter_cache)


### fuctions for MAIN ###
def get_input_dim(input_tensor: T.Tensor) -> int:
    # For [num_events, 1] (num pulses): input_dim = 1
    # For [num_events, 64, 4] (features): input_dim = 4
    if input_tensor.numel() == 0:
        raise ValueError("Input tensor is empty.")
    if input_tensor.dim() == 2:
        return input_tensor.shape[1]
    elif input_tensor.dim() == 3:
        return input_tensor.shape[2]
    else:
        raise ValueError("Unsupported input tensor shape: {}".format(input_tensor.shape))


def weighted_loss(pred: T.Tensor, 
                  target: T.Tensor, 
                  weight: T.Tensor | None, 
                  loss_torch: T.nn.Module) -> T.Tensor:
    loss = loss_torch(pred, target)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def parameters_validity(config: TrainingConfig) -> None:
    if not isinstance(config.features_name, str):
        raise TypeError("features_name should be type sting")
    if not isinstance(config.batch_size, int):
        raise TypeError("batch_size should be an integer")
    if not isinstance(config.validation_split, float):
        raise TypeError("validation_split should be type float")
    if not isinstance(config.evaluation_split, float):
        raise TypeError("evaluation_split should be type float")
    if not isinstance(config.epoch_count, int):
        raise TypeError("epoch_count should be an integer")
    if not isinstance(config.learning_rate, float):
        raise TypeError("learning_rate should be type float")
    if not isinstance(config.weight_decay, float):
        raise TypeError("weight_decay should be type float")
    if not isinstance(config.early_stop, bool):
        raise TypeError("early_stop should be a boolean")
    if not isinstance(config.early_stopping_patience, int):
        raise TypeError("early_stopping_patience should be type float")
    if not isinstance(config.early_stopping_min_delta, float):
        raise TypeError("early_stopping_min_delta should be type float")

    if config.batch_size <= 0:
        raise ValueError("batch_size should be an integer number > 0")
    if not (0 < config.validation_split < 1):
        raise ValueError("validation_split should be between 0 and 1, better if < 0.5")
    if not (0 < config.evaluation_split < 1):
        raise ValueError("evaluation_split should be between 0 and 1, better if < 0.5")
    if config.epoch_count < 1:
        raise ValueError("epoch_count should be > 0")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate should be > 0")
    if config.weight_decay < 0:
        raise ValueError("weight_decay should be >= 0")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience should be > 0")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta should be >= 0")



##### MAIN #####
def main():
    # loading configuration
    cli = simple_parsing.parse(TrainingCli)
    config = prepare_config(Path(f"outputs/{cli.run_name}/config.yaml"),
                            Path("config_default.yaml"),
                            cli.run_name)
   
    parameters_validity(config=config)

    #initialization of functions
    dataloader_train = None
    dataloader_valid = None
    truth_scaler = None


    #### Data ####
    print("loading 'features.parquet' and 'truth.parquet' and create datasets")
    dataloader = DataLoading(  feature_path='data/features.parquet',
                                truth_path='data/truth.parquet',
                                truth_keys="initial_state_energy",
                                maximum_length=64)
    
    # building a per-event dataset
    all_features = []
    all_number_pulses = []
    all_truth = []
    for i in range(len(dataloader)):
        sample = dataloader[i]
        all_features.append(sample["features"].unsqueeze(0))              # [1, 64, 4]
        all_number_pulses.append(sample["number_of_pulses"].unsqueeze(0)) # [1]
        all_truth.append(sample["truth"].unsqueeze(0))                    # [1] or [1, 1]

    features = T.cat(all_features, dim=0)               # [num_events, 64, 4]
    number_of_pulses = T.cat(all_number_pulses, dim=0)  # [num_events, 1]
    truth = T.cat(all_truth, dim=0)                     # [num_events] or [num_events, 1]
    
    dataset = T.utils.data.TensorDataset(features, number_of_pulses, truth)

    split_dataset = SplitDataLoader(dataset=dataset,
                                    validation_fraction=config.validation_split,
                                    evaluation_fraction=config.evaluation_split)
    
    truth_training = split_dataset.training_truth
    truth_validation = split_dataset.validation_truth

    print("Preprocess datasets")
    truth_training_np = truth_training.numpy().reshape(-1, 1)
    truth_validation_np = truth_validation.numpy().reshape(-1, 1)

    truth_scaler = DataScaler(method_list=["robust"], log_transform_cols=[0])
    truth_training_scaled = T.tensor(truth_scaler.fit_transform(truth_training_np), dtype=T.float32)
    truth_validation_scaled = T.tensor(truth_scaler.transform(truth_validation_np), dtype=T.float32)
    # save scaling for truth
    joblib.dump(truth_scaler, f"outputs/{cli.run_name}/truth_scaler.pkl")


    # selection of inputs data
    if config.features_name == "n_pulses":
        deepset_on = False
        n_pulses_training = split_dataset.training_number_of_pulses
        n_pulses_validation = split_dataset.validation_number_of_pulses

        n_pulses_training_np = np.log1p(n_pulses_training.numpy().reshape(-1, 1))
        n_pulses_validation_np = np.log1p(n_pulses_validation.numpy().reshape(-1, 1))

        n_pulses_scaler = DataScaler(method_list=["robust"], log_transform_cols=[0])
        features_training_scaled = T.tensor(n_pulses_scaler.fit_transform(n_pulses_training_np), dtype=T.float32)
        features_validation_scaled = T.tensor(n_pulses_scaler.transform(n_pulses_validation_np), dtype=T.float32)
        # save scaling for number of pulses
        joblib.dump(n_pulses_scaler, f"outputs/{cli.run_name}/n_pulses_scaler.pkl")

    elif config.features_name == "features":
        deepset_on = True
        features_training = split_dataset.training_features
        features_validation = split_dataset.validation_features

        features_training_np = features_training.numpy().reshape(-1, features_training.shape[-1])
        features_validation_np = features_validation.numpy().reshape(-1, features_validation.shape[-1])

        # features scaling: x_pos = standard, y_pos = standard, z_pos = minmax, t = log + robust
        features_scaler = DataScaler(method_list=["standard", "standard", "standard", "standard"],
                                        log_transform_cols= [3])
        
        features_training_scaled = T.tensor(features_scaler.fit_transform(features_training_np), dtype=T.float32).reshape(features_training.shape)
        features_validation_scaled = T.tensor(features_scaler.transform(features_validation_np), dtype=T.float32).reshape(features_validation.shape)
        # save scaling for features
        joblib.dump(features_scaler, f"outputs/{cli.run_name}/features_scaler.pkl")

    else:
        raise ValueError(
            "Invalid value for 'features_name' in 'config_default.yaml'. "
            "Allowed values are: 'n_pulses', 'features'."
            )
    

    # weighted loss function and calculation of weight
    truth_np = truth_training_scaled.detach().numpy().flatten()
    unique, _, _, counts = np.unique(truth_np, return_index=True, return_inverse=True, return_counts=True)
    count_dict = dict(zip(unique, counts))
    weights_np = np.array([1.0 / count_dict[v] for v in truth_np])
    weights = T.tensor(weights_np, dtype=T.float32)
    
    loss_torch= T.nn.SmoothL1Loss()
    loss_function = lambda pred, target, weights: weighted_loss(pred, target, weights, loss_torch)


    # Use weighted loss function for training batches
    train_loader = T.utils.data.TensorDataset(features_training_scaled, truth_training_scaled, weights)
    dataloader_train = T.utils.data.DataLoader(train_loader, batch_size=config.batch_size, shuffle=True)

    valid_loader = T.utils.data.TensorDataset(features_validation_scaled, truth_validation_scaled)
    dataloader_valid = T.utils.data.DataLoader(dataset=valid_loader, batch_size=config.batch_size, shuffle=False)


    #### Fit
    # model, optimizer and loss function
    # use gpu if possible, if not, use cpu
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    print("training the model")


    input_dim = get_input_dim(features_training_scaled)
    model = build_model(config=config.model,
                        deepset_on=deepset_on,
                        input_dim=input_dim,
                        output_dim=1,
                        hidden_activation=T.nn.ReLU(),
                        output_activation=T.nn.Identity())
    
    model = model.to(device=device)

    optimizer = T.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


    if dataloader_train is None:
        raise ValueError("dataloader_train must not be None before calling fit()")
    if dataloader_valid is None:
        raise ValueError("dataloader_valid must not be None before calling fit()")
    
    ## fit the model
    summary = fit(model             = model,
                  optimizer         = optimizer,
                  loss_function     = loss_function,
                  training_loader   = dataloader_train,
                  validation_loader = dataloader_valid,
                  epochs_max_count  = config.epoch_count,
                  patience          = config.early_stopping_patience,
                  min_delta         = config.early_stopping_min_delta,
                  checkpoint_path   = str(Path(f"outputs/{cli.run_name}/CheckPoint_model_parameters.pth")),
                  earlyStop         = config.early_stop
                  )

    
    if not config.early_stop:
        print(f"saving network parameters in outputs/{cli.run_name}")
        os.makedirs(f"outputs/{cli.run_name}", exist_ok=True)
        T.save(model.state_dict(), f"outputs/{cli.run_name}/postTraining_model_parameters.pth")

    # plots
    os.makedirs(f"outputs/{cli.run_name}/training_plots", exist_ok=True)

    summary.save_loss_plot(Path(f"outputs/{cli.run_name}/training_plots/loss_plot.png"))
    summary.save_pred_vs_truth_plot(model, 
                                    dataloader_valid, 
                                    truth_scaler, 
                                    Path(f"outputs/{cli.run_name}/training_plots/pred_truth_plot.png"), 
                                    desc="Validation plot")
    summary.save_regression_metrics_plot(model, 
                                         dataloader_valid, 
                                         truth_scaler, 
                                         Path(f"outputs/{cli.run_name}/training_plots/metrics_plot.png"))
    print(f"Training plots saved at 'outputs/{cli.run_name}/training_plots'")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() # protect multiprocessing under Windows 
    logging.basicConfig()
    main()