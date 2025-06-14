"""
Evaluation script for trained MLP/DeepSet models.

This script provides:
    - Loading of trained model and scalers.
    - Loading and preprocessing of evaluation data.
    - Consistent data splitting and scaling as in training.
    - Model evaluation and prediction on the evaluation set.
    - Generation and saving of prediction vs. truth plots.

Author: Kunz Bryan
"""
from pathlib import Path
import yaml
import joblib
import simple_parsing
import torch as T
from dataclasses import dataclass
from pydantic import BaseModel

from MLP_Data_Network.Prepare_data import DataLoading, SplitDataLoader, DataScaler
from MLP_Data_Network.Network import build_model, modelConfig
from MLP_Data_Network import FitSummary

@dataclass
class EvaluationCli:
    run_name: str

@dataclass
class EvaluationConfig(BaseModel):
    batch_size: int
    evaluation_split: float

def load_full_config(path: Path) -> dict:
    with open(path) as config_file:
        return yaml.safe_load(config_file)

def load_hyperparameters(path: Path) -> modelConfig:
    with open(path) as config_file:
        config = yaml.safe_load(config_file)
        model_params = config["training"]["model"]
        return modelConfig(**model_params)
    

def get_input_dim(input_tensor: T.Tensor) -> int:
    # For [num_events, 1] (num pulses): input_dim = 1
    # For [num_events, 64, 4] (features): input_dim = 4
    if input_tensor.dim() == 2:
        return input_tensor.shape[1]
    elif input_tensor.dim() == 3:
        return input_tensor.shape[2]
    else:
        raise ValueError("Unsupported input tensor shape: {}".format(input_tensor.shape))



### main ###
def main():
    cli = simple_parsing.parse(EvaluationCli)
    config = load_full_config(Path(f"outputs/{cli.run_name}/config.yaml"))

    print("loading 'features.parquet' and 'truth.parquet' and create dataset")
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

    print("Preprocess dataset")
    split_loader = SplitDataLoader(dataset=dataset,
                                   validation_fraction=config["training"]["validation_split"],
                                   evaluation_fraction=config["evaluation"]["evaluation_split"])
    
    
    # load the scaler of truth
    truth_scaler = joblib.load(f"outputs/{cli.run_name}/truth_scaler.pkl")
    truth_eval = split_loader.evaluation_truth
    truth_eval_np = truth_eval.numpy().reshape(-1, 1)
    truth_eval_scaled = truth_scaler.transform(truth_eval_np)
    truth_eval_scaled = T.tensor(truth_eval_scaled, dtype=T.float32)


    if config["training"]["features_name"] == "n_pulses":
        deepset_on = False
        # load the scaler of number of pulses
        n_pulses_scaler = joblib.load(f"outputs/{cli.run_name}/n_pulses_scaler.pkl")
        n_pulses_eval = split_loader.evaluation_number_of_pulses
        n_pulses_eval_np = n_pulses_eval.numpy().reshape(-1, 1)
        features_eval_scaled = n_pulses_scaler.transform(n_pulses_eval_np)
        features_eval_scaled = T.tensor(features_eval_scaled, dtype=T.float32)

    elif config["training"]["features_name"] == "features":
        deepset_on = True
        # load the scaler of features
        features_scaler = joblib.load(f"outputs/{cli.run_name}/features_scaler.pkl")
        features_eval = split_loader.evaluation_features
        original_shape = features_eval.shape
        features_eval_np = features_eval.numpy().reshape(-1, features_eval.shape[-1])
        features_eval_scaled = features_scaler.transform(features_eval_np)
        features_eval_scaled = T.tensor(features_eval_scaled, dtype=T.float32).reshape(original_shape)

    else:
        raise ValueError(
        "Invalid value for 'features_name' in 'config_default.yaml'. "
        "Allowed values are: 'n_pulses', 'features'.")


    if features_eval_scaled is None or truth_eval_scaled is None:
        raise ValueError("features and truth must not be None when creating the dataset.")

    eval_set = T.utils.data.TensorDataset(features_eval_scaled, truth_eval_scaled)
    eval_loader = T.utils.data.DataLoader(eval_set, batch_size=config["evaluation"]["batch_size"], shuffle=False)
    
    
    # Loading and building model
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    model_config = load_hyperparameters(Path(f"outputs/{cli.run_name}/config.yaml"))
    
    input_dim = get_input_dim(features_eval_scaled)
    model = build_model(
        config=model_config,
        deepset_on=deepset_on,
        input_dim=input_dim,
        output_dim=1,
        hidden_activation=T.nn.ReLU(),
        output_activation=T.nn.Identity()
    )
    model = model.to(device)
    

    if config["training"]["early_stop"]:
        model.load_state_dict(T.load(f"outputs/{cli.run_name}/CheckPoint_model_parameters.pth"))
    else:
        model.load_state_dict(T.load(f"outputs/{cli.run_name}/postTraining_model_parameters.pth"))
    model.eval()

    # plot
    summary = FitSummary()
    summary.save_pred_vs_truth_plot(model,
                                    eval_loader,
                                    truth_scaler=truth_scaler,
                                    path=Path(f"outputs/{cli.run_name}/evaluation_plot/eval_pred_truth_plot.png"),
                                    desc="Evaluation plot")
    print(f"Evaluation plot saved at 'outputs/{cli.run_name}/evaluation_plot'.")


if __name__ == "__main__":
    main()