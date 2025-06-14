# TRIDENT: Tropical Deepsea Neutrino Telescope Machine Learning Project

This repository provides a flexible and scalable template for regression on physics data use for Trident project, using PyTorch with DeepSet and MLP architectures. The structure and workflow are designed to respect and do a bit more than the standards expected in the Physics Applications of AI (2025) course given by the professor GOLLING Tobias. This project is a student project, chosen from three others, which is use to give a final grade.
---

## Overview

This project includes:
- A Python module: `MLP_Data_Network` folder and `pyproject.toml` for install details
- A `data` folder, which contains your features and truth data in Parquet format
- A `config_default.yaml` file, which holds default hyperparameters and configuration
- Three main scripts: `Training.py`, `Evaluation.py`, and `hyperpar_search.py` for training, evaluation, and hyperparameter optimization
- This `README.md` for documentation and guidance

Before proceeding, **install the Python module** in editable mode:
```
pip install -e .
```
This will also install all required dependencies.
---

## Project Structure

```
MLP_Data_Network/
├── __init__.py
├── Prepare_data.py
└── Network/
    ├── __init__.py
    └── model.py

data/
├── features.parquet
└── truth.parquet

outputs/
    └── ...

config_default.yaml
Training.py
Evaluation.py
hyperpar_search.py
README.md
pyproject.toml
```
---

## Training and Evaluation Scripts

The training and evaluation scripts can be run with the `python` command. Both accept command line arguments; use the `--help` flag to see available options:
```bash
python Training.py --help
```
It will show you the only option avialable is using '--run_name', which have to be follow by the name of the experiment.

### Training

To start a new training run:
```bash
python Training.py --run_name my_experiment
```
- This copies `config_default.yaml` into `outputs/my_experiment/config.yaml`.
- Edit the config in the run folder to set parameters for this run (e.g., batch size, learning rate, hidden_layer, ...).
- Press Enter to continue with training.
- The script creates subfolders for model weights and plots in `outputs/my_experiment/`.

### Evaluation

To evaluate a trained model:
```bash
python Evaluation.py --run_name my_experiment
```
- This loads the trained model from the run folder and evaluates on the evaluation set.
- Plot is save in `outputs/my_experiment/`.


## Python Module

The main code is organized as a Python module (`MLP_Data_Network`). This allows you to import classes and functions in your scripts, for example:
```python
from MLP_Data_Network.Prepare_data import DataLoading, SplitDataLoader, DataScaler
from MLP_Data_Network.Network import build_model
```
This modular structure makes your code reusable and easy to maintain.
---

## Data

Place your input data in the `data/` folder as Parquet files:
- `features.parquet`: Input features for each event
- `truth.parquet`: Target values (regression outputs) and length of an event (number of features used for an event)

You can load and preprocess data using the provided utilities in `MLP_Data_Network.Prepare_data`.
---

## Configuration

All experiment parameters (training, evaluation, data splits,, input data etc.) are set in a YAML config file.  
- The default config is `config_default.yaml`.
- For each run, a copy is made in the corresponding output folder, which you can edit before training.
---

## Dataclasses

This project makes extensive use of Python dataclasses for configuration and data handling. For example:
```python
from dataclasses import dataclass

@dataclass
class EventData:
    features: np.ndarray
    truth: float
    number_of_pulses: int
```
Dataclasses simplify initialization and making the code more maintainable and readable.
---

## Config and CLI Parsing

We use `simple-parsing` and `dacite` for configuration and command-line parsing.  
- CLI options are defined using dataclasses and parsed automatically.
- Config files are loaded from YAML and mapped to dataclasses for type safety and clarity.
---

## Results

- All results are saved in the `outputs/` folder, organized by run name.
- Each run folder contains the config, model weights, and plots.
---