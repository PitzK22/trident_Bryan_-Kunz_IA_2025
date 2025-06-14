"""
Utilities and summary/plotting tools for MLP/DeepSet models.

This module provides:
    - Early stopping and checkpointing classes for model training.
    - FitSummary class for tracking and plotting training/validation loss and predictions.
    - Functions for computing epoch loss and fitting models with early stopping.
    - Plotting utilities for regression metrics and prediction vs. truth.

Classes:
    EarlyStopper: Implements early stopping logic based on validation loss.
    CheckPoint: Saves the best model during training.
    FitSummary: Tracks losses and provides plotting utilities.

Functions:
    epoch_loss: Compute average loss over a DataLoader epoch.
    fit: Train a model with optional early stopping and checkpointing.

Author: Kunz Bryan
"""
import torch as T
import numpy as np
from collections.abc import Callable
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from MLP_Data_Network.Prepare_data import DataScaler



class EarlyStopper():
    """Implements early stopping logic based on validation loss.
    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change to qualify as an improvement.
    """
    def __init__(self, patience: int=1, min_delta: float=0.) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool: 
        if ( (validation_loss + self.min_delta) < self.min_validation_loss ):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class CheckPoint():
    """Saves the best model during training based on validation loss.
    Args:
        model_path (str): Path to save the best model weights.
    """
    def __init__(self, model_path: str='best_model.pth') -> None:
        self.best_model = None
        self.best_loss = float('inf')
        self.model_path = model_path

    def check(self, model: T.nn.Module, loss: float) -> None:
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = model
            print(f"Saving the best model with validation loss {loss:.4f} to {self.model_path}")
            T.save(model.state_dict(), self.model_path)



class FitSummary:
    """Tracks training and validation losses, and provides plotting utilities.
    Attributes:
        training_losses (list[float]): List of training losses per epoch.
        validation_losses (list[float]): List of validation losses per epoch.
        epoch_index (int): Current epoch index.
    """
    epoch_index: int
    training_losses: list[float]
    validation_losses: list[float]

    def __init__(self) -> None:
        self.training_losses = []
        self.validation_losses = []
        self.epoch_index = 0

    def append_summary(self, training_loss: float, validation_loss: float) -> tuple[list[float], list[float], int]:
        """Append training and validation loss for the current epoch.
        Args:
            training_loss (float): Training loss for the epoch.
            validation_loss (float): Validation loss for the epoch.
        Returns:
            tuple: (training_losses, validation_losses, epoch_index)
        """
        if self.epoch_index % 5 == 0:
            print(f"epoch {self.epoch_index}, training loss {training_loss:.4f}, validation loss {validation_loss:.4f}")

        self.training_losses.append(training_loss)
        self.validation_losses.append(validation_loss)
        self.epoch_index += 1

        return self.training_losses, self.validation_losses, self.epoch_index


    def save_loss_plot(self, path: Path) -> None:
        """Save a plot of training and validation loss over epochs.
        Args:
            path (Path): Path to save the plot.
        """
        figure, axes_loss = plt.subplots()
        epoch_numbers = list(range(self.epoch_index))

        axes_loss.plot(epoch_numbers, self.training_losses, label="training loss", color="C0")
        axes_loss.plot(epoch_numbers, self.validation_losses, label="validation loss", color="C1")
        axes_loss.set_xlabel("epoch")
        axes_loss.set_ylabel("loss")
        axes_loss.legend()

        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")
    

    def save_pred_vs_truth_plot(self, 
                                model: T.nn.Module, 
                                dataloader: T.utils.data.DataLoader, 
                                truth_scaler: DataScaler | None, 
                                path: Path, 
                                desc: str = "Evaluating"
                                ) -> None:
        """Save a scatter plot of predictions vs. truth for a given model and dataloader.
        Args:
            model (torch.nn.Module): Trained model.
            dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
            truth_scaler (DataScaler or None): Scaler for inverse-transforming predictions and truth.
            path (Path): Path to save the plot.
            desc (str): Description for tqdm progress bar.
        """
        model.eval()
        all_preds = []
        all_truths = []
        device = next(model.parameters()).device
        # T.no_grad() to prevent unnecessary computation and memory usage
        with T.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                features, truths = batch[:2]
                features = features.to(device)
                preds = model(features)
                all_preds.append(preds.cpu())
                all_truths.append(truths.cpu())

        all_preds = T.cat(all_preds).squeeze().numpy().flatten()
        all_truths = T.cat(all_truths).squeeze().numpy().flatten()

        if truth_scaler is not None:
            all_preds = truth_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
            all_truths = truth_scaler.inverse_transform(all_truths.reshape(-1, 1)).flatten()

        relative_error = np.abs(all_preds - all_truths) / (np.abs(all_truths))
        percent_within_5 = np.mean(relative_error < 0.05) * 100

        plt.figure(figsize=[6, 6])
        plt.scatter(all_truths, all_preds, alpha=0.3, s=5)
        plt.xlabel("Truth")
        plt.ylabel("Prediction")
        plt.title(f"Prediction vs Truth\n{percent_within_5:.1f}% of predicted values are close to truth values within 5% accuracy")
        plt.plot([all_truths.min(), all_truths.max()], [all_truths.min(), all_truths.max()], 'r--', label="perfect prediction")
        plt.grid(True)
        plt.legend()

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


    def save_regression_metrics_plot(self, 
                                     model: T.nn.Module, 
                                     dataloader: T.utils.data.DataLoader, 
                                     truth_scaler: DataScaler | None, 
                                     path: Path) -> None:
        """Save a bar plot of regression metrics (MAE, MSE, RMSE, R²) for model predictions.
        Args:
            model (torch.nn.Module): Trained model.
            dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
            truth_scaler (DataScaler or None): Scaler for inverse-transforming predictions and truth.
            path (Path): Path to save the plot.
        """
        model.eval()
        all_preds = []
        all_truths = []
        device = next(model.parameters()).device
        with T.no_grad():
            for batch in dataloader:
                features, truths = batch[:2]
                features = features.to(device)
                preds = model(features)
                all_preds.append(preds.cpu())
                all_truths.append(truths.cpu())

        all_preds = T.cat(all_preds).squeeze().numpy().flatten()
        all_truths = T.cat(all_truths).squeeze().numpy().flatten()

        if truth_scaler is not None:
            all_preds = truth_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
            all_truths = truth_scaler.inverse_transform(all_truths.reshape(-1, 1)).flatten()

        np_truths = np.array(all_truths)
        np_preds = np.array(all_preds)

        # Compute metrics
        mae = np.mean(np.abs(np_truths - np_preds))
        mse = np.mean((np_truths - np_preds) ** 2)
        rmse = np.sqrt(mse)

        ss_res = np.sum((np_truths - np_preds) ** 2)
        ss_tot = np.sum((np_truths - np.mean(np_truths)) ** 2)
        r2 = 1 - ss_res / ss_tot

        metrics = [mae, mse, rmse, r2]
        metric_names = ['Mean absolute error', 'Mean squared error', 'Root mean squared error', 'R²']

        plt.figure(figsize=(6,4))
        bars = plt.bar(metric_names, metrics, color=['C0', 'C1', 'C2', 'C3'])
        plt.title("Regression Metrics (Validation Set)")
        plt.ylabel("Score (log)")
        plt.yscale('log')
        plt.xticks(rotation=90)

        for bar, value in zip(bars, metrics):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.3f}", ha='center', va='bottom')

        plt.grid(axis='y', linestyle='--', alpha=0.7)

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()



#### functions for fit ####
def epoch_loss(loader: T.utils.data.DataLoader,
               model: T.nn.Module,
               loss_function: Callable[[T.Tensor, T.Tensor, Optional[T.Tensor]], T.Tensor], 
               optimizer = None, 
               is_training = True):
    """Compute the average loss over a DataLoader epoch.
    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the epoch.
        model (torch.nn.Module): Model to evaluate.
        loss_function (callable): Loss function.
        optimizer (torch.optim.Optimizer or None): Optimizer for training.
        is_training (bool): Whether to perform training (backprop) or just evaluation.
    Returns:
        float: Average loss over the epoch.
    """
    device = next(model.parameters()).device
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training" if is_training else "Validation"):
        features = batch[0].to(device)
        truth = batch[1].to(device)
        # Handle optional weights
        weight = batch[2].to(device) if len(batch) > 2 else None

        if is_training:
            if optimizer is None:
                raise ValueError("Optimizer must not be None during training.")
            optimizer.zero_grad()
            output = model(features)
            loss = loss_function(output, truth, weight)
            loss.backward()
            optimizer.step()
        else:
            with T.no_grad():
                output = model(features)
                loss = loss_function(output, truth, None)
        total_loss += loss.item()
    return total_loss / len(loader)


#### fit  ####
def fit(
        model: T.nn.Module,
        optimizer: T.optim.Optimizer,
        loss_function: Callable[[T.Tensor, T.Tensor, Optional[T.Tensor]], T.Tensor],
        training_loader: T.utils.data.DataLoader,
        validation_loader: T.utils.data.DataLoader,
        epochs_max_count: int,
        patience: int,
        min_delta: float,
        checkpoint_path: str,
        earlyStop: bool
        ) -> FitSummary:
    """Train a model with optional early stopping and checkpointing.
    Args:
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_function (callable): Loss function.
        training_loader (torch.utils.data.DataLoader): DataLoader for training data.
        validation_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        epochs_max_count (int): Maximum number of epochs.
        patience (int): Patience for early stopping.
        min_delta (float): Minimum delta for early stopping.
        checkpoint_path (str): Path to save the best model.
        earlyStop (bool): Whether to use early stopping.
    Returns:
        FitSummary: Summary object with training/validation losses.
    """
    if earlyStop == True:
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    else:
        early_stopper = None
        
    check_pointer = CheckPoint(model_path=checkpoint_path)
    summary = FitSummary()

    for epoch in tqdm(range(epochs_max_count), desc="Epochs"):
        # Training
        model.train()
        epoch_loss_train = epoch_loss(training_loader, model, loss_function, optimizer, is_training=True)

        # Validation
        model.eval()
        epoch_loss_valid = epoch_loss(validation_loader, model, loss_function, optimizer=None, is_training=False)

        summary.append_summary(epoch_loss_train, epoch_loss_valid)

        check_pointer.check(model, epoch_loss_valid)

        if (early_stopper is not None) and (early_stopper.early_stop(epoch_loss_valid)):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model weights after training
    if check_pointer.best_model is not None:
        model.load_state_dict(T.load(check_pointer.model_path))

    return summary