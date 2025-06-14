"""
Data loading, splitting, and scaling utilities for MLP/DeepSet models.

This module provides:
    - DataLoading: Loads and structures data from Parquet files for ML tasks.
    - SplitDataLoader: Splits a TensorDataset into training, validation, and evaluation sets.
    - DataScaler: Per-column scaling and optional log1p/expm1 transforms for tabular data.

Classes:
    DataStructure: TypedDict describing the structure of a single event.
    DataLoading: Loads features, truth, and pulse information from files.
    SplitDataLoader: Splits a dataset and provides convenient access to each split.
    DataScaler: Custom per-column scaler with optional log transforms.

Author: Kunz Bryan
"""
import torch as T
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


class DataStructure(TypedDict):
    """Structure of the data returned by a single event return by DataLoading."""
    features: T.Tensor
    mask: T.Tensor
    truth: T.Tensor
    number_of_pulses: T.Tensor


class DataLoading:
    """Class for loading data from Parquet files."""
    def __init__(
        self,
        feature_path: Path | str,
        truth_path: Path | str,
        truth_keys: list[str] | str = "initial_state_energy",
        maximum_length: int = 64) -> None:
        """Create a new instance of the DataLoading class.
        Args:
            feature_path: Path to the features parquet file.
            truth_path: Path to the truth parquet file.
            truth_keys: The keys of the truth DataFrame to get. If a single key is given, it will be returned as a single value.
            maximum_length: The maximum length of the features. The features will be padded to this length.
        """
        self._feature_path = feature_path
        self._truth_path = truth_path
        self._truth_keys = truth_keys
        self.maximum_length = maximum_length
        
        self._features = T.from_numpy(pd.read_parquet(self._feature_path).values).to(T.float32)

        truth_parquet = pd.read_parquet(self._truth_path)
        if isinstance(truth_keys, str):
            truth_type = truth_parquet[truth_keys]
        else:
            truth_type = truth_parquet[list(truth_keys)]

        truth_numpy = truth_type.to_numpy()
        if truth_numpy.ndim == 1:
            truth_numpy = truth_numpy.reshape(-1, 1)
        self._truth = T.from_numpy(truth_numpy).to(T.float32)

        self._cumulative_lengths = T.from_numpy(pd.read_parquet(self._truth_path)["cumulative_lengths"].values).to(T.int64)

        # number of pulses calculation
        self._number_of_pulses = T.tensor(
            np.diff(np.concatenate([np.zeros(1, dtype=np.int64), self._cumulative_lengths.numpy()])),
            dtype=T.float32)
    
    @property
    def number_of_pulses(self) -> T.Tensor:
        """Get the number of pulses of the dataset."""
        return self._number_of_pulses
    @property
    def features(self) -> T.Tensor:
        """Get the features of the dataset."""
        return self._features
    @property
    def truth(self) -> T.Tensor:
        """Get the truth of the dataset."""
        return self._truth
    @property
    def cumulative_lengths(self) -> T.Tensor:
        """Get the cumulative lengths of the dataset."""
        return self._cumulative_lengths

    def __getitem__(self, index: int) -> DataStructure:
        """Get the features and truth of a specific event.
        Args:
            index: The index of the event to get.
        Returns:
            A dictionary with the features, mask, truth and number of pulses of the event.
        """
        if index == 0:
            start = 0
        else:
            start = self.cumulative_lengths[index - 1]
        end = self.cumulative_lengths[index]

        event_feature = self.features[start:end]
        mask = T.zeros(self.maximum_length)
        mask[:len(event_feature)] = 1
        # Pad the features to the maximum length
        event_feature = T.nn.functional.pad(
            event_feature,
            (0, 0, 0, self.maximum_length - len(event_feature)),
            value=0,
        )
        return {"features": event_feature, 
                "mask": mask, 
                "truth": self.truth[index], 
                "number_of_pulses": self.number_of_pulses[index]}

    def __len__(self) -> int:
        """Get the number of events in the dataset."""
        return len(self.cumulative_lengths)

    def __str__(self) -> str:
        """Get the string representation of the dataset."""
        return (
            f"TridentDataset("
            f"\n  number of events: {len(self)},"
            f"\n  number of features: {self.features.shape[1]},"
            f"\n  Number of pulses: {self.cumulative_lengths[-1]},"
            f"\n  maximum length: {self.maximum_length},"
            f"\n  truth keys: {self._truth_keys},"
            f"\n)")

    def __repr__(self) -> str:
        """Get the string representation of the dataset."""
        return (
            f"TridentDataset("
            f"\n  feature_path={self._feature_path},"
            f"\n  truth_path={self._truth_path},"
            f"\n  truth_keys={self._truth_keys},"
            f"\n  maximum_length={self.maximum_length},"
            f"\n)")



@dataclass
class SplitDataLoader:
    """
    Split a TensorDataset into training, validation, and evaluation subsets.
    """
    training_set: object = None
    validation_set: object = None
    evaluation_set: object = None
    def __init__(self,
                dataset: T.utils.data.TensorDataset,
                validation_fraction: float = 0.2,
                evaluation_fraction: float = 0.2) -> None:
        """Create a new instance of the SplitDataLoader class.
        Args:
            dataset: TensorDataset that will be split.
            validation_fraction: Fraction of data use for validation dataset.
            evaluation_fraction: Fraction of data use for evaluation dataset.
        """
        indices = np.arange(len(dataset))

        if (validation_fraction + evaluation_fraction) >= 1.0:
            raise ValueError("Sum of validation_fraction and evaluation_fraction must be less than 1.")

        train_indices, remanant_indices = train_test_split(indices,
                                                           test_size=validation_fraction + evaluation_fraction,
                                                           shuffle=True)
        
        validation_size = validation_fraction / (validation_fraction + evaluation_fraction)
        valid_indices, eval_indices = train_test_split(remanant_indices,
                                                       test_size=1 - validation_size,
                                                       shuffle=True)

        # check if one split is empty
        if len(train_indices) == 0 or len(valid_indices) == 0 or len(eval_indices) == 0:
            raise ValueError("One of the splits is empty. Adjust your split fractions or dataset size.")

        self.training_set = T.utils.data.Subset(dataset, train_indices)
        self.validation_set = T.utils.data.Subset(dataset, valid_indices)
        self.evaluation_set = T.utils.data.Subset(dataset, eval_indices)

    @property
    def training_features(self) -> T.Tensor:
        """ Get training features of dataset."""
        if not isinstance(self.training_set, T.utils.data.Subset) or not isinstance(self.training_set.dataset, T.utils.data.TensorDataset):
            raise ValueError("Training_features set is not a valid TensorDataset Subset.")
        return self.training_set.dataset.tensors[0][self.training_set.indices]
    @property
    def validation_features(self) -> T.Tensor:
        """ Get validation features of dataset."""
        if not isinstance(self.validation_set, T.utils.data.Subset) or not isinstance(self.validation_set.dataset, T.utils.data.TensorDataset):
            raise ValueError("Validation_features set is not a valid TensorDataset Subset.")
        return self.validation_set.dataset.tensors[0][self.validation_set.indices]
    @property
    def evaluation_features(self) -> T.Tensor:
        """ Get evaluation features of dataset."""
        if not isinstance(self.evaluation_set, T.utils.data.Subset) or not isinstance(self.evaluation_set.dataset, T.utils.data.TensorDataset):
            raise ValueError("Evaluation_features set is not a valid TensorDataset Subset.")
        return self.evaluation_set.dataset.tensors[0][self.evaluation_set.indices]
    
    @property
    def training_number_of_pulses(self) -> T.Tensor:
        """ Get training number of pulses of dataset."""
        if not isinstance(self.training_set, T.utils.data.Subset) or not isinstance(self.training_set.dataset, T.utils.data.TensorDataset):
            raise ValueError("Training_number_of_pulses set is not a valid TensorDataset Subset.")
        return self.training_set.dataset.tensors[1][self.training_set.indices]
    @property
    def validation_number_of_pulses(self) -> T.Tensor:
        """ Get validation number of pulses of dataset."""
        if not isinstance(self.validation_set, T.utils.data.Subset) or not isinstance(self.validation_set.dataset, T.utils.data.TensorDataset):
            raise ValueError("Validation_number_of_pulses set is not a valid TensorDataset Subset.")
        return self.validation_set.dataset.tensors[1][self.validation_set.indices]
    @property
    def evaluation_number_of_pulses(self) -> T.Tensor:
        """ Get evaluation number of pulses of dataset."""
        if not isinstance(self.evaluation_set, T.utils.data.Subset) or not isinstance(self.evaluation_set.dataset, T.utils.data.TensorDataset):
            raise ValueError("Evaluation_number_of_pulses set is not a valid TensorDataset Subset.")
        return self.evaluation_set.dataset.tensors[1][self.evaluation_set.indices]
    
    @property
    def training_truth(self) -> T.Tensor:
        """ Get training truth of dataset."""
        if not isinstance(self.training_set, T.utils.data.Subset) or not isinstance(self.training_set.dataset, T.utils.data.TensorDataset):
            raise ValueError("Training_truth set is not a valid TensorDataset Subset.")
        return self.training_set.dataset.tensors[2][self.training_set.indices]
    @property
    def validation_truth(self) -> T.Tensor:
        """ Get validation truth of dataset."""
        if not isinstance(self.validation_set, T.utils.data.Subset) or not isinstance(self.validation_set.dataset, T.utils.data.TensorDataset):
            raise ValueError("Validation_truth set is not a valid TensorDataset Subset.")
        return self.validation_set.dataset.tensors[2][self.validation_set.indices]
    @property
    def evaluation_truth(self) -> T.Tensor:
        """ Get evaluation truth of dataset."""
        if not isinstance(self.evaluation_set, T.utils.data.Subset) or not isinstance(self.evaluation_set.dataset, T.utils.data.TensorDataset):
            raise ValueError("Evaluation_truth set is not a valid TensorDataset Subset.")
        return self.evaluation_set.dataset.tensors[2][self.evaluation_set.indices]



class DataScaler:
    """
    A custom scaler for tabular data that allows per-column scaling and optional log1p/expm1 transforms.
    """
    def __init__(self, method_list: list[str], log_transform_cols: list[int] | None) -> None:
        """Initialize the DataScaler.
        Args:
            method_list (list[str]): List of scaler types for each column.
            log_transform_cols (list[int] | None): Indices of columns to log-transform.
        """
        self.log_transform_cols = log_transform_cols or []
        self.scalers = []
        for method in method_list:
            if method == "standard":
                self.scalers.append(StandardScaler())
            elif method == "minmax":
                self.scalers.append(MinMaxScaler())
            elif method == "robust":
                self.scalers.append(RobustScaler())
            else:
                raise ValueError("Allowed methods: 'minmax', 'standard', 'robust'")

    def fit(self, data: np.ndarray) -> "DataScaler":
        """Fit the scalers to the data.
        Args:
            data (np.ndarray): 2D array of shape (n_samples, n_features).
        Returns:
            self: The fitted DataScaler instance.
        """
        data = data.copy()
        for indice in self.log_transform_cols:
            data[:, indice] = np.log1p(data[:, indice])
        for i, scaler in enumerate(self.scalers):
            scaler.fit(data[:, i].reshape(-1, 1))
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted scalers and log1p if specified.
        Args:
            data (np.ndarray): 2D array of shape (n_samples, n_features).
        Returns:
            np.ndarray: Transformed data of the same shape.
        """
        data = data.copy()
        for indice in self.log_transform_cols:
            data[:, indice] = np.log1p(data[:, indice])
        scaled = np.zeros_like(data)
        for i, scaler in enumerate(self.scalers):
            scaled[:, i] = scaler.transform(data[:, i].reshape(-1, 1)).flatten()
        return scaled

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit the scalers to the data, then transform it.
        Args:
            data (np.ndarray): 2D array of shape (n_samples, n_features).
        Returns:
            np.ndarray: Transformed data of the same shape.
        """
        data = data.copy()
        for indice in self.log_transform_cols:
            data[:, indice] = np.log1p(data[:, indice])
        scaled = np.zeros_like(data)
        for i, scaler in enumerate(self.scalers):
            scaled[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
        return scaled

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform the data using the fitted scalers and expm1 if specified.
        Args:
            data (np.ndarray): 2D array of shape (n_samples, n_features).
        Returns:
            np.ndarray: Inverse-transformed data of the same shape.
        """
        data = data.copy()
        inverse = np.zeros_like(data)
        for i, scaler in enumerate(self.scalers):
            inverse[:, i] = scaler.inverse_transform(data[:, i].reshape(-1, 1)).flatten()
        for indice in self.log_transform_cols:
            inverse[:, indice] = np.expm1(inverse[:, indice])
        return inverse