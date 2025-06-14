"""
MLP and DeepSet model definitions for tabular and set-based data.

Classes:
    ConfigMLP: Dataclass for MLP hyperparameters.
    MLP: Standard Multi-Layer Perceptron with optional depth, dropout, and batch normalization.
    DeepSet: DeepSet architecture for permutation-invariant set processing.

Author: Kunz Bryan
"""
from dataclasses import dataclass
import torch as T

@dataclass
class ConfigMLP():
    """Configuration dataclass for MLP hyperparameters."""
    hidden_dim : int
    depth : int
    dropout: float
    

class MLP(T.nn.Module):
    """Multi-Layer Perceptron (MLP) with configurable depth, dropout, and batch normalization."""
    def __init__(self,
                 input_dim: int=1,
                 hidden_dim: int=100,
                 out_dim: int=1,
                 hidden_act: T.nn.Module=T.nn.ReLU(),
                 out_act: T.nn.Module=T.nn.Identity(),
                 depth: int=0,
                 dropout: float=0.0) -> None:
        """Initialize the MLP model.
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of units in hidden layers.
            out_dim (int): Number of output units.
            hidden_act (torch.nn.Module): Activation function for hidden layers.
            out_act (torch.nn.Module): Activation function for output layer.
            depth (int): Number of additional hidden layers (after the first).
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.p = dropout
        self.layer_in = T.nn.Linear(input_dim, hidden_dim)
        self.batch_norm = T.nn.BatchNorm1d(hidden_dim)
        self.layer_out = T.nn.Linear(hidden_dim, out_dim)

        self.hidden_activation = hidden_act
        self.out_activation = out_act

        # tunable depth : number of sequence of layers
        self.depth = depth
        if self.depth != 0:
            self.hidden_layers = T.nn.ModuleList([
                T.nn.Linear(hidden_dim, hidden_dim) for _ in range(self.depth)])
            self.batch_norm_layers = T.nn.ModuleList([
                T.nn.BatchNorm1d(hidden_dim) for _ in range(self.depth)])
            self.dropouts = T.nn.ModuleList([
                T.nn.Dropout(p=self.p) for _ in range(self.depth)])
        else:
            self.hidden_layers = T.nn.ModuleList([])
            self.batch_norm_layers = T.nn.ModuleList([])
            self.dropouts = T.nn.ModuleList([])

    def forward(self, inputs: T.Tensor) -> T.Tensor:
        """Forward pass of the MLP.
        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, input_dim] or [batch_size, set_size, input_dim].
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim] or [batch_size, set_size, output_dim].
        """
        hidden = self.layer_in(inputs)
        # Handle features inputs
        if hidden.dim() == 3:
            batch, set_size, hid_dim = hidden.shape
            hidden = hidden.view(-1, hid_dim)
            hidden = self.batch_norm(hidden)
            hidden = hidden.view(batch, set_size, hid_dim)
        else:
            hidden = self.batch_norm(hidden)
        hidden = self.hidden_activation(hidden)

        if self.depth != 0:
            for layer, bn, dropout in zip(self.hidden_layers, self.batch_norm_layers, self.dropouts):
                if hidden.dim() == 3:
                    batch, set_size, hid_dim = hidden.shape
                    hidden = hidden.view(-1, hid_dim)
                    hidden = layer(hidden)
                    hidden = bn(hidden)
                    hidden = self.hidden_activation(hidden)
                    hidden = dropout(hidden)
                    hidden = hidden.view(batch, set_size, hid_dim)
                else:
                    hidden = layer(hidden)
                    hidden = bn(hidden)
                    hidden = self.hidden_activation(hidden)
                    hidden = dropout(hidden)

        return self.out_activation(self.layer_out(hidden))
    

class DeepSet(T.nn.Module):
    """DeepSet architecture for permutation-invariant set processing."""
    def __init__(self, 
                 mlp_in: T.nn.Module, 
                 mlp_mid: T.nn.Module, 
                 mlp_out: T.nn.Module) -> None:
        """Initialize the DeepSet model.
        Args:
            mlp_in (torch.nn.Module): MLP for embedding input features.
            mlp_mid (torch.nn.Module): MLP for processing pooled set representation.
            mlp_out (torch.nn.Module): MLP for producing final output.
        """
        super().__init__()
        self.mlp_in = mlp_in
        self.mlp_mid = mlp_mid
        self.mlp_out = mlp_out

    def forward(self, features: T.Tensor) -> T.Tensor:
        """Forward pass of the DeepSet.
        Args:
            features (torch.Tensor): Input tensor of shape [batch_size, set_size, input_dim].
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        features_embeded = self.mlp_in(features)
        features_embeded = features_embeded.sum(dim=1)
        features_embeded = self.mlp_mid(features_embeded)
        return self.mlp_out(features_embeded)