"""
Network module initialization and model builder.

This module provides:
    - Imports for MLP, ConfigMLP, and DeepSet architectures.
    - The build_model() function to construct either a standard MLP or a DeepSet model
      based on configuration and input arguments.

Functions:
    build_model: Build and return an MLP or DeepSet model according to the configuration.

Author: Kunz Bryan
"""
from torch.nn import Module, Identity
from MLP_Data_Network.Network.model import MLP, ConfigMLP, DeepSet

modelConfig = ConfigMLP
def build_model(config: modelConfig, 
                deepset_on: bool , 
                input_dim: int, 
                output_dim: int, 
                hidden_activation: Module, 
                output_activation: Module) -> Module:
    """Build and return an MLP or DeepSet model according to the configuration.
    Args:
        config (ConfigMLP): Model hyperparameters.
        deepset_on (bool): If True, build a DeepSet model; otherwise, build a standard MLP.
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        hidden_activation (torch.nn.Module): Activation function for hidden layers.
        output_activation (torch.nn.Module): Activation function for output layer.
    Returns:
        torch.nn.Module: The constructed model (MLP or DeepSet).
    """
    if deepset_on == True:
        return  DeepSet(
                        mlp_in= MLP(input_dim  = input_dim,
                                    hidden_dim = config.hidden_dim,
                                    out_dim    = config.hidden_dim,
                                    hidden_act = hidden_activation,
                                    out_act    = Identity(),
                                    depth      = config.depth,
                                    dropout    = config.dropout) ,
                        mlp_mid= MLP(input_dim = config.hidden_dim,
                                    hidden_dim = config.hidden_dim,
                                    out_dim    = config.hidden_dim,
                                    hidden_act = hidden_activation,
                                    out_act    = Identity(),
                                    depth      = config.depth,
                                    dropout    = config.dropout) ,
                        mlp_out= MLP(input_dim = config.hidden_dim,
                                    hidden_dim = config.hidden_dim,
                                    out_dim    = output_dim,
                                    hidden_act = hidden_activation,
                                    out_act    = output_activation,
                                    depth      = config.depth,
                                    dropout    = config.dropout)
                        )
    else:
        return  MLP(
                    input_dim  = input_dim,
                    hidden_dim = config.hidden_dim,
                    out_dim    = output_dim,
                    hidden_act = hidden_activation,
                    out_act    = output_activation,
                    depth      = config.depth,
                    dropout    = config.dropout
                    )