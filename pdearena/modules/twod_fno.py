# Author: Miguel Liu-Schiaffini (mliuschi@caltech.edu) and based on PDEArena library
import torch
from torch import nn
from torch.nn import functional as F

from .activations import ACTIVATION_REGISTRY

# Official implementation of FNO
from neuralop.models import FNO2d

class FNO2D(nn.Module):
    """
    Args:
        n_input_scalar_components (int): Number of input scalar components in the model
        n_input_vector_components (int): Number of input vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps to use in the input
        time_future (int): Number of time steps to predict in the output
        activation: Activation to use

        n_modes_width: Number of modes to keep in Fourier Layer, along the width
        n_modes_height: Number of modes to keep in Fourier Layer, along the width
        lifting_channels: Number of hidden channels of the lifting block of the FNO
        projection_channels: Number of hidden channels of the projection block of the FNO
        n_layers: Number of Fourier Layers
        hidden_channels: Number of channels in FNO
        use_mlp: Whether to use an MLP layer after each FNO block
    """
    padding = 9

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        activation: str = "gelu",

        n_modes_width: int,
        n_modes_height: int,
        lifting_channels: int,
        projection_channels: int,
        n_layers: int,
        hidden_channels: int = 64,
        use_mlp: bool,
    ):
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.non_linearity: nn.Module = ACTIVATION_REGISTRY.get(activation, None)

        self.insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        self.outsize_step = self.n_output_scalar_components + self.n_output_vector_components * 2
        self.outsize = time_future * self.outsize_step        

        self.FNO = FNO2d(
                n_modes_height = n_modes_height,
                n_modes_width = n_modes_width,
                hidden_channels = hidden_channels,
                in_channels = self.insize, 
                out_channels = self.outsize,
                lifting_channels = lifting_channels,
                projection_channels = projection_channels,
                n_layers = n_layers,
                non_linearity = self.non_linearity,
                use_mlp = use_mlp
            )

    def __repr__(self):
        return "FNO2D"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        out = self.FNO(x)

        return out.reshape(orig_shape[0], -1, self.outsize_step, *orig_shape[3:])