
import copy
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SDFModel(nn.Module):
    """Deep neural network for learning signed distance functions of multiple shapes.
    
    Implements the DeepSDF architecture with optional skip connections
    for improved gradient flow and shape representation.
    """

    def __init__(
        self,
        num_layers: int,
        skip_connections: bool,
        latent_size: int,
        inner_dim: int = 512,
        output_dim: int = 1
    ):
        """Initialize SDF model with specified architecture.
        
        Args:
            num_layers: Total number of layers in the network.
            skip_connections: Whether to use skip connections (requires num_layers >= 5).
            latent_size: Dimension of the shape latent code.
            inner_dim: Hidden layer dimension. Defaults to 512.
            output_dim: Output dimension (1 for SDF values). Defaults to 1.
        """
        super(SDFModel, self).__init__()
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.latent_size = latent_size
        self._dim_coords = 3
        self._input_dim = self.latent_size + self._dim_coords
        self.skip_tensor_dim = copy.copy(self._input_dim)

        self.net = self._build_network(inner_dim)
        self.final_layer = nn.Sequential(
            nn.Linear(inner_dim, output_dim),
            nn.Tanh()
        )
        self.skip_layer = nn.Sequential(
            nn.Linear(inner_dim, inner_dim - self.skip_tensor_dim),
            nn.ReLU()
        )

    # ========== Internal API ==========

    def _build_network(self, inner_dim: int) -> nn.Sequential:
        """Build the main sequential network layers.
        
        Args:
            inner_dim: Hidden layer dimension.
            
        Returns:
            Sequential module containing the main network layers.
        """
        num_extra_layers = 2 if (self.skip_connections and self.num_layers >= 8) else 1
        layers = []
        input_dim = self._input_dim

        for _ in range(self.num_layers - num_extra_layers):
            layers.append(
                nn.Sequential(
                    weight_norm(nn.Linear(input_dim, inner_dim)),
                    nn.ReLU()
                )
            )
            input_dim = inner_dim

        return nn.Sequential(*layers)

    def _forward_with_skip(self, x: torch.Tensor, input_data: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections.
        
        Args:
            x: Input tensor.
            input_data: Original input for skip connection.
            
        Returns:
            SDF values.
        """
        for i in range(3):
            x = self.net[i](x)
        x = self.skip_layer(x)
        x = torch.hstack((x, input_data))
        for i in range(self.num_layers - 5):
            x = self.net[3 + i](x)
        sdf = self.final_layer(x)
        return sdf

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass without skip connections.
        
        Args:
            x: Input tensor.
            
        Returns:
            SDF values.
        """
        x = self.net(x)
        sdf = self.final_layer(x)
        return sdf

    # ========== Public API ==========

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, latent_size + 3).
               Contains concatenated latent codes and 3D coordinates.
               
        Returns:
            SDF values of shape (batch_size, output_dim).
        """
        input_data = x.clone().detach()

        if self.skip_connections and self.num_layers >= 5:
            return self._forward_with_skip(x, input_data)
        else:
            if self.skip_connections:
                print(
                    "Warning: Skip connections require at least 5 layers. "
                    "Using standard forward pass."
                )
            return self._forward_standard(x)