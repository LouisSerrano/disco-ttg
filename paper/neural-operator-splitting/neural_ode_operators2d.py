"""
Neural ODE Operators for Advection and Diffusion

This module implements neural ODE operators that learn to approximate individual
advection and diffusion operators using simple MLP architectures and torchdiffeq.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    print("Warning: torchdiffeq not available. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "torchdiffeq"])
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True


def apply_circular_padding_2d(x: torch.Tensor, padding: int) -> torch.Tensor:
    """Apply circular/periodic padding to 2D tensor for periodic boundary conditions.
    
    Args:
        x: Input tensor of shape (batch, channels, height, width)
        padding: Amount of padding to apply on all sides
        
    Returns:
        Padded tensor with periodic boundary conditions
    """
    return F.pad(x, (padding, padding, padding, padding), mode='circular')


class SimpleMLPODE(nn.Module):
    """Simple MLP neural ODE function for learning operator dynamics."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 activation: str = 'relu'):
        """
        Initialize MLP ODE function.
        
        Args:
            input_dim: Dimension of input (spatial grid size)
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            activation: Activation function ('relu', 'relu', 'elu')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Choose activation function
        if activation.lower() == 'relu':
            self.activation = nn.relu()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation)
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with small values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for neural ODE.
        
        Args:
            t: Time tensor (scalar)
            u: State tensor (batch_size, spatial_dim)
            
        Returns:
            Time derivative du/dt
        """
        batch_size = u.shape[0]
        u_flat = u.view(batch_size, -1)  # Flatten spatial dimensions
        dudt = self.net(u_flat)
        return dudt.view_as(u)


class ResidualBlock1D(nn.Module):
    """1D Residual block with LayerNorm."""
    
    def __init__(self, channels: int, kernel_size: int = 3, padding_mode: str = 'zeros'):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.norm2 = nn.LayerNorm(channels)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        # First conv + norm + activation
        out = self.conv1(x)
        # LayerNorm expects (batch, features, seq_len) -> (batch, seq_len, features)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        
        # Second conv + norm
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        
        # Add residual connection
        out += identity
        out = self.activation(out)
        
        return out


class ResidualBlock2D(nn.Module):
    """2D Residual block with LayerNorm and periodic padding support."""
    
    def __init__(self, channels: int, kernel_size: int = 3, padding_mode: str = 'zeros', dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.dilation = dilation
        # Calculate padding for dilated convolutions to maintain spatial dimensions
        self.padding = dilation * (kernel_size - 1) // 2
        
        # For periodic padding, we'll handle it manually since Conv2d doesn't support 'circular'
        conv_padding = self.padding if padding_mode != 'circular' else 0
        conv_padding_mode = padding_mode if padding_mode != 'circular' else 'zeros'
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 
                              padding=conv_padding, padding_mode=conv_padding_mode, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 
                              padding=conv_padding, padding_mode=conv_padding_mode, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.activation = nn.ReLU()
        
    def _apply_periodic_padding(self, x):
        """Apply periodic (circular) padding to 2D tensor using F.pad."""
        if self.padding_mode == 'circular':
            return apply_circular_padding_2d(x, self.padding)
        return x
        
    def forward(self, x):
        identity = x
        
        # First conv + norm + activation
        if self.padding_mode == 'circular':
            out = self._apply_periodic_padding(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
            
        # GroupNorm expects (batch, channels, height, width)
        out = self.norm1(out)
        out = self.activation(out)
        
        # Second conv + norm
        if self.padding_mode == 'circular':
            out = self._apply_periodic_padding(out)
            out = self.conv2(out)
        else:
            out = self.conv2(out)
            
        out = self.norm2(out)
        
        # Add residual connection
        out += identity
        out = self.activation(out)
        
        return out


class ResNet1DODE(nn.Module):
    """1D ResNet neural ODE function with residual blocks and LayerNorm."""
    
    def __init__(self, 
                 input_dim: int,
                 n_channels: int = 32,
                 kernel_size: int = 3,
                 n_blocks: int = 3,
                 activation: str = 'relu',
                 padding_mode: str = 'zeros'):
        """
        Initialize 1D ResNet ODE function.
        
        Args:
            input_dim: Dimension of input (spatial grid size)
            n_channels: Number of convolutional channels
            kernel_size: Kernel size for convolutions
            n_blocks: Number of residual blocks
            activation: Activation function ('relu', 'elu')
            padding_mode: Padding mode ('zeros', 'reflect', 'replicate', 'circular')
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        self.padding_mode = padding_mode
        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        padding = kernel_size // 2
        
        # Input projection: 1 channel -> n_channels
        self.input_conv = nn.Conv1d(1, n_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.input_norm = nn.LayerNorm(n_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(n_channels, kernel_size, padding_mode) 
            for _ in range(n_blocks)
        ])
        
        # Output projection: n_channels -> 1 channel
        self.output_conv = nn.Conv1d(n_channels, 1, kernel_size, padding=padding, padding_mode=padding_mode)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with small values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ResNet neural ODE.
        
        Args:
            t: Time tensor (scalar)
            u: State tensor (batch_size, spatial_dim)
            
        Returns:
            Time derivative du/dt
        """
        batch_size = u.shape[0]
        
        # Reshape for 1D convolution: (batch_size, channels=1, spatial_dim)
        if len(u.shape) == 2:  # (batch_size, spatial_dim)
            u_reshaped = u.unsqueeze(1)  # (batch_size, 1, spatial_dim)
        elif len(u.shape) == 3:  # (batch_size, 1, spatial_dim)
            u_reshaped = u  # Already in correct format
        else:
            raise ValueError(f"Unexpected input shape: {u.shape}")
        
        # Input projection
        x = self.input_conv(u_reshaped)
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Output projection
        dudt_reshaped = self.output_conv(x)  # (batch_size, 1, spatial_dim)
        
        return dudt_reshaped


class CNN1DODE(nn.Module):
    """1D CNN neural ODE function for learning spatial operator dynamics."""
    
    def __init__(self, 
                 input_dim: int,
                 n_channels: int = 32,
                 kernel_size: int = 5,
                 n_layers: int = 3,
                 activation: str = 'relu',
                 padding_mode: str = 'zeros'):
        """
        Initialize 1D CNN ODE function.
        
        Args:
            input_dim: Dimension of input (spatial grid size)
            n_channels: Number of convolutional channels
            kernel_size: Kernel size for convolutions
            n_layers: Number of convolutional layers
            activation: Activation function ('relu', 'relu', 'elu')
            padding_mode: Padding mode ('zeros', 'reflect', 'replicate', 'circular')
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.padding_mode = padding_mode
        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Padding to maintain spatial dimensions
        padding = kernel_size // 2
        
        # Build CNN layers
        layers = []
        
        # Input layer: from 1 channel to n_channels
        layers.append(nn.Conv1d(1, n_channels, kernel_size, padding=padding, padding_mode=self.padding_mode))
        layers.append(self.activation)
        
        # Hidden layers: n_channels to n_channels
        for _ in range(n_layers - 1):
            layers.append(nn.Conv1d(n_channels, n_channels, kernel_size, padding=padding, padding_mode=self.padding_mode))
            layers.append(self.activation)
        
        # Output layer: from n_channels to 1 channel
        layers.append(nn.Conv1d(n_channels, 1, kernel_size, padding=padding, padding_mode=self.padding_mode))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with small values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CNN neural ODE.
        
        Args:
            t: Time tensor (scalar)
            u: State tensor (batch_size, spatial_dim)
            
        Returns:
            Time derivative du/dt
        """
        batch_size = u.shape[0]

        #print('u', u.shape)
        
        # Reshape for 1D convolution: (batch_size, channels=1, spatial_dim)
        if len(u.shape) == 2:  # (batch_size, spatial_dim)
            u_reshaped = u.unsqueeze(1)  # (batch_size, 1, spatial_dim)
        elif len(u.shape) == 3:  # (batch_size, 1, spatial_dim)
            u_reshaped = u  # Already in correct format
        else:
            raise ValueError(f"Unexpected input shape: {u.shape}")
        
        # Apply CNN
        dudt_reshaped = self.net(u_reshaped)  # (batch_size, 1, spatial_dim)
        #print('dudt', dudt_reshaped.shape)
        
        # Reshape back to original format
        #dudt = dudt_reshaped.squeeze(1)  # (batch_size, spatial_dim)
        
        return dudt_reshaped


class ResNet2DODE(nn.Module):
    """2D ResNet neural ODE function with residual blocks and LayerNorm."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 n_channels: int = 32,
                 kernel_size: int = 5,
                 n_blocks: int = 3,
                 activation: str = 'relu',
                 padding_mode: str = 'zeros',
                 use_multiscale: bool = True,
                 dilations: List[int] = None):
        """
        Initialize 2D ResNet ODE function.
        
        Args:
            input_shape: Tuple of (height, width) for input dimensions
            n_channels: Number of convolutional channels
            kernel_size: Kernel size for convolutions
            n_blocks: Number of residual blocks
            activation: Activation function ('relu', 'elu')
            padding_mode: Padding mode ('zeros', 'reflect', 'replicate', 'circular')
            use_multiscale: If True, use dilated convolutions for multi-scale features
            dilations: List of dilation rates for each block. If None and use_multiscale=True, 
                      uses [1, 2, 4, ...] pattern
        """
        super().__init__()
        self.input_shape = input_shape
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        self.padding_mode = padding_mode
        self.use_multiscale = use_multiscale
        
        # Set up dilation pattern for multi-scale features
        if use_multiscale:
            if dilations is None:
                # Default pattern: exponentially increasing dilations
                self.dilations = [2**i for i in range(n_blocks)]
            else:
                if len(dilations) != n_blocks:
                    raise ValueError(f"Length of dilations ({len(dilations)}) must match n_blocks ({n_blocks})")
                self.dilations = dilations
        else:
            self.dilations = [1] * n_blocks
        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # For periodic padding, we'll handle it manually since Conv2d doesn't support 'circular'
        self.padding = kernel_size // 2
        conv_padding = self.padding if padding_mode != 'circular' else 0
        conv_padding_mode = padding_mode if padding_mode != 'circular' else 'zeros'
        
        # Input projection: 1 channel -> n_channels
        self.input_conv = nn.Conv2d(1, n_channels, kernel_size, 
                                   padding=conv_padding, padding_mode=conv_padding_mode)
        self.input_norm = nn.GroupNorm(num_groups=min(8, n_channels), num_channels=n_channels)
        
        # Residual blocks with different dilations for multi-scale features
        self.res_blocks = nn.ModuleList([
            ResidualBlock2D(n_channels, kernel_size, padding_mode, dilation=self.dilations[i]) 
            for i in range(n_blocks)
        ])
        
        # Output projection: n_channels -> 1 channel
        self.output_conv = nn.Conv2d(n_channels, 1, kernel_size, 
                                    padding=conv_padding, padding_mode=conv_padding_mode)
        
        # Initialize weights
        self._initialize_weights()
    
    def _apply_periodic_padding(self, x):
        """Apply periodic (circular) padding to 2D tensor using F.pad."""
        if self.padding_mode == 'circular':
            return apply_circular_padding_2d(x, self.padding)
        return x
    
    def _initialize_weights(self):
        """Initialize network weights with small values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ResNet neural ODE.
        
        Args:
            t: Time tensor (scalar)
            u: State tensor (batch_size, height, width) or (batch_size, 1, height, width)
            
        Returns:
            Time derivative du/dt
        """
        #print(f"ResNet2DODE.forward - Input u shape: {u.shape}")
        batch_size = u.shape[0]
        
        # Reshape for 2D convolution: (batch_size, channels=1, height, width)
        if len(u.shape) == 3:  # (batch_size, height, width)
            u_reshaped = u.unsqueeze(1)  # (batch_size, 1, height, width)
            to_squeeze = True
            #print(f"ResNet2DODE.forward - Reshaped from 3D to 4D: {u_reshaped.shape}")
        elif len(u.shape) == 4:  # (batch_size, 1, height, width)
            u_reshaped = u  # Already in correct format
            to_squeeze = False
            #print(f"ResNet2DODE.forward - Already 4D: {u_reshaped.shape}")
        else:
            raise ValueError(f"Unexpected input shape: {u.shape}")
        
        # Input projection with periodic padding if needed
        if self.padding_mode == 'circular':
            x = self._apply_periodic_padding(u_reshaped)
            #print(f"ResNet2DODE.forward - After padding: {x.shape}")
            x = self.input_conv(x)
        else:
            #print(f"ResNet2DODE.forward - Before input_conv: {u_reshaped.shape}")
            x = self.input_conv(u_reshaped)
            
        # Apply group norm
        x = self.input_norm(x)  # (batch, channels, height, width)
        x = self.activation(x)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Output projection with periodic padding if needed
        if self.padding_mode == 'circular':
            x = self._apply_periodic_padding(x)
            dudt_reshaped = self.output_conv(x)
        else:
            dudt_reshaped = self.output_conv(x)

        if to_squeeze:
            dudt_reshaped = dudt_reshaped.squeeze(1)  # (batch_size, height, width)
            #print(f"ResNet2DODE.forward - Squeezed output shape: {dudt_reshaped.shape}")
        
        return dudt_reshaped


class WrapperNeuralODE(nn.Module):
    """Neural ODE for learning advection operator: ∂u/∂t + β*∂u/∂x = 0"""
    
    def __init__(self, 
                 nx: int = None,
                 L: float = 2*np.pi,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 activation: str = 'relu',
                 use_cnn: str = 'resnet',
                 kernel_size: int = 5,
                 padding_mode: str = "circular",
                 num_steps: int = 1,
                 input_shape: Tuple[int, int] = None):
        """
        Initialize advection neural ODE for 1D or 2D data.
        
        Args:
            nx: Number of spatial grid points (for 1D, ignored if input_shape is provided)
            L: Domain length
            hidden_dim: Hidden layer dimension (for MLP) or number of channels (for CNN)
            n_layers: Number of hidden layers
            activation: Activation function
            use_cnn: Architecture type - False/'mlp' for MLP, True/'cnn' for CNN, 'resnet' for ResNet
            kernel_size: CNN/ResNet kernel size (only used if use_cnn is True or 'resnet')
            padding_mode: Padding mode for CNN/ResNet ('zeros', 'reflect', 'replicate', 'circular')
            num_steps: Default number of time steps for evaluation (can be overridden at inference)
            input_shape: Tuple of (height, width) for 2D data. If provided, uses 2D ResNet
        """
        super().__init__()
        self.nx = nx
        self.L = L
        self.input_shape = input_shape
        self.use_cnn = use_cnn
        self.num_steps = num_steps
        self.is_2d = input_shape is not None
        
        # For 2D case, always use circular padding for periodic boundaries
        if self.is_2d:
            self.ode_func = ResNet2DODE(input_shape, hidden_dim, kernel_size, n_layers, activation, 'circular')
        else:
            raise ValueError("WrapperNeuralODE currently only supports 2D input")
        
        # Parameters that can be learned or fixed
        self.beta = nn.Parameter(torch.tensor(1.0))  # Advection coefficient
        
    def forward(self, u0: torch.Tensor, 
                t_span: torch.Tensor = None,
                T: float = 1.0,
                num_steps: int = None,
                method: str = 'rk4',
                rtol: float = 1e-7,
                atol: float = 1e-9,
                use_adjoint: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Integrate the neural ODE forward in time.
        
        Args:
            u0: Initial condition (batch_size, nx) for 1D or (batch_size, height, width) for 2D
            t_span: Time points to evaluate at (if None, uses T and num_steps)
            T: Final time (default 1.0) - used if t_span is None
            num_steps: Number of time steps (if None, uses self.num_steps)
            method: ODE solver method ('euler', 'rk4', 'dopri5', etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            use_adjoint: If True, use adjoint method for memory-efficient gradients
            
        Returns:
            Tuple of (number of steps, solution at time points)
        """
        # Use default num_steps if not provided
        if num_steps is None:
            num_steps = self.num_steps
            
        # Create default t_span if not provided
        if t_span is None:
            device = u0.device
            # Ensure T is a scalar (0-dimensional tensor)
            T_scalar = T.item() if isinstance(T, torch.Tensor) and T.numel() == 1 else T
            t_span = torch.linspace(0, T_scalar, num_steps + 1, device=device)
            ##print(f"T: {T}, t_span: {t_span}")
        
        # Choose between standard and adjoint ODE solver
        solver_func = odeint_adjoint if use_adjoint else odeint
        
        # Solve neural ODE
        solution = solver_func(
            self.ode_func,
            u0,
            t_span,
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        # Handle potential nested tuple structure from odeint
        if isinstance(solution, tuple):
            # Extract the actual tensor from nested structure
            actual_solution = solution
            while isinstance(actual_solution, tuple) and len(actual_solution) > 0:
                if isinstance(actual_solution[-1], torch.Tensor):
                    actual_solution = actual_solution[-1]
                    break
                actual_solution = actual_solution[-1]
            solution = actual_solution
        
        # Return tuple format expected by training code
        return len(t_span), solution
    
    def get_advection_coefficient(self) -> float:
        """Get the learned advection coefficient."""
        return self.beta.item()


class DiffusionNeuralODE(nn.Module):
    """Neural ODE for learning diffusion operator: ∂u/∂t = ν*∂²u/∂x²"""
    
    def __init__(self, 
                 nx: int = None,
                 L: float = 2*np.pi,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 activation: str = 'relu',
                 use_cnn: str = 'resnet',
                 kernel_size: int = 5,
                 padding_mode: str = 'zeros',
                 num_steps: int = 10,
                 input_shape: Tuple[int, int] = None):
        """
        Initialize diffusion neural ODE for 1D or 2D data.
        
        Args:
            nx: Number of spatial grid points (for 1D, ignored if input_shape is provided)
            L: Domain length  
            hidden_dim: Hidden layer dimension (for MLP) or number of channels (for CNN)
            n_layers: Number of hidden layers
            activation: Activation function
            use_cnn: Architecture type - False/'mlp' for MLP, True/'cnn' for CNN, 'resnet' for ResNet
            kernel_size: CNN/ResNet kernel size (only used if use_cnn is True or 'resnet')
            padding_mode: Padding mode for CNN/ResNet ('zeros', 'reflect', 'replicate', 'circular')
            num_steps: Default number of time steps for evaluation (can be overridden at inference)
            input_shape: Tuple of (height, width) for 2D data. If provided, uses 2D ResNet
        """
        super().__init__()
        self.nx = nx
        self.L = L
        self.input_shape = input_shape
        self.use_cnn = use_cnn
        self.num_steps = num_steps
        self.is_2d = input_shape is not None
        
        # For 1D case
        if not self.is_2d:
            if nx is None:
                raise ValueError("Either nx or input_shape must be provided")
            self.dx = L / nx
            # Spatial grid
            self.register_buffer('x', torch.linspace(0, L, nx, dtype=torch.float32))
            
            # Neural ODE function - choose between MLP, CNN, and ResNet
            if use_cnn == 'resnet':
                self.ode_func = ResNet1DODE(nx, hidden_dim, kernel_size, n_layers, activation, padding_mode)
            elif use_cnn:
                self.ode_func = CNN1DODE(nx, hidden_dim, kernel_size, n_layers, activation, padding_mode)
            else:
                self.ode_func = SimpleMLPODE(nx, hidden_dim, n_layers, activation)
        else:
            # For 2D case, always use ResNet2DODE with circular padding for periodic boundaries
            padding_mode_2d = 'circular'  # Force circular padding for 2D cases
            self.ode_func = ResNet2DODE(input_shape, hidden_dim, kernel_size, n_layers, activation, padding_mode_2d)
        
        # Diffusion coefficient
        self.nu = nn.Parameter(torch.tensor(0.1))  # Diffusion coefficient
        
    def forward(self, u0: torch.Tensor,
                t_span: torch.Tensor = None,
                T: float = 1.0,
                num_steps: int = None,
                method: str = 'rk4',
                rtol: float = 1e-7,
                atol: float = 1e-9,
                use_adjoint: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Integrate the neural ODE forward in time.
        
        Args:
            u0: Initial condition (batch_size, nx)
            t_span: Time points to evaluate at (if None, uses T and num_steps)
            T: Final time (default 1.0) - used if t_span is None
            num_steps: Number of time steps (if None, uses self.num_steps)
            method: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            use_adjoint: If True, use adjoint method for memory-efficient gradients
            
        Returns:
            Tuple of (number of steps, solution at time points (len(t_span), batch_size, nx))
        """
        # Use default num_steps if not provided
        if num_steps is None:
            num_steps = self.num_steps
            
        # Create default t_span if not provided
        if t_span is None:
            device = u0.device
            # Ensure T is a scalar (0-dimensional tensor)
            T_scalar = T.item() if isinstance(T, torch.Tensor) and T.numel() == 1 else T
            t_span = torch.linspace(0, T_scalar, num_steps + 1, device=device)
            ##print(f"T: {T}, t_span: {t_span}")
        
        # Choose between standard and adjoint ODE solver
        solver_func = odeint_adjoint if use_adjoint else odeint
        
        solution = solver_func(
            self.ode_func,
            u0,
            t_span,
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        # Handle potential nested tuple structure from odeint
        if isinstance(solution, tuple):
            # Extract the actual tensor from nested structure
            actual_solution = solution
            while isinstance(actual_solution, tuple) and len(actual_solution) > 0:
                if isinstance(actual_solution[-1], torch.Tensor):
                    actual_solution = actual_solution[-1]
                    break
                actual_solution = actual_solution[-1]
            solution = actual_solution
        
        # Return tuple format expected by training code
        return len(t_span), solution
    
    def get_diffusion_coefficient(self) -> float:
        """Get the learned diffusion coefficient."""
        return self.nu.item()



def create_neural_operators(nx: int = None, 
                          L: float = 2*np.pi,
                          hidden_dim: int = 64,
                          n_layers: int = 3,
                          use_cnn: str = 'resnet',
                          kernel_size: int = 5,
                          activation: str = 'relu',
                          padding_mode: str = 'zeros',
                          num_steps: int = 10,
                          input_shape: Tuple[int, int] = None) -> Dict[str, nn.Module]:
    """
    Create neural operators for advection and diffusion.
    
    Args:
        nx: Number of spatial grid points (for 1D, ignored if input_shape is provided)
        L: Domain length
        hidden_dim: Hidden layer dimension (for MLP) or number of channels (for CNN/ResNet)
        n_layers: Number of hidden layers (for MLP/CNN) or residual blocks (for ResNet)
        use_cnn: Architecture type - False for MLP, True for CNN, 'resnet' for ResNet
        kernel_size: CNN/ResNet kernel size (only used if use_cnn is True or 'resnet')
        activation: Activation function
        padding_mode: Padding mode for CNN/ResNet ('zeros', 'reflect', 'replicate', 'circular')
        num_steps: Number of integration steps for ODE solver
        input_shape: Tuple of (height, width) for 2D data. If provided, creates 2D operators
        
    Returns:
        Dictionary containing the neural operators
    """
    advection_ode = WrapperNeuralODE(nx, L, hidden_dim, n_layers, activation, 
                                     use_cnn, kernel_size, padding_mode, num_steps, input_shape)
    diffusion_ode = DiffusionNeuralODE(nx, L, hidden_dim, n_layers, activation, 
                                     use_cnn, kernel_size, padding_mode, num_steps, input_shape)
    
    return {
        'advection': advection_ode,
        'diffusion': diffusion_ode
    }


if __name__ == "__main__":
    # Test the neural operators
    #print("Testing Neural ODE Operators...")
    
    nx = 64
    L = 2 * np.pi
    batch_size = 4
    
    # Create operators
    operators = create_neural_operators(nx, L, hidden_dim=32, n_layers=2)
    
    # Test 2D ResNet
    #print("\nTesting 2D ResNet Neural ODE...")
    try:
        # Create 2D test data
        input_shape = (32, 32)
        resnet_2d = ResNet2DODE(input_shape, n_channels=16, n_blocks=2, padding_mode='circular')
        
        # Create test tensor (batch_size, height, width)
        u0_2d = torch.randn(batch_size, input_shape[0], input_shape[1])
        t_dummy = torch.tensor(0.0)
        
        # Forward pass
        dudt_2d = resnet_2d(t_dummy, u0_2d)
        #print(f"2D ResNet input shape: {u0_2d.shape}")
        #print(f"2D ResNet output shape: {dudt_2d.shape}")
        #print(f"2D ResNet parameters: {sum(p.numel() for p in resnet_2d.parameters())}")
        #print("2D ResNet test passed!")
        
    except Exception as e:
        print(f"2D ResNet test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test data
    x = torch.linspace(0, L, nx)
    u0 = torch.sin(x).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, nx)
    t_span = torch.linspace(0, 0.1, 6)  # Small time span for testing
    
    print(f"Initial condition shape: {u0.shape}")
    print(f"Time span: {t_span}")
    
    # Test advection operator
    print("\nTesting Advection Neural ODE...")
    advection_ode = operators['advection']
    try:
        n_steps, advection_solution = advection_ode(u0, t_span, method='euler')
        print(f"Advection solution shape: {advection_solution.shape}")
        print(f"Number of steps: {n_steps}")
        print(f"Learned advection coefficient: {advection_ode.get_advection_coefficient():.4f}")
    except Exception as e:
        print(f"Advection test failed: {e}")
    
    # Test diffusion operator
    print("\nTesting Diffusion Neural ODE...")
    diffusion_ode = operators['diffusion']
    try:
        n_steps, diffusion_solution = diffusion_ode(u0, t_span, method='euler')
        print(f"Diffusion solution shape: {diffusion_solution.shape}")
        print(f"Number of steps: {n_steps}")
        print(f"Learned diffusion coefficient: {diffusion_ode.get_diffusion_coefficient():.4f}")
    except Exception as e:
        print(f"Diffusion test failed: {e}")
    
    # Test CNN-based operators
    print("\nTesting CNN-Based Neural ODEs...")
    cnn_operators = create_neural_operators(nx, L, use_cnn=True, hidden_dim=16, kernel_size=3)
    
    try:
        cnn_advection = cnn_operators['advection']
        n_steps, cnn_solution = cnn_advection(u0, t_span, method='euler')
        print(f"CNN advection solution shape: {cnn_solution.shape}")
        print(f"CNN model parameters: {sum(p.numel() for p in cnn_advection.parameters())}")
        
        cnn_diffusion = cnn_operators['diffusion']
        n_steps, cnn_diff_solution = cnn_diffusion(u0, t_span, method='euler')
        print(f"CNN diffusion solution shape: {cnn_diff_solution.shape}")
        print(f"CNN diffusion parameters: {sum(p.numel() for p in cnn_diffusion.parameters())}")
        
    except Exception as e:
        print(f"CNN test failed: {e}")
    
    # Test ResNet-based operators
    print("\nTesting ResNet-Based Neural ODEs...")
    resnet_operators = create_neural_operators(nx, L, use_cnn='resnet', hidden_dim=16, n_layers=2, kernel_size=3)
    
    try:
        resnet_advection = resnet_operators['advection']
        n_steps, resnet_solution = resnet_advection(u0, t_span, method='euler')
        print(f"ResNet advection solution shape: {resnet_solution.shape}")
        print(f"ResNet model parameters: {sum(p.numel() for p in resnet_advection.parameters())}")
        
        resnet_diffusion = resnet_operators['diffusion']
        n_steps, resnet_diff_solution = resnet_diffusion(u0, t_span, method='euler')
        print(f"ResNet diffusion solution shape: {resnet_diff_solution.shape}")
        print(f"ResNet diffusion parameters: {sum(p.numel() for p in resnet_diffusion.parameters())}")
        
    except Exception as e:
        print(f"ResNet test failed: {e}")
    
    print("\nNeural operator tests completed!")