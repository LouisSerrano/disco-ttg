"""
Neural Operator Splitting Framework

This module implements neural operator splitting methods for studying the composition
of neural operators in solving advection-diffusion equations. It provides tools for:

1. Data generation using existing FFT solvers
2. Neural ODE definitions for individual operators
3. Training neural operators on separate dynamics
4. Operator splitting methods with neural networks
5. Analysis and comparison with ground truth solutions
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

from .neural_ode_operators import AdvectionNeuralODE, DiffusionNeuralODE
from .data_generation import generate_training_data
from .neural_splitting_methods import NeuralOperatorSplitting

__all__ = [
    "AdvectionNeuralODE",
    "DiffusionNeuralODE", 
    "generate_training_data",
    "NeuralOperatorSplitting"
]