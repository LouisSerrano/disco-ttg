"""
Neural Operator Splitting Package

This package contains modularized components for neural operator splitting experiments,
including data generation, operator utilities, visualization, and results management.
"""

from .data_generation import RelativeL2, TemporalBatchDatasetFly
from .operator_utils import (
    get_batched_operators, 
    sequential_operator_composition, 
    solve_single_operator_ode, 
    sparsify_weights
)
from .visualization import (
    create_trajectory_snapshots, 
    create_coefficient_estimation_plots, 
    create_plots
)
from .results_management import organize_results_by_parameters, save_results
from .experiment_core import NeuralOperatorSplittingExperiment

__all__ = [
    # Data generation
    'RelativeL2',
    'TemporalBatchDatasetFly',
    
    # Operator utilities
    'get_batched_operators',
    'sequential_operator_composition', 
    'solve_single_operator_ode',
    'sparsify_weights',
    
    # Visualization
    'create_trajectory_snapshots',
    'create_coefficient_estimation_plots',
    'create_plots',
    
    # Results management
    'organize_results_by_parameters',
    'save_results',
    
    # Core experiment
    'NeuralOperatorSplittingExperiment'
]