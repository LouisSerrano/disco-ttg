"""
Simplified main script for neural operator splitting experiments.
Uses modular components for better maintainability.
"""

import argparse
import os

# Import the experiment core
from experiment_core import main as run_main_experiment

# Import additional utilities
from visualization import create_coefficient_estimation_plots
from results_management import organize_results_by_parameters


def main():
    """Main entry point that delegates to the original main function."""
    # Simply call the main function from experiment_core
    run_main_experiment()


if __name__ == "__main__":
    main()