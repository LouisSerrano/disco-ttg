# Neural Operator Splitting Framework

A comprehensive framework for studying neural operator splitting methods applied to advection-diffusion equations. This implementation compares neural network-based operator splitting with classical methods across various temporal integrators and splitting schemes.

## Overview

This framework implements and tests the concept of using trained neural ODEs to approximate individual operators (advection and diffusion) and then compose them using operator splitting methods (Lie, Strang). The key research question is: **How well can neural operators compose compared to classical operator splitting and direct neural ODE solutions?**

## Key Features

- **Neural ODE Operators**: Simple MLP architectures trained to approximate pure advection and diffusion operators
- **Operator Splitting Methods**: Implementation of Lie and Strang splitting using neural operators
- **Comprehensive Testing**: Compare neural vs classical methods across different:
  - Time step sizes (dt convergence studies)
  - ODE solvers (euler, rk2, rk4, dopri5)
  - Initial conditions and parameter regimes
- **Ground Truth Comparison**: Use existing FFT solvers as reference solutions
- **Visualization & Analysis**: Comprehensive analysis tools and automated report generation

## Directory Structure

```
neural-operator-splitting/
├── __init__.py                    # Package initialization
├── data_generation.py             # Generate training data using FFT
├── neural_ode_operators.py        # Neural ODE definitions
├── training.py                    # Train neural operators
├── neural_splitting_methods.py    # Neural operator splitting implementation
├── test_neural_splitting.py       # Comprehensive test suite
├── analysis.py                    # Visualization and analysis tools
└── README.md                      # This file
```

## Quick Start

### 1. Install Dependencies

The framework requires:
- PyTorch 
- torchdiffeq (automatically installed if missing)
- NumPy, SciPy, Matplotlib
- Seaborn, Pandas (for analysis)

### 2. Run Basic Test

```bash
cd neural-operator-splitting/
python test_neural_splitting.py
```

When prompted, choose:
- `quick`: Fast test with minimal training (recommended for first run)
- `full`: Complete pipeline with comprehensive analysis
- `debug`: Minimal test for debugging

### 3. View Results

Results are saved in `test_results/` directory:
- `summary_report.txt`: Human-readable summary
- `full_pipeline_results.json`: Detailed results data
- `figures/`: Generated plots and visualizations

## Detailed Usage

### Training Neural Operators

```python
from training import train_neural_operators

# Train neural ODEs for advection and diffusion
results = train_neural_operators(
    nx=64,              # Spatial grid points
    hidden_dim=32,      # Neural network hidden dimension
    n_layers=3,         # Number of layers
    num_epochs=50,      # Training epochs
    method='rk4'        # ODE solver for training
)
```

### Using Neural Splitting Methods

```python
from neural_splitting_methods import NeuralOperatorSplitting
from neural_ode_operators import create_neural_operators

# Create and load trained models
operators = create_neural_operators(nx=64, L=2*np.pi)
# ... load trained weights ...

# Create splitting framework
neural_splitting = NeuralOperatorSplitting(
    operators['advection'], 
    operators['diffusion']
)

# Apply Lie splitting
solution = neural_splitting.lie_splitting(
    u0=initial_condition,
    dt=0.01,
    nt=100,
    method='rk4'
)
```

### Comprehensive Analysis

```python
from analysis import NeuralSplittingAnalyzer

# Initialize analyzer
analyzer = NeuralSplittingAnalyzer('./test_results')

# Create comprehensive report
report_dir = analyzer.create_comprehensive_report(results)
```

## Architecture Details

### Neural ODE Operators

The framework implements two types of neural operators:

1. **Simple MLPs**: General neural networks that learn operator dynamics
   ```python
   class AdvectionNeuralODE(nn.Module):
       # Learns: ∂u/∂t + β*∂u/∂x = 0
   
   class DiffusionNeuralODE(nn.Module):
       # Learns: ∂u/∂t = ν*∂²u/∂x²
   ```

2. **Physics-Informed**: Networks that incorporate known differential operators
   ```python
   class PhysicsInformedODE(nn.Module):
       # Physics term + neural correction
   ```

### Operator Splitting Methods

- **Lie Splitting** (First-order): `A(dt) ∘ D(dt)`
- **Strang Splitting** (Second-order): `D(dt/2) ∘ A(dt) ∘ D(dt/2)`
- **Alternating Splitting**: Switches order at each time step

### Integration with Existing Framework

The neural splitting framework leverages the existing `operator-splitting/` directory:
- Uses FFT solvers for ground truth generation
- Imports classical splitting methods for comparison
- Maintains consistent interface and error metrics

## Key Experiments

### 1. Training Quality Assessment
- Train neural ODEs on pure advection/diffusion trajectories
- Evaluate fitting quality vs FFT ground truth
- Study parameter sensitivity (β, ν coefficients)

### 2. Operator Composition Study
- Compare neural splitting vs direct neural ODE on combined problem
- Analyze composition errors and stability
- Test different splitting orders and methods

### 3. Temporal Integrator Impact
- Study effect of ODE solver choice (euler, rk4, dopri5)
- Investigate stability limits and accuracy trade-offs
- Compare computational efficiency

### 4. dt Convergence Analysis
- Test convergence rates with decreasing time steps
- Compare theoretical vs observed convergence orders
- Identify optimal dt ranges for different methods

## Results Interpretation

### Success Metrics
1. **Training Convergence**: Neural ODEs successfully fit individual operators
2. **Composition Accuracy**: Neural splitting matches classical methods
3. **Stability**: Methods remain stable across dt range
4. **Convergence Rates**: Appropriate theoretical convergence orders

### Key Outputs
- **Error Plots**: L2 and L∞ errors vs ground truth
- **Convergence Studies**: Error vs dt on log-log scale
- **Method Comparison**: Relative performance across test cases
- **Solver Analysis**: Impact of different ODE integrators

## Limitations and Future Work

### Current Limitations
- 1D spatial domain only
- Simple MLP architectures
- Limited to advection-diffusion equations
- Small network sizes for computational efficiency

### Future Extensions
- 2D/3D spatial domains
- More sophisticated neural architectures (CNNs, Transformers)
- Additional PDE operators (reaction, nonlinear terms)
- Adaptive time stepping and error control
- Integration with physics-informed neural networks

## Citation

If you use this framework in your research, please cite:

```
Neural Operator Splitting Framework for Advection-Diffusion Equations
Implementation as part of DISCO-Ball project
```

## Contributing

This framework is designed for research purposes. Key areas for contribution:
- Additional neural architectures
- Extended PDE operators
- Advanced splitting methods
- Performance optimizations
- Validation on complex test cases

## Troubleshooting

### Common Issues

1. **Training Instability**: Reduce learning rate, increase regularization
2. **ODE Solver Failures**: Try different solvers or smaller time steps
3. **Memory Issues**: Reduce batch size or network dimensions
4. **Convergence Problems**: Check data generation and normalization

### Debug Mode

Run with debug settings for rapid testing:
```bash
python test_neural_splitting.py
# Choose 'debug' when prompted
```

This uses minimal training (5 epochs) and simple test cases for quick validation.

## License

This code is part of the DISCO-Ball research project and follows the same licensing terms.