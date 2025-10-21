import torch
import argparse
import os
import time
from datetime import datetime

# Update these imports based on your actual project structure
from ttc_utils import (
    load_model_from_checkpoint, 
    create_dataset_for_equation,
    save_results,
    TRAINING_FILES,
    DEVICE
)
from ttc_methods import (
    test_direct_prediction,
    encode_operators_from_training_data,
    greedy_operator_selection,
    random_operator_selection,
    gradient_selection_multi_operator
)


# CONFIGURATION - Update these for your specific equation
EQUATION_TYPE = 'your_equation_type'  # Change this
DEFAULT_NUM_OPERATORS = 20
DEFAULT_NUM_SAMPLES = 10

# Strategy-specific parameters
GREEDY_MAX_OPERATORS = 5
RANDOM_NUM_COMPOSITIONS = 100
RANDOM_COMPOSITION_LENGTHS = [1, 2, 3, 4, 5]
GRADIENT_NUM_OPERATORS = 3
GRADIENT_EPOCHS = 200
GRADIENT_LR = 0.01
GRADIENT_REFINEMENT = 1
GRADIENT_SPLITTING = "strang"


def main():
    parser = argparse.ArgumentParser(description=f'Test time compute for {EQUATION_TYPE}')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_operators', type=int, default=DEFAULT_NUM_OPERATORS, help='Number of operators to encode')
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_SAMPLES, help='Number of test samples to evaluate')
    
    # Add custom arguments if needed
    # parser.add_argument('--custom_param', type=float, default=1.0, help='Custom parameter')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, lit_model = load_model_from_checkpoint(args.model_path)
    if model is None:
        print("Failed to load model")
        return
    
    # Load datasets
    print("\nLoading datasets...")
    test_loader, test_dataset = create_dataset_for_equation(EQUATION_TYPE, 'test')
    if test_loader is None:
        print("Failed to load test dataset")
        return
    
    # Encode operators from training data
    print("\nEncoding operators from training data...")
    theta_operators, _, operator_metadata = encode_operators_from_training_data(
        model, 
        TRAINING_FILES[EQUATION_TYPE],
        num_operators=args.num_operators
    )
    
    # Get test samples
    test_batch = next(iter(test_loader))
    test_input = test_batch['input'][:args.num_samples].to(DEVICE)
    test_target = test_batch['target'][:args.num_samples].to(DEVICE)
    
    # Extract any equation-specific parameters
    # Example: test_params = test_batch.get('your_param', torch.zeros(args.num_samples))[:args.num_samples]
    
    results = {
        'equation_type': EQUATION_TYPE,
        'timestamp': datetime.now().isoformat(),
        'num_operators': args.num_operators,
        'num_samples': args.num_samples,
        'config': {
            'greedy_max_operators': GREEDY_MAX_OPERATORS,
            'random_num_compositions': RANDOM_NUM_COMPOSITIONS,
            'gradient_num_operators': GRADIENT_NUM_OPERATORS,
            'gradient_epochs': GRADIENT_EPOCHS
        },
        'methods': {}
    }
    
    # 1. Direct prediction baseline
    print("\n1. Testing direct prediction...")
    start_time = time.time()
    direct_error, direct_pred = test_direct_prediction(model, test_loader)
    direct_time = time.time() - start_time
    results['methods']['direct'] = {
        'error': direct_error,
        'time': direct_time
    }
    print(f"Direct prediction error: {direct_error:.6f}")
    
    # 2. Greedy selection
    print("\n2. Testing greedy operator selection...")
    greedy_results = []
    start_time = time.time()
    
    for i in range(args.num_samples):
        # Add any sample-specific logging here
        composition, error, pred = greedy_operator_selection(
            model, theta_operators, 
            test_input[i:i+1], test_target[i:i+1],
            max_operators=GREEDY_MAX_OPERATORS
        )
        
        result_dict = {
            'sample_idx': i,
            'composition': composition,
            'error': error
        }
        
        # Add equation-specific analysis if needed
        # Example: result_dict['custom_metric'] = calculate_custom_metric(pred, target)
        
        greedy_results.append(result_dict)
        print(f"Sample {i}: composition {composition}, error {error:.6f}")
    
    greedy_time = time.time() - start_time
    avg_greedy_error = sum(r['error'] for r in greedy_results) / len(greedy_results)
    results['methods']['greedy'] = {
        'avg_error': avg_greedy_error,
        'time': greedy_time,
        'details': greedy_results
    }
    
    # 3. Random selection
    print("\n3. Testing random operator selection...")
    random_results = []
    start_time = time.time()
    
    for i in range(args.num_samples):
        composition, error, pred = random_operator_selection(
            model, theta_operators,
            test_input[i:i+1], test_target[i:i+1],
            num_compositions=RANDOM_NUM_COMPOSITIONS,
            composition_lengths=RANDOM_COMPOSITION_LENGTHS
        )
        
        result_dict = {
            'sample_idx': i,
            'composition': composition,
            'error': error
        }
        
        random_results.append(result_dict)
        print(f"Sample {i}: composition {composition}, error {error:.6f}")
    
    random_time = time.time() - start_time
    avg_random_error = sum(r['error'] for r in random_results) / len(random_results)
    results['methods']['random'] = {
        'avg_error': avg_random_error,
        'time': random_time,
        'details': random_results
    }
    
    # 4. Gradient-based selection
    print("\n4. Testing gradient-based operator selection...")
    start_time = time.time()
    
    # Run gradient optimization on first sample
    theta_latents, grad_pred = gradient_selection_multi_operator(
        model, theta_operators,
        test_input[:1], test_target[:1],
        num_operators=GRADIENT_NUM_OPERATORS,
        epochs=GRADIENT_EPOCHS,
        lr=GRADIENT_LR,
        refinement_factor=GRADIENT_REFINEMENT,
        splitting_method=GRADIENT_SPLITTING
    )
    
    # Evaluate on all samples
    grad_results = []
    for i in range(args.num_samples):
        with torch.no_grad():
            from ttc_methods import multi_operator_splitting
            pred = multi_operator_splitting(
                model, theta_latents,
                test_input[i:i+1, -1],
                nt=test_target.shape[1],
                dt=4/250,
                refinement_factor=GRADIENT_REFINEMENT,
                splitting_method=GRADIENT_SPLITTING
            )
            from ttc_utils import get_relative_l2_error
            error = get_relative_l2_error()(pred, test_target[i:i+1]).item()
            
            result_dict = {
                'sample_idx': i,
                'error': error
            }
            
            grad_results.append(result_dict)
    
    grad_time = time.time() - start_time
    avg_grad_error = sum(r['error'] for r in grad_results) / len(grad_results)
    results['methods']['gradient'] = {
        'avg_error': avg_grad_error,
        'time': grad_time,
        'details': grad_results
    }
    
    # Summary
    print("\n" + "="*50)
    print(f"SUMMARY FOR {EQUATION_TYPE.upper()}:")
    print(f"Direct prediction: {direct_error:.6f} (time: {direct_time:.2f}s)")
    print(f"Greedy selection: {avg_greedy_error:.6f} (time: {greedy_time:.2f}s)")
    print(f"Random selection: {avg_random_error:.6f} (time: {random_time:.2f}s)")
    print(f"Gradient selection: {avg_grad_error:.6f} (time: {grad_time:.2f}s)")
    
    # Calculate relative improvements
    print("\nRelative improvements over direct prediction:")
    print(f"Greedy: {((direct_error - avg_greedy_error) / direct_error * 100):.1f}%")
    print(f"Random: {((direct_error - avg_random_error) / direct_error * 100):.1f}%")
    print(f"Gradient: {((direct_error - avg_grad_error) / direct_error * 100):.1f}%")
    print("="*50)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"{EQUATION_TYPE}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results(results, output_file)
    
    # Optional: Save predictions for visualization
    # torch.save({
    #     'direct_pred': direct_pred[:args.num_samples],
    #     'grad_pred': grad_pred,
    #     'test_input': test_input.cpu(),
    #     'test_target': test_target.cpu()
    # }, os.path.join(args.output_dir, f"{EQUATION_TYPE}_predictions.pt"))


if __name__ == "__main__":
    main()