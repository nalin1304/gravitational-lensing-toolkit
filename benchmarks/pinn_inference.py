"""
PINN Inference Speed Benchmark

Measures PINN model performance across different:
- Batch sizes: [1, 4, 8, 16, 32]
- Input sizes: 64x64, 128x128, 256x256
- Devices: CPU vs GPU (if available)

Outputs results to benchmarks/pinn_results.json
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import torch
import psutil
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.pinn import PhysicsInformedNN


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_forward_pass(
    model: PhysicsInformedNN,
    batch_size: int,
    input_size: int,
    device: str,
    num_warmup: int = 5,
    num_iterations: int = 50
) -> Dict:
    """
    Benchmark forward pass performance.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        Model to benchmark
    batch_size : int
        Batch size to test
    input_size : int
        Input image size (input_size x input_size)
    device : str
        'cpu' or 'cuda'
    num_warmup : int
        Number of warmup iterations
    num_iterations : int
        Number of benchmark iterations
        
    Returns
    -------
    results : dict
        Benchmark results including timing and memory
    """
    model.eval()
    model.to(device)
    
    # Create random input
    input_tensor = torch.randn(batch_size, 1, input_size, input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark
    mem_before = get_memory_usage()
    
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            output_params, output_logits = model(input_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    mem_after = get_memory_usage()
    mem_delta = mem_after - mem_before
    
    # Compute statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Images per second
    images_per_sec = batch_size / mean_time
    
    # Parameters per second (5 parameters per image)
    params_per_sec = images_per_sec * 5
    
    results = {
        'batch_size': batch_size,
        'input_size': input_size,
        'device': device,
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'images_per_sec': images_per_sec,
        'params_per_sec': params_per_sec,
        'memory_delta_mb': mem_delta,
        'num_iterations': num_iterations
    }
    
    return results


def run_comprehensive_benchmark() -> Dict:
    """
    Run comprehensive PINN inference benchmark.
    
    Returns
    -------
    results : dict
        Complete benchmark results
    """
    print("="*80)
    print("PINN Inference Speed Benchmark")
    print("="*80)
    
    # Check device availability
    device_cpu = 'cpu'
    device_gpu = 'cuda' if torch.cuda.is_available() else None
    
    if device_gpu:
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPU Not Available - CPU only benchmark")
    
    print()
    
    # Test configurations
    batch_sizes = [1, 4, 8, 16, 32]
    input_sizes = [64, 128, 256]
    devices = [device_cpu] + ([device_gpu] if device_gpu else [])
    
    # Initialize model
    print("Initializing PhysicsInformedNN model...")
    model = PhysicsInformedNN(
        input_size=64,
        dropout_rate=0.2
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Run benchmarks
    all_results = []
    total_tests = len(batch_sizes) * len(input_sizes) * len(devices)
    current_test = 0
    
    for device in devices:
        print(f"\n{'='*80}")
        print(f"Device: {device.upper()}")
        print(f"{'='*80}\n")
        
        for input_size in input_sizes:
            print(f"  Input Size: {input_size}x{input_size}")
            print(f"  {'-'*60}")
            
            for batch_size in batch_sizes:
                current_test += 1
                print(f"    [{current_test}/{total_tests}] Batch size: {batch_size:2d} ... ", end='', flush=True)
                
                try:
                    result = benchmark_forward_pass(
                        model=model,
                        batch_size=batch_size,
                        input_size=input_size,
                        device=device,
                        num_warmup=5,
                        num_iterations=50
                    )
                    
                    all_results.append(result)
                    
                    # Print result
                    print(f"{result['mean_time_ms']:6.2f} ms  "
                          f"({result['images_per_sec']:6.1f} img/s, "
                          f"{result['params_per_sec']:7.1f} params/s)")
                    
                except Exception as e:
                    print(f"FAILED: {str(e)}")
            
            print()
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # CPU results
    cpu_results = [r for r in all_results if r['device'] == 'cpu']
    if cpu_results:
        cpu_best = max(cpu_results, key=lambda x: x['images_per_sec'])
        print(f"\nCPU Best Performance:")
        print(f"  Configuration: batch_size={cpu_best['batch_size']}, "
              f"input_size={cpu_best['input_size']}x{cpu_best['input_size']}")
        print(f"  Speed: {cpu_best['images_per_sec']:.1f} images/sec")
        print(f"  Latency: {cpu_best['mean_time_ms']:.2f} ms")
        
        target_cpu = 1.0  # 1 image/sec target
        cpu_meets_target = cpu_best['images_per_sec'] >= target_cpu
        print(f"  Target (>1 img/s): {'✓ PASS' if cpu_meets_target else '✗ FAIL'}")
    
    # GPU results
    if device_gpu:
        gpu_results = [r for r in all_results if r['device'] == 'cuda']
        if gpu_results:
            gpu_best = max(gpu_results, key=lambda x: x['images_per_sec'])
            print(f"\nGPU Best Performance:")
            print(f"  Configuration: batch_size={gpu_best['batch_size']}, "
                  f"input_size={gpu_best['input_size']}x{gpu_best['input_size']}")
            print(f"  Speed: {gpu_best['images_per_sec']:.1f} images/sec")
            print(f"  Latency: {gpu_best['mean_time_ms']:.2f} ms")
            
            target_gpu = 10.0  # 10 images/sec target
            gpu_meets_target = gpu_best['images_per_sec'] >= target_gpu
            print(f"  Target (>10 img/s): {'✓ PASS' if gpu_meets_target else '✗ FAIL'}")
            
            # Speedup
            if cpu_results:
                # Compare same configuration
                cpu_same = next((r for r in cpu_results 
                               if r['batch_size'] == gpu_best['batch_size'] 
                               and r['input_size'] == gpu_best['input_size']), None)
                if cpu_same:
                    speedup = gpu_best['images_per_sec'] / cpu_same['images_per_sec']
                    print(f"  Speedup vs CPU: {speedup:.1f}x")
    
    # Prepare output
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'PhysicsInformedNN',
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'results': all_results
    }
    
    return output


def save_results(results: Dict, output_path: Path):
    """Save benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main benchmark execution."""
    try:
        # Run benchmark
        results = run_comprehensive_benchmark()
        
        # Save results
        output_path = Path(__file__).parent / 'pinn_results.json'
        save_results(results, output_path)
        
        print("\n" + "="*80)
        print("Benchmark Complete!")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
