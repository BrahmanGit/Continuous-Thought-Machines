# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 16:40:41 2026
Modified to measure inference times under different RAM constraints
Cross-platform version (Windows/Linux/macOS)
@author: joni_
"""
import numpy as np
import time
import models_jit
import NN_utils
import torch
import csv
from datetime import datetime
import sys
import gc
import os
import platform
import psutil

# Try different memory limiting approaches based on platform
PLATFORM = platform.system()
print(f"Detected platform: {PLATFORM}")

def set_single_core_affinity():
    """Set process to use only a single CPU core"""
    try:
        p = psutil.Process()
        # Set affinity to only the first CPU core
        p.cpu_affinity([0])
        print(f"Process affinity set to CPU core 0")
        print(f"Available cores after affinity: {p.cpu_affinity()}")
        return True
    except Exception as e:
        print(f"Warning: Could not set CPU affinity: {e}")
        return False

def set_thread_limits():
    """Limit number of threads for various libraries"""
    # NumPy/OpenBLAS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # PyTorch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    print("Thread limits set to 1 for all libraries")

if PLATFORM == "Windows":
    try:
        import win32api
        import win32job
        HAS_WIN32 = True
        print("Windows job object support available")
    except ImportError:
        HAS_WIN32 = False
        print("Warning: pywin32 not installed. Install with: pip install pywin32")
        print("Running without memory limits...")
else:
    try:
        import resource
        HAS_RESOURCE = True
        print("Unix resource module available")
    except ImportError:
        HAS_RESOURCE = False
        print("Warning: resource module not available")

# Set single core affinity and thread limits at startup
print("\nConfiguring single-core execution...")
set_thread_limits()
affinity_set = set_single_core_affinity()

class MemoryLimiter:
    """Cross-platform memory limiter"""
    
    def __init__(self):
        self.job_handle = None
        self.platform = PLATFORM
        
    def set_limit(self, limit_mb):
        """Set memory limit in MB"""
        limit_bytes = limit_mb * 1024 * 1024
        
        if self.platform == "Windows" and HAS_WIN32:
            return self._set_limit_windows(limit_bytes)
        elif self.platform in ["Linux", "Darwin"] and HAS_RESOURCE:
            return self._set_limit_unix(limit_bytes)
        else:
            print(f"  [Memory limiting not available on this platform]")
            return False
    
    def _set_limit_windows(self, limit_bytes):
        """Windows-specific memory limiting using job objects"""
        try:
            # Create a job object
            self.job_handle = win32job.CreateJobObject(None, "")
            
            # Set memory limit
            job_info = win32job.QueryInformationJobObject(
                self.job_handle, 
                win32job.JobObjectExtendedLimitInformation
            )
            job_info['ProcessMemoryLimit'] = limit_bytes
            job_info['BasicLimitInformation']['LimitFlags'] = (
                win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            )
            
            win32job.SetInformationJobObject(
                self.job_handle,
                win32job.JobObjectExtendedLimitInformation,
                job_info
            )
            
            # Assign current process to job
            process_handle = win32api.GetCurrentProcess()
            win32job.AssignProcessToJobObject(self.job_handle, process_handle)
            
            print(f"  Windows memory limit set to {limit_bytes // (1024*1024)} MB")
            return True
            
        except Exception as e:
            print(f"  Warning: Could not set Windows memory limit: {e}")
            return False
    
    def _set_limit_unix(self, limit_bytes):
        """Unix-specific memory limiting"""
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            print(f"  Unix memory limit set to {limit_bytes // (1024*1024)} MB")
            return True
        except Exception as e:
            print(f"  Warning: Could not set Unix memory limit: {e}")
            return False
    
    def clear_limit(self):
        """Remove memory limit"""
        if self.platform == "Windows" and self.job_handle:
            try:
                import win32api
                win32api.CloseHandle(self.job_handle)
                self.job_handle = None
            except:
                pass
        elif self.platform in ["Linux", "Darwin"] and HAS_RESOURCE:
            try:
                import resource
                resource.setrlimit(
                    resource.RLIMIT_AS, 
                    (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
                )
            except:
                pass

def create_dummy_dataset(
    n_classes=9,
    n_features=15,
    samples_per_class=1,
    class_std=0.5,
    random_state=None
):
    rng = np.random.default_rng(random_state)
    X = []
    y = []
    centroids = rng.uniform(-5, 5, size=(n_classes, n_features))
    for class_idx in range(n_classes):
        samples = (
            centroids[class_idx]
            + rng.normal(0, class_std, size=(samples_per_class, n_features))
        )
        X.append(samples)
        y.append(np.full(samples_per_class, class_idx))
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y

def run_gravity_model_benchmark(Xf, y, obs, dt, integration_window_length, 
                                 noise_KF, noise_obs, beta, n_iterations=100):
    """Run gravity model benchmark and return timing results."""
    try:
        grav_model = models_jit.create_dirichlet_model(
            X=Xf,
            y=y,
            noise_process=noise_KF,
            noise_obs=noise_obs,
            x0=obs[0],
            dt=dt,
            integration_window_length=integration_window_length,
            gravitational_const=beta
        )
        
        times = []
        for i in range(n_iterations):
            for observation in obs:
                start = time.time()
                mean, var, ev = grav_model.predict_step(observation)
                end = time.time()
            if i > 0:
                times.append(end - start)
        
        return {
            'success': True,
            'times': times,
            'mean': np.mean(times) * 1000,
            'std': np.std(times) * 1000,
            'min': np.min(times) * 1000,
            'max': np.max(times) * 1000
        }
    except MemoryError as e:
        return {'success': False, 'error': 'MemoryError', 'message': str(e)}
    except Exception as e:
        return {'success': False, 'error': type(e).__name__, 'message': str(e)}

def run_cnn_benchmark(input_dim, hidden_dim, fc_dim, output_dim, 
                     integration_window_length, n_iterations=100):
    """Run CNN benchmark and return timing results."""
    try:
        model = NN_utils.CNNNetwork(input_dim, hidden_dim, fc_dim, output_dim)
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        dummy_input = torch.randn(1, integration_window_length, input_dim).to(device)
        dummy_lengths = torch.tensor([integration_window_length], dtype=torch.long).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_input, dummy_lengths)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timing
        cnn_times = []
        with torch.no_grad():
            for i in range(n_iterations):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                output = model(dummy_input, dummy_lengths)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.time()
                cnn_times.append(end - start)
        
        return {
            'success': True,
            'times': cnn_times,
            'mean': np.mean(cnn_times) * 1000,
            'std': np.std(cnn_times) * 1000,
            'min': np.min(cnn_times) * 1000,
            'max': np.max(cnn_times) * 1000,
            'device': str(device)
        }
    except MemoryError as e:
        return {'success': False, 'error': 'MemoryError', 'message': str(e)}
    except Exception as e:
        return {'success': False, 'error': type(e).__name__, 'message': str(e)}

# =============================================================================
# Main Execution
# =============================================================================

print("=" * 80)
print("RAM-Constrained Single-Core Inference Time Benchmark")
print("=" * 80)
print(f"CPU Affinity Set: {affinity_set}")
print(f"PyTorch Threads: {torch.get_num_threads()}")
print(f"PyTorch Interop Threads: {torch.get_num_interop_threads()}")

# Configuration
RAM_LIMITS = [2, 4, 8, 16, 32, 64, 128]  # MB
N_ITERATIONS = 100

# Dataset parameters
Xf, y = create_dummy_dataset(
    n_classes=9,
    n_features=15,
    samples_per_class=200,
    random_state=42
)

# Model parameters
dt = 0.015
integration_window_length = 20
noise_KF, noise_obs = 10e-6, 10e-6
beta = 0.15
obs = Xf[1:2]
x0 = Xf[0]

input_dim = 15
hidden_dim = 32
fc_dim = 16
output_dim = 9

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create memory limiter
limiter = MemoryLimiter()

# Storage for all results
all_results = []

# Run benchmarks for each RAM limit
for ram_limit_mb in RAM_LIMITS:
    print(f"\n{'=' * 80}")
    print(f"Testing with RAM limit: {ram_limit_mb} MB")
    print(f"{'=' * 80}")
    
    # Clear any previous limits and force garbage collection
    limiter.clear_limit()
    gc.collect()
    
    # Set new memory limit
    limit_set = limiter.set_limit(ram_limit_mb)
    
    result_entry = {
        'ram_limit_mb': ram_limit_mb,
        'limit_enforced': limit_set,
        'platform': PLATFORM
    }
    
    # Test Gravity Model
    print(f"\n--- Gravity Model ---")
    grav_result = run_gravity_model_benchmark(
        Xf, y, obs, dt, integration_window_length,
        noise_KF, noise_obs, beta, N_ITERATIONS
    )
    
    if grav_result['success']:
        print(f"  Mean: {grav_result['mean']:.4f} ms")
        print(f"  Std:  {grav_result['std']:.4f} ms")
        print(f"  Min:  {grav_result['min']:.4f} ms")
        print(f"  Max:  {grav_result['max']:.4f} ms")
        result_entry['grav_success'] = True
        result_entry['grav_mean_ms'] = grav_result['mean']
        result_entry['grav_std_ms'] = grav_result['std']
        result_entry['grav_min_ms'] = grav_result['min']
        result_entry['grav_max_ms'] = grav_result['max']
    else:
        print(f"  FAILED: {grav_result['error']} - {grav_result['message']}")
        result_entry['grav_success'] = False
        result_entry['grav_error'] = grav_result['error']
    
    # Clear memory before CNN test
    gc.collect()
    
    # Test CNN
    print(f"\n--- CNN ---")
    cnn_result = run_cnn_benchmark(
        input_dim, hidden_dim, fc_dim, output_dim,
        integration_window_length, N_ITERATIONS
    )
    
    if cnn_result['success']:
        print(f"  Device: {cnn_result['device']}")
        print(f"  Mean: {cnn_result['mean']:.4f} ms")
        print(f"  Std:  {cnn_result['std']:.4f} ms")
        print(f"  Min:  {cnn_result['min']:.4f} ms")
        print(f"  Max:  {cnn_result['max']:.4f} ms")
        result_entry['cnn_success'] = True
        result_entry['cnn_mean_ms'] = cnn_result['mean']
        result_entry['cnn_std_ms'] = cnn_result['std']
        result_entry['cnn_min_ms'] = cnn_result['min']
        result_entry['cnn_max_ms'] = cnn_result['max']
        result_entry['cnn_device'] = cnn_result['device']
    else:
        print(f"  FAILED: {cnn_result['error']} - {cnn_result['message']}")
        result_entry['cnn_success'] = False
        result_entry['cnn_error'] = cnn_result['error']
    
    all_results.append(result_entry)
    
    # Clear limit for next iteration
    limiter.clear_limit()
    gc.collect()

# =============================================================================
# Save Results to CSV
# =============================================================================

print(f"\n{'=' * 80}")
print("Saving results...")
print(f"{'=' * 80}")

# Detailed results CSV
detailed_csv = f"ram_sweep_detailed_{timestamp}.csv"
with open(detailed_csv, 'w', newline='') as csvfile:
    fieldnames = [
        'ram_limit_mb', 'limit_enforced', 'platform',
        'grav_success', 'grav_mean_ms', 'grav_std_ms', 'grav_min_ms', 'grav_max_ms', 'grav_error',
        'cnn_success', 'cnn_mean_ms', 'cnn_std_ms', 'cnn_min_ms', 'cnn_max_ms', 'cnn_device', 'cnn_error'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in all_results:
        row = {field: result.get(field, '') for field in fieldnames}
        writer.writerow(row)

print(f"Detailed results saved to: {detailed_csv}")

# Summary CSV (only successful runs)
summary_csv = f"ram_sweep_summary_{timestamp}.csv"
with open(summary_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ram_limit_mb', 'model', 'mean_ms', 'std_ms', 'min_ms', 'max_ms'])
    
    for result in all_results:
        ram_limit = result['ram_limit_mb']
        
        if result.get('grav_success'):
            writer.writerow([
                ram_limit, 'Gravity Model',
                result['grav_mean_ms'], result['grav_std_ms'],
                result['grav_min_ms'], result['grav_max_ms']
            ])
        
        if result.get('cnn_success'):
            writer.writerow([
                ram_limit, 'CNN',
                result['cnn_mean_ms'], result['cnn_std_ms'],
                result['cnn_min_ms'], result['cnn_max_ms']
            ])

print(f"Summary results saved to: {summary_csv}")

print(f"\n{'=' * 80}")
print("Benchmark completed successfully!")
print(f"{'=' * 80}")
print(f"\nNote: Memory limits {'were' if any(r['limit_enforced'] for r in all_results) else 'were NOT'} enforced")
print(f"Platform: {PLATFORM}")
print(f"CPU Affinity: {affinity_set}")
print(f"Thread Count: 1")