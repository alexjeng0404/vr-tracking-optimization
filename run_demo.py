# run_demo.py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import src
import json
import time
import torch

def main():
    print("=== VR Tracking Optimization Demo (Fixed Version) ===")
    
    # Ensure the output folder exists
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Generate data
    print("1. Generate trajectory data...")
    x_clean = src.gen_trajectory(T=200, kind='circle')
    x_noisy = src.add_noise(x_clean, 
                       sigma=0.03, 
                       drift_std=0.0005, 
                       outlier_prob=0.02, 
                       outlier_scale=0.3)
    
    print(f"   - Clean trajectory: {x_clean.shape}")
    print(f"   - Noisy trajectory: {x_noisy.shape}")
    print(f"   - Baseline RMSE: {src.rmse(x_noisy, x_clean):.4f}")
    
    # 2. Perform Kalman filtering
    print("2. Perform Kalman filtering...")
    x_kf = src.run_kf(x_noisy, dt=1.0, pos_var = 1e-3, vel_var = 1e-3, meas_var=1e-2)
    print(f"   - Kalman Filter RMSE: {src.rmse(x_kf, x_clean):.4f}")
    
    # 3. Execute optimization smoothing - USE OPTIMAL PARAMETERS λ=1.0
    print("3. Execute optimization smoothing...")
    x_opt = src.optimize_smoothing(x_noisy, lam=1.0, lr=0.1, iters=500, verbose=True)
    print(f"   - Optimization RMSE: {src.rmse(x_opt, x_clean):.4f}")
    
    # 4. Execute advanced optimization smoothing - CONSERVATIVE PARAMETERS
    print("4. Execute advanced optimization smoothing...")
    x_opt_adv = src.optimize_smoothing_advanced(x_noisy, 
                                          lam1=1.0,    # Smaller first-order smoothing
                                          lam2=0.1,    # Smaller second-order smoothing
                                          lr=0.1, 
                                          iters=500, 
                                          use_huber=True,
                                          verbose=True)
    print(f"   - Advanced Optimization RMSE: {src.rmse(x_opt_adv, x_clean):.4f}")
    
    # 5. Calculate error metrics
    print("5. Calculate error metrics...")
    methods = {
        'Noisy': x_noisy,
        'Kalman': x_kf, 
        'Optimization': x_opt,
        'Optimization+': x_opt_adv
    }
    
    print("\n=== Error Comparison ===")
    print(f"{'Method':<15} | {'RMSE':<8} | {'Max Error':<10}")
    print("-" * 40)
    
    for name, trajectory in methods.items():
        rmse_val = src.rmse(trajectory, x_clean)
        max_err = src.max_error(trajectory, x_clean)
        print(f"{name:<15} | {rmse_val:.6f} | {max_err:.6f}")
    
    # 6. Save results
    save_results(x_clean, x_noisy, methods)
    
    # 7. Plot results
    print("7. Plot results...")
    plot_results(x_clean, x_noisy, x_kf, x_opt, x_opt_adv, methods)
    
    print("8. Complete! Check figures/ folder for results.")

def save_results(x_clean, x_noisy, methods):
    """Save results to results/ directory"""
    timestamp = int(time.time())
    
    # Save error metrics
    results_data = {
        'timestamp': timestamp,
        'parameters': {
            'trajectory_length': len(x_clean),
            'noise_sigma': 0.03,
            'drift_std': 0.0005,
            'outlier_prob': 0.02
        },
        'metrics': {}
    }
    
    for name, trajectory in methods.items():
        results_data['metrics'][name] = {
            'rmse': float(src.rmse(trajectory, x_clean)),
            'max_error': float(src.max_error(trajectory, x_clean))
        }
    
    # Save as JSON
    with open(f'results/results_{timestamp}.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save trajectory data
    np.savez(f'data/trajectories_{timestamp}.npz',
             clean=x_clean, noisy=x_noisy, **methods)
    
    print(f"✓ Results saved to results/results_{timestamp}.json")
    print(f"✓ Data saved to data/trajectories_{timestamp}.npz")

def plot_results(clean, noisy, kf, opt, opt_adv, methods):
    """Plot all result charts"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 3D trajectory comparison
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(clean[:, 0], clean[:, 1], clean[:, 2], 
             'b-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax1.plot(noisy[:, 0], noisy[:, 1], noisy[:, 2], 
             'r.', alpha=0.3, label='Noisy Measurements', markersize=2)
    ax1.plot(opt[:, 0], opt[:, 1], opt[:, 2], 
             'g--', linewidth=2, label='Optimization')
    ax1.plot(opt_adv[:, 0], opt_adv[:, 1], opt_adv[:, 2], 
             'm:', linewidth=2, label='Optimization+')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y') 
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    
    # 2. X coordinate time series
    ax2 = fig.add_subplot(234)
    time = np.arange(len(clean))
    ax2.plot(time, clean[:, 0], 'b-', linewidth=2, label='Ground Truth')
    ax2.plot(time, noisy[:, 0], 'r.', alpha=0.4, label='Noisy', markersize=2)
    ax2.plot(time, kf[:, 0], 'orange', linewidth=1.5, label='Kalman')
    ax2.plot(time, opt[:, 0], 'g--', linewidth=1.5, label='Optimization')
    ax2.plot(time, opt_adv[:, 0], 'm:', linewidth=1.5, label='Optimization+')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('X Position')
    ax2.set_title('X Coordinate Over Time')
    ax2.legend()
    
    # 3. Y coordinate time series
    ax3 = fig.add_subplot(235)
    ax3.plot(time, clean[:, 1], 'b-', linewidth=2, label='Ground Truth')
    ax3.plot(time, noisy[:, 1], 'r.', alpha=0.4, label='Noisy', markersize=2)
    ax3.plot(time, kf[:, 1], 'orange', linewidth=1.5, label='Kalman')
    ax3.plot(time, opt[:, 1], 'g--', linewidth=1.5, label='Optimization')
    ax3.plot(time, opt_adv[:, 1], 'm:', linewidth=1.5, label='Optimization+')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Y Position')
    ax3.set_title('Y Coordinate Over Time')
    ax3.legend()
    
    # 4. Z coordinate time series
    ax4 = fig.add_subplot(236)
    ax4.plot(time, clean[:, 2], 'b-', linewidth=2, label='Ground Truth')
    ax4.plot(time, noisy[:, 2], 'r.', alpha=0.4, label='Noisy', markersize=2)
    ax4.plot(time, kf[:, 2], 'orange', linewidth=1.5, label='Kalman')
    ax4.plot(time, opt[:, 2], 'g--', linewidth=1.5, label='Optimization')
    ax4.plot(time, opt_adv[:, 2], 'm:', linewidth=1.5, label='Optimization+')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Z Position')
    ax4.set_title('Z Coordinate Over Time')
    ax4.legend()
    
    # 5. Error over time
    ax5 = fig.add_subplot(232)
    for name, trajectory in methods.items():
        error = np.sqrt(np.sum((trajectory - clean) ** 2, axis=1))
        if name == 'Noisy':
            ax5.plot(time, error, 'r-', alpha=0.6, label=name, linewidth=1)
        elif name == 'Kalman':
            ax5.plot(time, error, 'orange', label=name, linewidth=1.5)
        elif name == 'Optimization':
            ax5.plot(time, error, 'g-', linewidth=2, label=name)
        elif name == 'Optimization+':
            ax5.plot(time, error, 'm-', linewidth=2, label=name)
        else:
            ax5.plot(time, error, 'gray', alpha=0.7, label=name)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Euclidean Error')
    ax5.set_title('Error Over Time')
    ax5.legend()
    
    # 6. RMSE comparison bar chart
    ax6 = fig.add_subplot(233)
    method_names = list(methods.keys())
    rmse_values = [src.rmse(methods[name], clean) for name in method_names]
    
    colors = ['red', 'orange', 'green', 'purple']
    bars = ax6.bar(method_names, rmse_values, color=colors, alpha=0.7)
    
    # Display values on bars
    for bar, value in zip(bars, rmse_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    ax6.set_ylabel('RMSE')
    ax6.set_title('RMSE Comparison by Method')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/final_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()