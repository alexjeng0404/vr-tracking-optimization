# src/generate.py
import numpy as np

def gen_trajectory(T=200, kind='sin'):
    """
    Generate 3D trajectory

    Parameters:
        T: Number of time steps
        kind: Trajectory type, options 'sin', 'line', 'circle'
    Returns:
        x: (T, 3) array, each row is 3D coordinates
    """
    t = np.linspace(0, 4*np.pi, T)
    
    if kind == 'sin':
        x = np.vstack([np.sin(t), np.cos(t), 0.5*np.sin(2*t)]).T
    elif kind == 'line':
        x = np.vstack([t/np.max(t), 0.5*np.sin(t), 0.2*t]).T
    elif kind == 'circle':
        x = np.vstack([np.cos(t), np.sin(t), 0.1*t]).T
    else:
        raise ValueError("Unknown trajectory kind. Choose from 'sin', 'line', 'circle'.")
    
    return x.astype(np.float32)

def add_noise(x, sigma=0.02, drift_std=0.001, outlier_prob=0.01, outlier_scale=0.5):
    T = x.shape[0]
    
    noise = np.random.normal(scale=sigma, size=x.shape).astype(np.float32)
    drift = np.cumsum(np.random.normal(scale=drift_std, size=(T, 3)).astype(np.float32), axis=0)
    
    x_noisy = x + noise + drift
    
    mask = np.random.rand(T) < outlier_prob
    if np.sum(mask) > 0:
        outlier_noise = np.random.normal(scale=outlier_scale, size=(np.sum(mask), 3)).astype(np.float32)
        x_noisy[mask] += outlier_noise
    
    return x_noisy.astype(np.float32)