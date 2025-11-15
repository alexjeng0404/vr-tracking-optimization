# src/__init__.py
"""
VR Tracking Optimization Package

This package contains modules for generating synthetic 3D trajectories,
applying noise models, and implementing various smoothing algorithms.
"""

# version information
__version__ = "1.0.0"
__author__ = "Po Hung, Cheng"
__email__ = "alexjeng0404@gmail.com"

# Import key functions for easy access directly from the suite level
from .generate import gen_trajectory, add_noise
from .kalman import SimpleKF, run_kf
from .optimizer import optimize_smoothing, optimize_smoothing_advanced
from .utils import rmse, max_error

# define public API
__all__ = [
    # generate modules
    'gen_trajectory',
    'add_noise',
    
    # Kalman filter module
    'SimpleKF', 
    'run_kf',
    
    # optimization modules
    'optimize_smoothing',
    'optimize_smoothing_advanced',
    
    # tool function
    'rmse',
    'max_error',
]

# Package initialization message
print(f"Initializing VR Tracking Optimization {__version__}")

# Check necessary kits
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    print("✓ All dependencies loaded successfully")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Please install required packages: pip install numpy matplotlib torch")