# src/utils.py
import numpy as np

def rmse(a, b):
    """
    Caculate the RMSE(root mean square error) of the two trajectory

    Parameters:
        a: (T, 3) array, traj1
        b: (T, 3) array, traj2
    Returns:
        rmse_value: scaler
    """
    return np.sqrt(np.mean(np.sum((a - b)**2, axis=1)))

def max_error(a, b):
    """
    Caculate the maximum Euclidean distance error of the two trajectory

    Parameters:
        a: (T, 3) array, traj1
        b: (T, 3) array, traj2
    Returns:
        max_error: scaler

    """
    return np.max(np.sqrt(np.sum((a - b)**2, axis=1)))