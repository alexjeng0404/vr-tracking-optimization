# src/kalman.py
import numpy as np

class SimpleKF:
    """
    Simple 3D constant speed Kalman filter

    State: [x, y, z, vx, vy, vz]
    """
    def __init__(self, dt=1.0, pos_var = 1e-4, vel_var = 1e-4, meas_var=1e-3):
        # time step
        self.dt = dt
        
        # state transition matrix F
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0], 
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # observation matrix H (can only observe position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # process noise covariance Q
        self.Q = np.diag([pos_var, pos_var, pos_var, 
                  vel_var, vel_var, vel_var])
        
        # observation noise covariance R  
        self.R = meas_var * np.eye(3)
        
        # state covariance P
        self.P = np.eye(6)
        
        # state vector [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)

    def init_state(self, pos):
        """
        Initialize state with initial position

        Parameters:
            pos: [x, y, z] initial position
        """
        self.x[:3] = pos
        self.x[3:] = 0.0  # The initial speed is set to 0

    def step(self, measurement):
        """
        Perform a Kalman filtering step

        Parameters:
            measurement: [x, y, z] observation value
        Returns:
            estimated_pos: [x, y, z] estimated position
        """
        # prediction step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # update step
        y = measurement - self.H @ self.x  # measurement residuals
        S = self.H @ self.P @ self.H.T + self.R  # remaining difference
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        return self.x[:3].copy()

def run_kf(measurements, dt=1.0, pos_var = 1e-4, vel_var = 1e-4, meas_var=1e-3):
    """
    Perform Kalman filtering on the entire trajectory

    Parameters:
        measurements: (T, 3) observation trajectory
        dt: time step
        process_var: process noise variance
        meas_var: observation noise variance
    Returns:
        smoothed: (T, 3) smoothed trajectory
    """
    kf = SimpleKF(dt=dt, pos_var = 1e-4, vel_var = 1e-4, meas_var=meas_var)
    T = measurements.shape[0]
    smoothed = np.zeros_like(measurements)
    
    # Initialize with first observation
    kf.init_state(measurements[0])
    smoothed[0] = measurements[0]
    
    # Perform filtering at each time step
    for t in range(1, T):
        smoothed[t] = kf.step(measurements[t])
    
    return smoothed

# test funciton
if __name__ == "__main__":
    # simple test
    print("Kalman Filter test")
    test_meas = np.array([[1, 2, 3], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
    result = run_kf(test_meas)
    print(f"Input: {test_meas}")
    print(f"Output: {result}")