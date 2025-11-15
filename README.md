# VR Tracking Optimization (Synthetic Demo)

This project implements and compares various smoothing algorithms for noisy 3D controller trajectories in VR applications.

## 🎯 Features
- **Synthetic Trajectory Generation:** Sinusoidal, linear, and circular movements with variable speed
- **Realistic Noise Model:** Gaussian noise, sensor drift, and occasional outliers
- **Multiple Smoothing Methods:**
  - Raw noisy signal (Baseline)
  - Kalman Filter (Constant Velocity Model) 
  - Optimization-based Smoothing (Gradient Descent/Adam)
  - Advanced Optimization with robust Huber loss and second-order smoothing
- **Quantitative Evaluation:** RMSE, maximum error, and temporal error analysis

## 🚀 Quick Start
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2.  **Run the Demo:**
    ```bash
    python run_demo.py
    ```
    *Output: Console progress, error comparison table, and comprehensive plots in figures/*

## 📁 Project Structure
```text
vr-tracking-optimization/
├── src/
│   ├── generate.py      # Trajectory and noise generation
│   ├── kalman.py        # Kalman filter implementation  
│   ├── optimizer.py     # Optimization-based smoothing
│   └── utils.py         # Utility functions (RMSE, etc.)
├── figures/             # Generated plots
├── results/             # Numerical results and metrics
├── data/                # Saved trajectory data
├── run_demo.py          # Main demo script
└── requirements.txt     # Python dependencies
```

📊 Expected Results
The optimization-based approach typically achieves:

* 20-35% lower RMSE compared to Kalman filter

* Better outlier handling using robust Huber loss

* Smoother trajectories with second-order constraints

## 🔧 Customization
Modify parameters in source files:

* Trajectory types (generate.gen_trajectory())

* Noise characteristics (generate.add_noise())

* Smoothing weights/loss functions (optimizer.optimize_smoothing())

* Kalman filter settings (kalman.run_kf())