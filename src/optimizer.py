# src/optimizer.py
import torch
import numpy as np

def optimize_smoothing(y_np, lam=1.0, lr=0.1, iters=500, verbose=False):
    """
    Smoothing optimization using gradient descent (Adam)

    Energy function: 
        E = Σ_t ||x_t - y_t||^2 + λ * Σ_t ||x_t - x_{t-1}||^2
    Parameters:
        y_np: (T, 3) Observation trajectory (numpy array)
        lam: Weights of the smoothing term
        lr: Learning rate
        iters: Number of iterations
    Returns: 
        x_opt: (T, 3) Smoothed trajectory (numpy array)
    """
    if y_np.dtype != np.float32:
        y_np = y_np.astype(np.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
        print(f"Input data: shape={y_np.shape}, range=[{y_np.min():.3f}, {y_np.max():.3f}]")
    
    # Convert to PyTorch tensor - use torch.tensor for proper gradient computation
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    T, D = y.shape
    
    # Initialize with small noise to ensure optimization has something to learn
    initial_guess = y.clone() + torch.randn_like(y) * 0.01
    hat = torch.nn.Parameter(initial_guess)
    
    # Optimizer
    optimizer = torch.optim.Adam([hat], lr=lr)
    
    losses = []
    for i in range(iters):
        optimizer.zero_grad()
        
        # Data term - use mean instead of sum to avoid large values
        data_loss = torch.mean((hat - y) ** 2)
        
        # Smooth term (first difference)
        diff = hat[1:] - hat[:-1]
        smooth_loss = torch.mean(diff ** 2)
        
        # Total loss
        loss = data_loss + lam * smooth_loss
        losses.append(loss.item())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if verbose and (i + 1) % 100 == 0:
            grad_norm = torch.norm(hat.grad).item() if hat.grad is not None else 0.0
            print(f"Iteration {i+1}/{iters}, Loss: {loss.item():.6f}, GradNorm: {grad_norm:.6f}")
    
    # Return numpy array on CPU
    result = hat.detach().cpu().numpy()
    if verbose:
        print(f"Optimization completed. Final loss: {losses[-1]:.6f}")
    return result

def optimize_smoothing_advanced(y_np, lam1=1.0, lam2=0.1, lr=0.1, iters=500, use_huber=False, verbose=False):
    """
    Advanced version of smoothing optimization, including second-order differencing and robust loss.

    Parameters:
        y_np: (T, 3) Observation trajectory
        lam1: First-order smoothing weights
        lam2: Second-order smoothing weights
        lr: Learning rate
        iters: Number of iterations
        use_huber: Whether to use Huber loss to combat outliers
    Returns: 
        x_opt: (T, 3) Smoothed trajectory
    """
    if y_np.dtype != np.float32:
        y_np = y_np.astype(np.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
        print(f"Input data: shape={y_np.shape}, range=[{y_np.min():.3f}, {y_np.max():.3f}]")
    
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    T, D = y.shape
    
    # Initialize with small noise
    initial_guess = y.clone() + torch.randn_like(y) * 0.01
    hat = torch.nn.Parameter(initial_guess)
    
    optimizer = torch.optim.Adam([hat], lr=lr)
    
    losses = []
    for i in range(iters):
        optimizer.zero_grad()
        
        # Data term
        if use_huber:
            data_loss = torch.nn.functional.huber_loss(hat, y, reduction='mean', delta=1.0)
        else:
            data_loss = torch.mean((hat - y) ** 2)
        
        # First-order smoothing term
        diff1 = hat[1:] - hat[:-1]
        smooth_loss1 = torch.mean(diff1 ** 2)
        
        # Second-order smoothing term
        if T >= 3:
            diff2 = hat[2:] - 2 * hat[1:-1] + hat[:-2]
            smooth_loss2 = torch.mean(diff2 ** 2)
        else:
            smooth_loss2 = torch.tensor(0.0, device=device)
        
        loss = data_loss + lam1 * smooth_loss1 + lam2 * smooth_loss2
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if verbose and (i + 1) % 100 == 0:
            grad_norm = torch.norm(hat.grad).item() if hat.grad is not None else 0.0
            print(f"Iteration {i+1}/{iters}, Loss: {loss.item():.6f}, GradNorm: {grad_norm:.6f}")
    
    result = hat.detach().cpu().numpy()
    if verbose:
        print(f"Advanced optimization completed. Final loss: {losses[-1]:.6f}")
    return result

# Test function
if __name__ == "__main__":
    print("=== Optimizer Test ===")
    
    # Create test data
    test_data = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    print(f"Input data type: {test_data.dtype}")
    print(f"Input data shape: {test_data.shape}")
    
    # Basic version
    result = optimize_smoothing(test_data, lam=1.0, lr=0.1, iters=50, verbose=True)
    print(f"Output data type: {result.dtype}")
    print(f"Output data shape: {result.shape}")
    print(f"Input: {test_data}")
    print(f"Output: {result}")
    
    # Advanced version
    result_adv = optimize_smoothing_advanced(test_data, lam1=1.0, lam2=0.1, iters=50, verbose=True)
    print(f"Advanced version output: {result_adv}")