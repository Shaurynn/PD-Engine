import torch
import torch.nn as nn
from typing import List

# =============================================================================
# 1. Conditional Pareto Hypernetwork
# =============================================================================

class ConditionalParetoHypernetwork(nn.Module):
    """
    Generates a continuous Pareto-optimal frontier by conditioning the policy 
    generation on preference weights (w).
    
    Instead of collapsing multi-objective evaluations into a single scalar policy 
    (which violates incommensurability), this hypernetwork maps the state and a 
    sampled preference vector to a specific non-dominated policy on the frontier.
    """
    def __init__(self, belief_dim: int, num_objectives: int, act_dim: int):
        super().__init__()
        # The network ingests both the non-Markovian belief state and the preference weights
        self.net = nn.Sequential(
            nn.Linear(belief_dim + num_objectives, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Softmax(dim=-1) # Outputs a valid probability distribution over discrete actions
        )

    def forward(self, belief: torch.Tensor, pref_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            belief (torch.Tensor): The POMDP belief state [batch_size, belief_dim]
            pref_weights (torch.Tensor): The sampled preference vector from the K-simplex [batch_size, num_objectives]
            
        Returns:
            torch.Tensor: The conditional policy distribution [batch_size, act_dim]
        """
        # Concatenate the belief state with the preference weights to guide the Pareto exploration
        x = torch.cat([belief, pref_weights], dim=-1)
        return self.net(x)

# =============================================================================
# 2. Multiple Gradient Descent Algorithm (MGDA) via Frank-Wolfe
# =============================================================================

def frank_wolfe_mgda(gradients: List[torch.Tensor], max_iters: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """
    Frank-Wolfe algorithm to find the minimum-norm element in the convex hull 
    of K gradients. 
    
    This solver explicitly prevents mathematical commensuration (linear scalarization). 
    It computes a shared update direction that monotonically improves all incommensurable 
    patches simultaneously, supporting an arbitrary number of objectives (K > 2).
    
    Args:
        gradients (List[torch.Tensor]): A list of K gradient tensors corresponding to each Presheaf patch.
        max_iters (int): Maximum iterations for the Frank-Wolfe optimization.
        tol (float): Tolerance threshold for early stopping.
        
    Returns:
        torch.Tensor: The shared, Pareto-stationary gradient update direction.
    """
    K = len(gradients)
    
    # Flatten all gradients to 1D vectors and stack them into a matrix of shape [K, num_parameters]
    grads_flat = torch.stack([g.flatten() for g in gradients])
    
    # Initialize the convex combination weights (alpha) uniformly across all K objectives
    alpha = torch.ones(K, device=grads_flat.device) / K
    
    for _ in range(max_iters):
        # Compute the current shared gradient vector based on the alpha weights
        shared_grad = torch.matmul(alpha, grads_flat)
        
        # Frank-Wolfe Oracle: 
        # Find the specific gradient that minimizes the dot product with the current shared_grad.
        # This identifies the direction of steepest descent within the convex hull.
        dot_products = torch.matmul(grads_flat, shared_grad)
        min_idx = torch.argmin(dot_products)
        
        # Create a target alpha vector placing all weight on the steepest descent direction
        target_alpha = torch.zeros(K, device=grads_flat.device)
        target_alpha[min_idx] = 1.0
        
        # Exact Line Search for the optimal step size (gamma)
        diff_grad = grads_flat[min_idx] - shared_grad
        
        # gamma = -(shared_grad^T * diff_grad) / ||diff_grad||^2
        gamma = -torch.dot(shared_grad, diff_grad) / (torch.dot(diff_grad, diff_grad) + 1e-10)
        
        # Constrain gamma to [0, 1] to ensure we remain inside the convex hull
        gamma = torch.clamp(gamma, 0.0, 1.0)
        
        # Early stopping if the step size falls below the tolerance threshold
        if gamma < tol:
            break
            
        # Update the alpha weights
        alpha = (1 - gamma) * alpha + gamma * target_alpha

    # Return the final computed minimum-norm gradient direction
    return torch.matmul(alpha, grads_flat)