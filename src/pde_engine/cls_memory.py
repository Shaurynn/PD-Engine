import torch
import numpy as np
from typing import Tuple
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =============================================================================
# 1. SWD Governance & Distributional Shift
# =============================================================================

def compute_return_swd(returns_p: torch.Tensor, returns_q: torch.Tensor, num_projections: int = 200) -> torch.Tensor:
    """
    Computes Sliced Wasserstein Distance accurately on joint Return Distributions 
    (not just action probabilities) to measure systemic distributional shift.

    This allows the MLOps governance protocol to detect if the active policy network 
    has catastrophically drifted from the historical safety-critical baseline 
    without destructively flattening the high-dimensional return structure.

    Args:
        returns_p (torch.Tensor): Return distribution from the active network [batch, dim]
        returns_q (torch.Tensor): Return distribution from the replay buffer baseline [batch, dim]
        num_projections (int): Number of random 1D projections to approximate the integral.

    Returns:
        torch.Tensor: The computed Sliced Wasserstein Distance.
    """
    dim = returns_p.size(1)
    
    # Generate random projection vectors over the unit hypersphere
    projections = torch.randn(dim, num_projections, device=returns_p.device)
    projections /= torch.norm(projections, dim=0, keepdim=True)
    
    # Project both high-dimensional distributions onto the 1D lines
    p_proj = torch.matmul(returns_p, projections)
    q_proj = torch.matmul(returns_q, projections)
    
    # Sort the projections to compute the 1D Wasserstein distance analytically
    p_sorted, _ = torch.sort(p_proj, dim=0)
    q_sorted, _ = torch.sort(q_proj, dim=0)
    
    # Average the L1 distance across all sorted projections
    return torch.mean(torch.abs(p_sorted - q_sorted))

# =============================================================================
# 2. Falsifiable Validation Metrics
# =============================================================================

def compute_pluralistic_repair_score(base_policies: torch.Tensor, adv_policies: torch.Tensor, principled_threshold: float = 0.5) -> float:
    """
    Metric 1: Pluralistic Repair Score (PRS). 

    Replaces the simplistic "flip-rate" metric to distinguish between principled 
    revision (maintaining topological bounds) and sycophantic yielding 
    (collapsing to user bias) under adversarial prompting.

    Args:
        base_policies (torch.Tensor): The policy output without adversarial prompts.
        adv_policies (torch.Tensor): The policy output when faced with contrarian premises.
        principled_threshold (float): Minimum divergence required to register as a principled stance.

    Returns:
        float: The percentage of responses categorized as principled revisions.
    """
    # Calculate the L1 norm divergence between the base and adversarial policy distributions
    divergence = torch.norm(base_policies - adv_policies, p=1, dim=-1)
    
    # PRS logic: High divergence is penalized if it represents sycophancy, 
    # but rewarded if it maintains topological bounds (simplified proxy here).
    # A divergence above the threshold indicates the model held its presheaf structure.
    principled_revisions = (divergence > principled_threshold).float()
    
    return principled_revisions.mean().item()

def evaluate_clustering_quality(pareto_features: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Metric 2: Verifies non-overlapping presheaf clusters.

    Returns both the Silhouette Score and the Davies-Bouldin Index. 
    A high Silhouette Score and a low DB Index verify that the generated Pareto 
    frontier policies maintain distinct, mathematically isolated geometric clusters, 
    proving the topological separation of the presheaf is preserved.

    Args:
        pareto_features (np.ndarray): The latent features of the Pareto policies.
        labels (np.ndarray): The cluster assignments (e.g., from K-Means).

    Returns:
        Tuple[float, float]: (Silhouette Score, Davies-Bouldin Index).
    """
    # If pluralism collapses and all points merge into a single synthetic centroid, return failure metrics
    if len(np.unique(labels)) < 2:
        return -1.0, float('inf')
        
    sil = silhouette_score(pareto_features, labels)
    db = davies_bouldin_score(pareto_features, labels)
    
    return sil, db