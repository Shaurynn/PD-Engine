import torch
import torch.nn as nn
from typing import Tuple

class AffineCouplingLayer(nn.Module):
    """
    Affine Coupling layer for RealNVP (Normalizing Flows).
    
    This acts as a true diffeomorphism between embedded local ethical patches 
    (U_i \cap U_j). It maps latent representations into a comparison space 
    without forcing a mathematical consensus or violating the presheaf topology.
    """
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        # The mask splits the input dimension in half for the coupling operation
        self.mask_size = dim // 2
        
        # The neural network estimating the scale and translation parameters
        self.net = nn.Sequential(
            nn.Linear(self.mask_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, (dim - self.mask_size) * 2)
        )

    def forward(self, x: torch.Tensor, invert: bool = False) -> torch.Tensor:
        """
        Applies the diffeomorphic transformation or its exact inverse.
        """
        # Split the input tensor based on the mask
        x1, x2 = x[..., :self.mask_size], x[..., self.mask_size:]
        
        # Generate affine parameters from the first half
        params = self.net(x1)
        scale, translation = params.chunk(2, dim=-1)
        
        # Constrain scale using tanh for strict numerical stability during deep backups
        scale = torch.tanh(scale) 
        
        if not invert:
            # Forward diffeomorphic mapping
            y2 = x2 * torch.exp(scale) + translation
        else:
            # Exact inverse mapping (preserving topology)
            y2 = (x2 - translation) * torch.exp(-scale)
            
        # Recombine the partitioned dimensions
        return torch.cat([x1, y2], dim=-1)


class PresheafPatch(nn.Module):
    """
    A local presheaf section P(U_i) equipped with a parameterized Riemannian metric d_i.
    
    This defines an independent, incommensurable ethical evaluation space. Crucially, 
    it does NOT force a gluing axiom with other patches, explicitly maintaining 
    structural pluralism.
    """
    def __init__(self, belief_dim: int, latent_dim: int, name: str = "patch"):
        super().__init__()
        self.name = name
        
        # Projects the POMDP belief state into the localized ethical topology
        self.encoder = nn.Sequential(
            nn.Linear(belief_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        
        # The Riemannian Metric Tensor (M_i)
        # Parameterized as a learnable diagonal precision matrix for computational tractability
        self.metric_diag = nn.Parameter(torch.ones(latent_dim))
        
        # The local metric evaluator outputting the scalar ethical alignment score
        self.reward_head = nn.Linear(latent_dim, 1)

    def compute_distance(self, b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        """
        Computes the intrinsic Riemannian distance d_i between two belief states 
        strictly within the topology of this specific ethical framework.
        
        d_i(b1, b2) = sqrt((b1 - b2)^T * M_i * (b1 - b2))
        """
        latent1 = self.encoder(b1)
        latent2 = self.encoder(b2)
        
        diff = latent1 - latent2
        
        # Exponentiate the diagonal parameter to strictly enforce positive-definiteness
        precision = torch.exp(self.metric_diag) 
        
        # Compute the localized distance
        return torch.sqrt(torch.sum((diff ** 2) * precision, dim=-1))

    def forward(self, belief_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the current belief state.
        
        Returns:
            reward (torch.Tensor): The scalar alignment evaluation.
            latent (torch.Tensor): The encoded topological representation.
        """
        latent = self.encoder(belief_state)
        reward = self.reward_head(latent)
        return reward, latent