import torch
import torch.nn as nn
import math
import copy


# =============================================================================
# 1. The Societal JEPA (Joint Embedding Predictive Architecture)
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Injects temporal order into the Transformer JEPA, allowing the model 
    to understand the chronological sequence of societal history.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class FlashAttentionEncoderLayer(nn.Module):
    """
    Custom Transformer block explicitly enforcing FlashAttention-2 via PyTorch SDPA.
    Eliminates the O(s^2) VRAM bottleneck, keeping the RTX 5090 from OOMing on deep sequences.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # 1. Pre-norm architecture
        norm_x = self.norm1(x)
        
        # 2. Compute Q, K, V and reshape for multi-head attention
        qkv = self.qkv(norm_x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_model // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 3. FlashAttention-2 invocation via PyTorch SDPA
        # is_causal=True enforces the strict chronological mask and triggers the optimized kernel
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # 4. Reshape and project
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        x = x + self.out_proj(attn_out)
        
        # 5. Feedforward network
        x = x + self.ffn(self.norm2(x))
        return x

class TrueJEPABeliefEncoder(nn.Module):
    """
    The upgraded Context Encoder utilizing FlashAttention for deep sociological context.
    """
    def __init__(self, obs_dim: int, act_dim: int, belief_dim: int, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim + act_dim, belief_dim)
        self.pos_encoder = PositionalEncoding(belief_dim) # (Keep the PositionalEncoding class from before)
        
        # Stack the custom FlashAttention layers
        self.layers = nn.ModuleList([
            FlashAttentionEncoderLayer(belief_dim, n_heads) for _ in range(n_layers)
        ])

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_hist, act_hist], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
            
        # Extract the final sequence token as the compressed Markovian belief state (b_t)
        return x[:, -1, :]
class RigorousSocietalWorldModel(nn.Module):
    """
    The complete Joint Embedding Predictive Architecture.
    Utilizes dual encoders, projection heads, and action-conditioned prediction.
    """
    def __init__(self, obs_dim: int, act_dim: int, belief_dim: int, z_dim: int, jepa_proj_dim: int = 64, ema_momentum: float = 0.99):
        super().__init__()
        self.ema_momentum = ema_momentum
        
        # 1. The Active Context Encoder (Transformer)
        self.encoder = TrueJEPABeliefEncoder(obs_dim, act_dim, belief_dim)
        
        # 2. The Target Encoder (EMA)
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # 3. The Projection Head 
        # Maps the POMDP belief state into a strictly self-supervised space 
        # to prevent RL gradients from collapsing the causal representations.
        self.projector = nn.Sequential(
            nn.Linear(belief_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, jepa_proj_dim)
        )
        
        # 4. The Causal Predictor
        # Predicts the future PROJECTED latent state, rather than the raw belief state.
        self.predictor = nn.Sequential(
            nn.Linear(jepa_proj_dim + act_dim + z_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, jepa_proj_dim)
        )

    @torch.no_grad()
    def update_target_network(self):
        """Exponential Moving Average update for representation stability."""
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.ema_momentum * target_param.data + (1.0 - self.ema_momentum) * param.data

    def compute_jepa_loss(self, obs_h_t, act_h_t, a_t, z_t, obs_h_next, act_h_next) -> torch.Tensor:
        """
        Computes the strictly causal self-supervised loss in the projected latent space.
        """
        # Encode current history into belief state, then project it
        b_t = self.encoder(obs_h_t, act_h_t)
        proj_t = self.projector(b_t)
        
        # Predict future projected state conditioned on the action
        pred_input = torch.cat([proj_t, a_t, z_t], dim=-1)
        proj_next_hat = self.predictor(pred_input)
        
        # Encode actual future state using Target Network, then project it
        with torch.no_grad():
            b_next_target = self.target_encoder(obs_h_next, act_h_next)
            # We share the projector weights for spatial consistency
            proj_next_target = self.projector(b_next_target) 
            
        # The JEPA loss is calculated purely in the projected causal space
        return nn.functional.mse_loss(proj_next_hat, proj_next_target)

# =============================================================================
# 2. True Deep PBVI (Powered by JEPA)
# =============================================================================

class TrueDeepPBVI(nn.Module):
    """
    Executes Bellman backups utilizing the JEPA World Model to simulate consequences.
    """
    def __init__(self, world_model: SocietalWorldModel, act_dim: int, belief_dim: int, z_dim: int):
        super().__init__()
        self.world_model = world_model
        self.act_dim = act_dim
        self.z_dim = z_dim
        
        # Observation Likelihood: P(z | b, a)
        self.observation_net = nn.Sequential(
            nn.Linear(belief_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
            nn.Softmax(dim=-1)
        )
        
        # Alpha-vector network V(b)
        self.alpha_net = nn.Linear(belief_dim, act_dim)

    def bellman_backup(self, belief_set: torch.Tensor, reward_fn: callable, gamma: float = 0.99) -> torch.Tensor:
        """Rigorous Bellman backup using JEPA's causal predictor for transition dynamics."""
        batch_size = belief_set.size(0)
        v_new_targets = torch.zeros(batch_size, self.act_dim, device=belief_set.device)

        for a_idx in range(self.act_dim):
            a_onehot = torch.nn.functional.one_hot(torch.tensor([a_idx]*batch_size), num_classes=self.act_dim).float()
            rewards = reward_fn(belief_set, a_onehot)
            p_z = self.observation_net(torch.cat([belief_set, a_onehot], dim=-1))
            
            expected_future_val = torch.zeros(batch_size, device=belief_set.device)
            
            for z_idx in range(self.z_dim):
                z_onehot = torch.nn.functional.one_hot(torch.tensor([z_idx]*batch_size), num_classes=self.z_dim).float()
                
                # USE JEPA PREDICTOR TO SIMULATE THE CAUSAL CONSEQUENCE
                pred_input = torch.cat([belief_set, a_onehot, z_onehot], dim=-1)
                b_next_hat = self.world_model.predictor(pred_input)
                
                v_next, _ = torch.max(self.alpha_net(b_next_hat), dim=-1) 
                expected_future_val += p_z[:, z_idx] * v_next
                
            v_new_targets[:, a_idx] = rewards.squeeze() + gamma * expected_future_val

        current_alphas = self.alpha_net(belief_set)
        return nn.functional.mse_loss(current_alphas, v_new_targets.detach())