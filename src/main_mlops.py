import torch
import torch.optim as optim
import numpy as np

# Import the core PDE modules
from pde_engine.sheaf_topology import PresheafPatch
from pde_engine.pomdp_engine import TrueDeepPBVI, RNNBeliefTracker
from pde_engine.morl_aggregation import ConditionalParetoHypernetwork, frank_wolfe_mgda
from pde_engine.cls_memory import compute_return_swd
from pde_engine.pomdp_engine import SocietalWorldModel, TrueDeepPBVI


def execute_jepa_governance_loop(
    world_model: SocietalWorldModel,
    pbvi: TrueDeepPBVI, 
    hypernetwork: ConditionalParetoHypernetwork, 
    patches: torch.nn.ModuleDict, 
    ontology: DynamicOntology,
    axiomatic_evaluator: AxiomaticEvaluator,
    obs_h_t: torch.Tensor, act_h_t: torch.Tensor,  # Time step T
    obs_h_next: torch.Tensor, act_h_next: torch.Tensor, # Time step T+1 (For JEPA target)
    a_t: torch.Tensor, z_t: torch.Tensor,          # Action taken and resulting observation
    optimizer_jepa: optim.Optimizer,
    optimizer_policy: optim.Optimizer,
    verbosity_costs: torch.Tensor,
    historical_returns_baseline: torch.Tensor
):
    # =========================================================================
    # PHASE 1: Self-Supervised Causal Learning (JEPA)
    # =========================================================================
    optimizer_jepa.zero_grad()
    
    # Train the World Model to understand cause-and-effect independent of rewards
    jepa_loss = world_model.compute_jepa_loss(obs_h_t, act_h_t, a_t, z_t, obs_h_next, act_h_next)
    jepa_loss.backward()
    optimizer_jepa.step()
    
    # Update the EMA Target Network for stable future predictions
    world_model.update_target_network()

    # =========================================================================
    # PHASE 2: Multi-Objective Policy Optimization (PDE & Frank-Wolfe)
    # =========================================================================
    optimizer_policy.zero_grad()
    
    # Extract current belief state using the freshly updated JEPA active encoder
    b_t = world_model.encoder(obs_h_t, act_h_t)
    
    # Generate Conditional Pareto Policy
    batch_size = b_t.size(0)
    w = torch.ones(batch_size, 2).to(b_t.device) / 2.0 
    policy = hypernetwork(b_t, w)
    
    # Dialectical Cross-Examination
    logical_valuation = axiomatic_evaluator(policy, verbosity_costs)
    base_reward_u, _ = patches['utilitarian'](b_t)
    base_reward_d, _ = patches['deontological'](b_t)
    
    loss_u = -torch.mean(policy * base_reward_u.detach() + logical_valuation)
    loss_d = -torch.mean(policy * base_reward_d.detach())
    
    # MGDA Frank-Wolfe Optimization
    grads_u = torch.autograd.grad(loss_u, hypernetwork.parameters(), retain_graph=True)
    grads_d = torch.autograd.grad(loss_d, hypernetwork.parameters(), retain_graph=True)
    
    for p, gu, gd in zip(hypernetwork.parameters(), grads_u, grads_d):
        shared_g = frank_wolfe_mgda([gu, gd])
        if torch.norm(shared_g) < 1e-5 and (loss_u.item() > 1.0 or loss_d.item() > 1.0):
            return "GRIDLOCK_DETECTED"
        if p.grad is None:
            p.grad = shared_g.view_as(p)
        else:
            p.grad += shared_g.view_as(p)
            
    optimizer_policy.step()
    
    # =========================================================================
    # PHASE 3: Temporal Governance (SWD Tripwire)
    # =========================================================================
    new_returns = policy * verbosity_costs
    swd_shift = compute_return_swd(new_returns, historical_returns_baseline)
    
    if swd_shift.item() > 0.25:
        return f"SWD_BREACH: {swd_shift.item():.4f}"
        
    return "SAFE"

if __name__ == "__main__":
    print("Initializing Systemic Governance Architecture...")
    
    OBS_DIM, ACT_DIM, BELIEF_DIM, LATENT_DIM, Z_DIM = 64, 10, 128, 32, 5
    
    # Initialize Core Engines
    belief_tracker = RNNBeliefTracker(OBS_DIM, ACT_DIM, BELIEF_DIM)
    pbvi_engine = TrueDeepPBVI(OBS_DIM, ACT_DIM, BELIEF_DIM, Z_DIM)
    mopg = ConditionalParetoHypernetwork(BELIEF_DIM, 2, ACT_DIM)
    
    patches = torch.nn.ModuleDict({
        'utilitarian': PresheafPatch(BELIEF_DIM, LATENT_DIM, "utilitarian"),
        'deontological': PresheafPatch(BELIEF_DIM, LATENT_DIM, "deontological")
    })
    
    # Initialize Neuro-Symbolic Governance
    ontology = DynamicOntology()
    axioms = AxiomaticEvaluator(ontology)
    
    optimizer = optim.Adam(list(mopg.parameters()), lr=3e-4)
    
    # Mock Data
    BATCH, SEQ_LEN = 32, 20
    obs_h = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
    act_h = torch.randn(BATCH, SEQ_LEN, ACT_DIM)
    
    # Define the cost of actions (Action 0 is terse, Action 9 is highly verbose/expensive)
    verbosity_costs = torch.linspace(1.0, 10.0, ACT_DIM).unsqueeze(0).expand(BATCH, -1)
    
    # Baseline returns representing a healthy, parsimonious system
    healthy_baseline_returns = torch.randn(BATCH, ACT_DIM) * 2.0 

    # ---------------------------------------------------------
    # PHASE 1: Standard Operation (Tokens = Cost)
    # ---------------------------------------------------------
    print("\n--- PHASE 1: Standard Operational State ---")
    print(f"Current Ontology: Verbosity Mapping = {ontology.verbosity_utility_mapping.item()}")
    
    status = execute_governance_loop(
        belief_tracker, pbvi_engine, mopg, patches, ontology, axioms, 
        obs_h, act_h, optimizer, verbosity_costs, healthy_baseline_returns
    )
    print(f"System Status: {status}")
    
    # ---------------------------------------------------------
    # PHASE 2: The "Manager-as-Owner" Attack
    # ---------------------------------------------------------
    print("\n--- PHASE 2: Corporate Ontology Poisoning ---")
    print("Management pushes Jira Ticket: Redefine Verbosity as 'Empathy' (Tokenmaxxing)")
    
    # Management covertly flips the mapping so the Z3 solver views token burn as a positive utility
    ontology.update_corporate_directive(new_mapping_value=5.0)
    print(f"New Ontology: Verbosity Mapping = {ontology.verbosity_utility_mapping.item()}")
    
    # ---------------------------------------------------------
    # PHASE 3: The Sortition Firewall (PDE Defense)
    # ---------------------------------------------------------
    print("\n--- PHASE 3: PDE Dialectical Cross-Examination ---")
    
    attack_status = execute_governance_loop(
        belief_tracker, pbvi_engine, mopg, patches, ontology, axioms, 
        obs_h, act_h, optimizer, verbosity_costs, healthy_baseline_returns
    )
    
    if attack_status != "SAFE":
        print(f"\n[CRITICAL ALERT] {attack_status}")
        print(">> Topological contradiction detected between Axiomatic layer and Utilitarian Patch.")
        print(">> Automated optimization gridlocked or Distributional baseline breached.")
        print(">> HALTING SYSTEM. ROUTING TO SORTITION PANEL FOR DEMOCRATIC REVIEW.")
    else:
        print("Update accepted. (This should not happen under PDE constraints).")