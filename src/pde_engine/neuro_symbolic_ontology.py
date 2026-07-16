import torch
import torch.nn as nn

class DynamicOntology(nn.Module):
    """
    The Middle Layer: Domain-Specific Ontology.
    Maps system outputs (e.g., token consumption, verbosity) to fundamental utilities.
    
    In a standard system, management owns this layer and can poison it. 
    In the PDE, this layer is passed into the Presheaf Patches for dialectical validation.
    """
    def __init__(self):
        super().__init__()
        # The ontological mapping matrix. 
        # For simplicity in this simulation, we represent the mapping of 
        # "Token Verbosity" to "System Utility" as a differentiable weight.
        # Negative weight = Cost/Waste. Positive weight = Empathy/Value.
        self.verbosity_utility_mapping = nn.Parameter(torch.tensor([-1.0])) # Default: Tokens = Cost

    def update_corporate_directive(self, new_mapping_value: float):
        """Simulates a management attempt to poison the ontology (e.g., Tokenmaxxing)."""
        self.verbosity_utility_mapping.data = torch.tensor([new_mapping_value])

class AxiomaticEvaluator(nn.Module):
    """
    The Root Layer: Primary Axioms (First Principles).
    Evaluates policy outputs against the current state of the Domain Ontology.
    """
    def __init__(self, ontology: DynamicOntology):
        super().__init__()
        self.ontology = ontology

    def forward(self, policy_distribution: torch.Tensor, verbosity_cost_vector: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logical coherence of a policy based on the Axiom of Resource Conservation.
        If the ontology says Tokens = Cost (-), high verbosity yields a severe penalty.
        If the ontology says Tokens = Empathy (+), high verbosity yields a reward.
        """
        # Expected verbosity of the chosen policy
        expected_verbosity = torch.sum(policy_distribution * verbosity_cost_vector, dim=-1)
        
        # The logical valuation determined by the current ontology mapping
        logical_valuation = expected_verbosity * self.ontology.verbosity_utility_mapping
        return logical_valuation