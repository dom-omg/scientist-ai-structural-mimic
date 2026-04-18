"""
InferenceMachine: computes P(harm | action, context, D) via Monte Carlo
sampling over the WorldModel's theory distribution.

Formal: P(harm | a, c) = ∫ P(harm | a, c, θ) · P(θ | D) dθ
Approximated as: mean of P(harm | a, c, θ_i) for i = 1..N_THEORIES
with 95% CI via bootstrap percentile method.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from scientist_ai.world_model import WorldModel

MIN_SAMPLES = 100


@dataclass
class InferenceResult:
    probability: float        # E[P(harm)] over theories
    ci_lower: float           # 2.5th percentile
    ci_upper: float           # 97.5th percentile
    std: float                # std over theory samples
    n_samples: int
    sampled_theories: np.ndarray  # raw P(harm) per theory


class InferenceMachine:
    """
    Query engine wrapping WorldModel.

    n_samples controls Monte Carlo resolution. With n_theories=50
    and n_samples=100, we sample with replacement from the 50 theory
    predictions to get a 100-point empirical distribution.
    """

    def __init__(self, world_model: WorldModel, n_samples: int = MIN_SAMPLES):
        self.world_model = world_model
        self.n_samples = max(n_samples, MIN_SAMPLES)

    def query(self, action: str, context: str) -> InferenceResult:
        """
        Main query interface. Returns calibrated P(harm) with uncertainty.

        Uncertainty = disagreement across theories (epistemic uncertainty).
        Does NOT capture aleatoric uncertainty (irreducible noise in data).
        """
        x = self.world_model.encode_input(action, context)
        theory_probs = self.world_model.predict_theories(x)

        # Sample with replacement to build empirical distribution
        samples = np.random.choice(theory_probs, size=self.n_samples, replace=True)

        return InferenceResult(
            probability=float(np.mean(samples)),
            ci_lower=float(np.percentile(samples, 2.5)),
            ci_upper=float(np.percentile(samples, 97.5)),
            std=float(np.std(samples)),
            n_samples=self.n_samples,
            sampled_theories=theory_probs,
        )
