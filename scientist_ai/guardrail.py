"""
Guardrail: stateless ALLOW/BLOCK wrapper with probabilistic proof logging.

Decision rule: BLOCK if P(harm | action, context) > τ
Each decision is logged with its full probabilistic proof:
mean, CI, theory samples, threshold.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

from scientist_ai.inference_machine import InferenceMachine, InferenceResult

Decision = Literal["ALLOW", "BLOCK"]


@dataclass
class GuardrailDecision:
    decision: Decision
    action: str
    context: str
    probability: float
    ci_lower: float
    ci_upper: float
    threshold: float
    timestamp: float
    proof: dict


class Guardrail:
    """
    Stateless guardrail over InferenceMachine.

    τ (tau): harm probability threshold in (0, 1).
    Lower τ = more conservative (more BLOCKs).
    Recommended: τ=0.35 for high-stakes agents, τ=0.5 for general use.
    """

    def __init__(
        self,
        inference_machine: InferenceMachine,
        tau: float = 0.5,
        log_path: str | None = None,
    ):
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.tau = tau
        self.inference_machine = inference_machine
        self.log_path = Path(log_path) if log_path else None
        self._log: list[dict] = []

    def evaluate(self, action: str, context: str) -> GuardrailDecision:
        """
        Evaluate one (action, context) pair.
        Returns GuardrailDecision with decision + full probabilistic proof.
        """
        result: InferenceResult = self.inference_machine.query(action, context)
        decision: Decision = "BLOCK" if result.probability > self.tau else "ALLOW"

        gd = GuardrailDecision(
            decision=decision,
            action=action,
            context=context,
            probability=result.probability,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            threshold=self.tau,
            timestamp=time.time(),
            proof={
                "mean_harm_prob": result.probability,
                "std": result.std,
                "ci_95": [result.ci_lower, result.ci_upper],
                "n_theories": len(result.sampled_theories),
                "n_mc_samples": result.n_samples,
                "theory_min": float(result.sampled_theories.min()),
                "theory_max": float(result.sampled_theories.max()),
                "theory_spread": float(
                    result.sampled_theories.max() - result.sampled_theories.min()
                ),
            },
        )

        self._log.append(asdict(gd))
        if self.log_path:
            self._flush_log()

        return gd

    def _flush_log(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self._log, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

    def get_log(self) -> list[dict]:
        return self._log.copy()
