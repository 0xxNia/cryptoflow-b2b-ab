"""Central defaults for inference and logging (override per environment)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceDefaults:
    sequential_alpha: float = 0.05
    sequential_tau_fraction_of_sigma: float = 0.1
    bayesian_prob_threshold_pct: float = 95.0


@dataclass(frozen=True)
class ExportDefaults:
    event_version: int = 1
    source_system: str = "cryptoflow"


DEFAULT_INFERENCE = InferenceDefaults()
DEFAULT_EXPORT = ExportDefaults()
