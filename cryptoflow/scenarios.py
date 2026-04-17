"""
Synthetic stress hooks for simulators (not a substitute for live inference).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StressScenario(str, Enum):
    FLASH_CRASH = "flash_crash"
    ETF_APPROVAL_SHOCK = "etf_approval_shock"
    LIQUIDATION_CASCADE = "liquidation_cascade"
    BASELINE = "baseline"


@dataclass(frozen=True)
class ScenarioParams:
    """Tune simulator agents / volume multipliers (implement in agents layer)."""

    volume_multiplier: float = 1.0
    volatility_multiplier: float = 1.0
    sentiment_shift: float = 0.0  # -1..1


def params_for_scenario(scenario: StressScenario) -> ScenarioParams:
    if scenario == StressScenario.FLASH_CRASH:
        return ScenarioParams(volume_multiplier=2.5, volatility_multiplier=3.0, sentiment_shift=-0.8)
    if scenario == StressScenario.ETF_APPROVAL_SHOCK:
        return ScenarioParams(volume_multiplier=1.8, volatility_multiplier=1.5, sentiment_shift=0.6)
    if scenario == StressScenario.LIQUIDATION_CASCADE:
        return ScenarioParams(volume_multiplier=2.0, volatility_multiplier=2.2, sentiment_shift=-0.5)
    return ScenarioParams()
