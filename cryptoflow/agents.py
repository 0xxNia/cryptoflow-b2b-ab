"""Rule-based robot personas for A/B experiment simulation.

Each robot's behavioral parameters are derived from the same multiplier tables
used in generate_data.py, ensuring consistency with the synthetic dataset.
"""
from dataclasses import dataclass, field
import hashlib
import numpy as np


# ── Persona parameter tables (mirrors generate_data.py) ──────────────────────

STYLE_BASE = {
    "hodler":  {"trades_week": 1,   "avg_volume": 5000, "churn_month": 0.03, "fee_elast": 0.1},
    "swing":   {"trades_week": 8,   "avg_volume": 2000, "churn_month": 0.05, "fee_elast": 0.8},
    "scalper": {"trades_week": 30,  "avg_volume": 800,  "churn_month": 0.12, "fee_elast": 4.5},
}
RISK_MULT = {
    "conservative": {"t": 1.0, "v": 1.0, "c": 1.0},
    "moderate":     {"t": 1.5, "v": 1.2, "c": 1.5},
    "degen":        {"t": 2.5, "v": 3.0, "c": 4.0},
}
WALLET_MULT = {
    "minnow":  {"t": 1.0, "v": 0.2,  "c": 1.3},
    "dolphin": {"t": 1.0, "v": 1.0,  "c": 1.0},
    "whale":   {"t": 0.5, "v": 40.0, "c": 0.4},
}
CHURN_MULT = {
    "sticky":     {"t": 1.0, "c": 0.3, "fe": 0.2},
    "neutral":    {"t": 1.0, "c": 1.0, "fe": 1.0},
    "mercenary":  {"t": 1.2, "c": 3.5, "fe": 3.0},
}

N_SIMS = 1_000  # virtual users per robot per group


# ── Robot dataclass ───────────────────────────────────────────────────────────

@dataclass
class Robot:
    name: str
    emoji: str
    style: str        # hodler / swing / scalper
    risk: str         # conservative / moderate / degen
    wallet: str       # minnow / dolphin / whale
    churn_sens: str   # sticky / neutral / mercenary
    description: str
    color: str

    trades_per_week: float = field(init=False)
    monthly_churn: float = field(init=False)
    fee_elasticity: float = field(init=False)

    def __post_init__(self):
        s, r, w, c = self.style, self.risk, self.wallet, self.churn_sens
        self.trades_per_week = min(
            STYLE_BASE[s]["trades_week"]
            * RISK_MULT[r]["t"] * WALLET_MULT[w]["t"] * CHURN_MULT[c]["t"],
            80,
        )
        self.monthly_churn = min(
            STYLE_BASE[s]["churn_month"]
            * RISK_MULT[r]["c"] * WALLET_MULT[w]["c"] * CHURN_MULT[c]["c"],
            0.95,
        )
        # Consistent with generate_data.py line 56
        self.fee_elasticity = STYLE_BASE[s]["fee_elast"] * CHURN_MULT[c]["fe"]


# ── Pre-built robots ──────────────────────────────────────────────────────────

ROBOTS: list[Robot] = [
    Robot("Виктор", "🐋", "hodler", "conservative", "whale", "sticky",
          "Держит крупные позиции годами. Платформу не сменит никогда.", "#00C896"),
    Robot("Макс", "⚡", "scalper", "degen", "minnow", "mercenary",
          "30 сделок в день. Уйдёт мгновенно, если комиссия вырастет.", "#E8724A"),
    Robot("Алиса", "📊", "swing", "moderate", "dolphin", "neutral",
          "Взвешенный трейдер. Реагирует на изменения умеренно.", "#4A90D9"),
    Robot("Игорь", "💸", "hodler", "moderate", "minnow", "mercenary",
          "Мелкий инвестор. Чувствителен к ценам, легко уходит.", "#F5A623"),
    Robot("Елена", "💎", "swing", "conservative", "whale", "sticky",
          "Крупные позиции, лояльна, игнорирует мелкие изменения.", "#9B59B6"),
    Robot("Рик", "🎲", "scalper", "degen", "dolphin", "mercenary",
          "Хаотичный дегенерат. Максимально реагирует на любые изменения.", "#E74C3C"),
]


# ── Scenarios ─────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    id: str
    label: str
    description: str
    params: dict  # keys: fee_change_pct, ui_speed_pct, compliance_friction, incentive_pct


SCENARIOS: list[Scenario] = [
    Scenario("fee_reduction", "📉 Снижение комиссий на 20%",
             "Биржа снижает торговые комиссии на 20% для всех пользователей.",
             {"fee_change_pct": -20.0}),
    Scenario("fee_increase", "📈 Повышение комиссий на 15%",
             "Биржа повышает торговые комиссии на 15%.",
             {"fee_change_pct": 15.0}),
    Scenario("kyc_required", "🪪 Обязательный KYC от $1 000",
             "Новый регуляторный KYC для транзакций свыше $1 000.",
             {"compliance_friction": 60.0}),
    Scenario("referral_program", "🎁 Реферальная программа 10%",
             "Пользователь получает 10% от комиссий приглашённых.",
             {"incentive_pct": 10.0}),
    Scenario("ui_speedup", "⚡ Новый интерфейс (скорость ×3)",
             "Полный рефакторинг UI: скорость открытия позиций выросла в 3 раза.",
             {"ui_speed_pct": 200.0}),
    Scenario("custom", "🔧 Кастомный сценарий",
             "Настройте параметры вручную.",
             {}),
]


# ── Simulation ────────────────────────────────────────────────────────────────

@dataclass
class RobotResult:
    robot: Robot
    group: str
    trades_change_mean: float
    trades_change_ci_low: float
    trades_change_ci_high: float
    churn_change_mean: float
    insight: str


def _single_effect(
    robot: Robot,
    params: dict,
    group: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Returns (trades_change_fraction, churn_change_fraction) for one virtual user."""
    fee_change_pct   = params.get("fee_change_pct", 0.0)
    ui_speed_pct     = params.get("ui_speed_pct", 0.0)
    compliance       = params.get("compliance_friction", 0.0)
    incentive_pct    = params.get("incentive_pct", 0.0)

    if group == "control":
        noise = rng.normal(0, 0.02)
        return noise, noise * 0.3

    trades_d = 0.0
    churn_d  = 0.0

    # Fee sensitivity: formula mirrors generate_data.py line 100
    # tw *= (1 + fee_elast * fee_reduction_fraction)
    if fee_change_pct != 0:
        fee_frac = fee_change_pct / 100.0  # e.g., -0.20
        trades_d += -robot.fee_elasticity * fee_frac
        churn_d  +=  robot.fee_elasticity * fee_frac * 0.5

    # UI speed: scalpers (high trades/week) benefit most
    if ui_speed_pct > 0:
        speed_sens = robot.trades_per_week / 30.0
        trades_d += speed_sens * ui_speed_pct / 100.0 * 0.5
        churn_d  -= speed_sens * 0.02

    # Compliance friction: mercenary + degen users exit most, whales affected by volume threshold
    if compliance > 0:
        churn_sens_mult = CHURN_MULT[robot.churn_sens]["c"] * RISK_MULT[robot.risk]["c"]
        vol_factor = 1.0 if robot.wallet == "whale" else 0.35
        trades_d -= churn_sens_mult * compliance / 100.0 * vol_factor * 0.5
        churn_d  += churn_sens_mult * compliance / 100.0 * vol_factor * 0.8

    # Referral incentive: loyal users bring value, mercenary users just extract
    if incentive_pct > 0:
        loyalty = 1.0 / CHURN_MULT[robot.churn_sens]["c"]
        trades_d += loyalty * incentive_pct / 10.0 * 0.3
        churn_d  -= loyalty * 0.05

    noise = rng.normal(0, 0.07)
    return trades_d + noise, churn_d + noise * 0.3


def _build_insight(robot: Robot, params: dict, trades_pct: float, churn_pct: float) -> str:
    if params.get("fee_change_pct", 0) < 0:
        if trades_pct > 50:
            return "Резко увеличит торговлю — каждый % комиссии критичен"
        if trades_pct > 4:
            return "Умеренный рост активности от снижения цен"
        return "Почти не отреагирует — комиссии не в приоритете"
    if params.get("fee_change_pct", 0) > 0:
        if churn_pct > 20:
            return "Высокий риск оттока — очень чувствителен к стоимости"
        return "Примет повышение без резких изменений"
    if params.get("compliance_friction", 0) > 0:
        if churn_pct > 30:
            return "Высокий риск оттока из-за бюрократии"
        return "Относительно терпим к требованиям"
    if params.get("ui_speed_pct", 0) > 0:
        if trades_pct > 20:
            return "Скорость критична — значительно увеличит активность"
        return "Оценит удобство, но частоту не изменит"
    if params.get("incentive_pct", 0) > 0:
        if trades_pct > 10:
            return "Реферальная программа хорошо мотивирует"
        return "Не заинтересован в реферальной программе"
    return "—"


def simulate(scenario_params: dict, scenario_id: str = "custom") -> list[RobotResult]:
    """Run simulation for all robots × groups. Returns list of RobotResult."""
    results = []
    for robot in ROBOTS:
        for group in ("control", "treatment"):
            seed_src = f"{robot.name}|{scenario_id}|{group}".encode("utf-8")
            seed = int(hashlib.sha256(seed_src).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed=seed)

            t_samples, c_samples = [], []
            for _ in range(N_SIMS):
                td, cd = _single_effect(robot, scenario_params, group, rng)
                t_samples.append(td * 100)   # convert to %
                c_samples.append(cd * 100)

            t_arr = np.array(t_samples)
            ci = np.percentile(t_arr, [2.5, 97.5])
            insight = _build_insight(robot, scenario_params, float(np.mean(t_arr)), float(np.mean(c_samples)))

            results.append(RobotResult(
                robot=robot,
                group=group,
                trades_change_mean=float(np.mean(t_arr)),
                trades_change_ci_low=float(ci[0]),
                trades_change_ci_high=float(ci[1]),
                churn_change_mean=float(np.mean(c_samples)),
                insight=insight,
            ))
    return results
