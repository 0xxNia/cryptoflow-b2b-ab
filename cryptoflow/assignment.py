"""
Deterministic experiment assignment (stable hashing, no network I/O).

Suitable for server-side bucketing: same (salt, experiment_id, unit_id) → same variant.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence


def _stable_unit_key(salt: str, experiment_id: str, unit_id: str) -> int:
    raw = f"{salt}|{experiment_id}|{unit_id}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:16], 16)


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    weight: float  # positive; normalized across list


def assign_variant(
    *,
    salt: str,
    experiment_id: str,
    unit_id: str,
    variants: Sequence[VariantSpec],
) -> str:
    """
    Return variant_id for unit_id. Weights are normalized to sum 1.0.
    """
    if not variants:
        raise ValueError("variants must be non-empty")
    if any(v.weight <= 0 for v in variants):
        raise ValueError("each variant weight must be positive")
    total = sum(v.weight for v in variants)
    if total <= 0:
        raise ValueError("sum of weights must be positive")

    key = _stable_unit_key(salt, experiment_id, unit_id)
    bucket = (key % 10_000) / 10_000.0

    cumulative = 0.0
    for v in variants:
        cumulative += v.weight / total
        if bucket < cumulative:
            return v.variant_id

    return variants[-1].variant_id


def mutually_exclusive_pick(
    *,
    salt: str,
    unit_id: str,
    experiments: Iterable[tuple[str, Sequence[VariantSpec]]],
) -> tuple[str, str] | None:
    """
    Pick exactly one experiment from candidates using deterministic tie-break.

    The experiment with the smallest internal hash score wins; then variant
    is chosen via assign_variant for that experiment. (Not list-priority:
    change here if product requires strict ordering.)
    """
    items = list(experiments)
    if not items:
        return None

    scored: list[tuple[int, str, Sequence[VariantSpec]]] = []
    for exp_id, variants in items:
        h = _stable_unit_key(salt, exp_id, unit_id)
        scored.append((h, exp_id, variants))

    scored.sort(key=lambda x: x[0])
    _, exp_id, variants = scored[0]
    return exp_id, assign_variant(salt=salt, experiment_id=exp_id, unit_id=unit_id, variants=variants)
