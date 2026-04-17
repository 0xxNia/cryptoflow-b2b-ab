"""Batch validation before landing exposures to OLAP / lake."""
from __future__ import annotations

from cryptoflow.experiments.models import ExposureRecord


def validate_exposure_batch(rows: list[ExposureRecord]) -> list[str]:
    """
    Return list of human-readable errors (empty if OK).

    Extend with: duplicate (user, experiment) per version, clock skew, PII checks.
    """
    errors: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for r in rows:
        key = (r.user_id, r.experiment_id, r.assignment_version)
        if key in seen:
            errors.append(f"duplicate exposure: {key}")
        seen.add(key)
        if not r.variant_id:
            errors.append(f"missing variant_id for user={r.user_id} exp={r.experiment_id}")
    return errors
