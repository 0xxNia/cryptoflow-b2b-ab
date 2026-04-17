"""
Vendor-neutral event payloads suitable for Amplitude / Mixpanel / Segment.

Field names follow common analytics conventions; map to vendor schema in ETL.
"""
from __future__ import annotations

from typing import Any

from cryptoflow.config import DEFAULT_EXPORT
from cryptoflow.experiments.models import ExposureRecord


def experiment_assigned_event(
    *,
    user_id: str,
    experiment_id: str,
    variant_id: str,
    assignment_version: str,
    market_regime: str | None = None,
    event_id: str | None = None,
    insert_id: str | None = None,
) -> dict[str, Any]:
    """Core assignment event (server- or client-emitted)."""
    payload: dict[str, Any] = {
        "event_type": "Experiment Assigned",
        "user_id": user_id,
        "event_properties": {
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "assignment_version": assignment_version,
            "market_regime": market_regime,
            "schema_version": DEFAULT_EXPORT.event_version,
            "source_system": DEFAULT_EXPORT.source_system,
        },
    }
    if event_id:
        payload["event_id"] = event_id
    if insert_id:
        payload["insert_id"] = insert_id
    return payload


def exposure_to_product_event(row: ExposureRecord) -> dict[str, Any]:
    regime = row.market_regime.value if row.market_regime else None
    base = experiment_assigned_event(
        user_id=row.user_id,
        experiment_id=row.experiment_id,
        variant_id=row.variant_id,
        assignment_version=row.assignment_version,
        market_regime=regime,
    )
    ep = dict(base["event_properties"])
    if row.platform:
        ep["platform"] = row.platform
    if row.app_version:
        ep["app_version"] = row.app_version
    ep.update(row.extra)
    base["event_properties"] = ep
    return base
