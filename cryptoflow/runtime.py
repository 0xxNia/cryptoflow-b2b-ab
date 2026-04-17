"""
Glue: assign user to experiment and materialize exposure + product event.

Use from assignment service or offline backfills.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from cryptoflow.assignment import VariantSpec, assign_variant
from cryptoflow.experiments.models import ExperimentSpec, ExposureRecord
from cryptoflow.export.amplitude import exposure_to_product_event
from cryptoflow.regimes import MarketRegime


def assign_and_record(
    spec: ExperimentSpec,
    *,
    user_id: str,
    ts: datetime | None = None,
    market_regime: MarketRegime | None = None,
    platform: str | None = None,
    app_version: str | None = None,
    assignment_version: str = "v1",
) -> tuple[ExposureRecord, dict[str, Any]]:
    equal_weights = tuple(VariantSpec(v.variant_id, weight=1.0) for v in spec.variants)
    vid = assign_variant(
        salt=spec.salt,
        experiment_id=spec.experiment_id,
        unit_id=user_id,
        variants=equal_weights,
    )
    ts_use = ts or datetime.now(timezone.utc)
    exposure = ExposureRecord(
        ts=ts_use,
        user_id=user_id,
        experiment_id=spec.experiment_id,
        variant_id=vid,
        assignment_version=assignment_version,
        market_regime=market_regime,
        platform=platform,
        app_version=app_version,
    )
    return exposure, exposure_to_product_event(exposure)
