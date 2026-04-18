"""
cryptoflow.chart_helpers — Plotly decorators for market-regime context.

`add_regime_overlay(fig, regime_series)` paints each regime segment as a
semi-transparent vertical band under the existing traces. This is the
"killer feature" for crypto-exchange PMs: a conversion uplift that shows
up only during high_vol isn't a design win — it's a market-driven artefact.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd
import plotly.graph_objects as go


# Regime palette — tuned for dark dashboard backgrounds.
# Hue encodes direction, opacity stays low so traces stay readable.
DEFAULT_REGIME_COLORS: dict[str, str] = {
    "bull":     "rgba( 46, 204, 113, {alpha})",  # green
    "bear":     "rgba(231,  76,  60, {alpha})",  # red
    "high_vol": "rgba(241, 196,  15, {alpha})",  # amber
}


def _segments(regime_series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Collapse a per-timestamp regime series into (start, end, regime) runs.

    Consecutive equal values are merged so one `vrect` is drawn per run,
    not per tick.
    """
    if regime_series.empty:
        return []

    # Ensure index is datetime-like (Plotly needs absolute timestamps for vrect)
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        regime_series = regime_series.copy()
        regime_series.index = pd.to_datetime(regime_series.index)

    regime_series = regime_series.sort_index()

    values      = regime_series.to_numpy()
    timestamps  = regime_series.index.to_numpy()
    segments    = []

    start_ts  = timestamps[0]
    cur       = values[0]
    for i in range(1, len(values)):
        if values[i] != cur:
            segments.append((pd.Timestamp(start_ts), pd.Timestamp(timestamps[i]), str(cur)))
            start_ts, cur = timestamps[i], values[i]
    segments.append((pd.Timestamp(start_ts), pd.Timestamp(timestamps[-1]), str(cur)))
    return segments


def add_regime_overlay(
    fig:            go.Figure,
    regime_series:  pd.Series,
    alpha:          float = 0.15,
    colors:         Mapping[str, str] | None = None,
    add_legend:     bool  = True,
) -> go.Figure:
    """
    Paint regime-coloured vertical bands under every trace in `fig`.

    Parameters
    ----------
    fig:
        Plotly Figure to mutate. Returned for chaining.
    regime_series:
        pd.Series with a DatetimeIndex and values in {"bull", "bear", "high_vol"}.
        Unknown regime values are skipped silently (so callers can pass a
        pre-filtered slice without breaking on edge cases).
    alpha:
        Band opacity. Default 0.15 keeps traces fully legible.
    colors:
        Optional override of {regime → rgba template with {alpha}}.
    add_legend:
        If True, adds a small invisible scatter trace per regime so a
        legend entry appears (otherwise vrects don't show in the legend).
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    palette = dict(DEFAULT_REGIME_COLORS) if colors is None else dict(colors)

    for start, end, regime in _segments(regime_series):
        template = palette.get(regime)
        if template is None:
            continue
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=template.format(alpha=alpha),
            line_width=0,
            layer="below",
            annotation_text=None,
        )

    if add_legend:
        # Add one invisible marker per regime so Plotly renders a legend swatch.
        for regime, template in palette.items():
            if regime not in set(regime_series.astype(str).unique()):
                continue
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=template.format(alpha=0.6), symbol="square"),
                name=f"regime: {regime}",
                showlegend=True,
                hoverinfo="skip",
            ))

    return fig
