"""Configurable reward clipping and per-batch standardization.

The transform runs after raw reward scoring and before advantage computation so
the group-relative advantages consumed by GRPO/GSPO are derived from a
stabilized reward distribution. ``disabled`` (the default) is a numerical
no-op: for every finite input the output equals the input, so existing runs are
unchanged.

Three modes are supported:
    * disabled      - no-op; rewards pass through untouched.
    * clip          - clamp every reward to ``[clip_min, clip_max]``.
    * standardize   - per-batch z-score: ``(r - mean) / (std + eps)``.

:func:`transform_rewards` returns the transformed rewards plus a ``summary``
dict that keeps raw and transformed distribution statistics in separate blocks
so dashboards and tests can observe both at once.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

RewardTransformMode = Literal["disabled", "clip", "standardize"]

_VALID_MODES = ("disabled", "clip", "standardize")


@dataclass(frozen=True, slots=True)
class RewardTransformConfig:
    """Resolved reward-transform settings, validated at construction time."""

    mode: RewardTransformMode = "disabled"
    clip_min: float | None = None
    clip_max: float | None = None
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"reward_transform mode must be one of {_VALID_MODES}, got {self.mode!r}")
        if self.eps is None or not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"reward_transform eps must be a positive finite number, got {self.eps}")
        if self.mode == "clip":
            if self.clip_min is None or self.clip_max is None:
                raise ValueError("reward_transform mode='clip' requires both clip_min and clip_max")
            if not np.isfinite(self.clip_min) or not np.isfinite(self.clip_max):
                raise ValueError("reward_transform clip_min and clip_max must be finite")
            if self.clip_min > self.clip_max:
                raise ValueError(
                    f"reward_transform clip_min ({self.clip_min}) must be <= clip_max ({self.clip_max})"
                )

    @property
    def enabled(self) -> bool:
        """True for any non-disabled mode."""

        return self.mode != "disabled"


def transform_rewards(rewards: Iterable[float], config: RewardTransformConfig) -> tuple[list[float], dict]:
    """Apply ``config`` to ``rewards`` and return transformed values plus a summary.

    Guarantees:
        * Empty input -> ``([], summary)`` with empty distribution stats.
        * Non-finite input raises ``ValueError`` naming the stage; the message
          reports the offending position, never the full training samples.
        * Output is finite for every well-formed (finite) input.

    ``summary`` always carries both ``raw`` and ``transformed`` distribution
    blocks (``count``/``mean``/``std``/``min``/``max``); in ``disabled`` mode
    the transformed block mirrors raw exactly.
    """

    raw = [float(value) for value in rewards]
    if raw:
        _ensure_finite(raw, stage="reward_transform raw reward")
    if not raw:
        # Empty input has nothing to transform; skip the mode dispatch so
        # standardize never calls np.mean/np.std on a zero-length array
        # (which emits harmless but noisy RuntimeWarnings).
        transformed = []
    elif config.mode == "disabled":
        transformed = list(raw)
    elif config.mode == "clip":
        transformed = _clip(raw, config.clip_min, config.clip_max)
    else:  # standardize
        transformed = _standardize(raw, config.eps)
    summary = {
        "mode": config.mode,
        "raw": _distribution_stats(raw),
        "transformed": _distribution_stats(transformed),
    }
    return transformed, summary


def _clip(values: list[float], lo: float | None, hi: float | None) -> list[float]:
    arr = np.clip(np.asarray(values, dtype=np.float64), lo, hi)
    return arr.tolist()


def _standardize(values: list[float], eps: float) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean()
    std = arr.std()
    # Constant batches have zero std; using std=0 would divide by ``eps`` alone
    # and silently amplify rounding noise. Treating zero std as unit scale keeps
    # the result at exactly zero (r - mean == 0) and finite.
    scale = std + eps if std > 0 else 1.0
    return ((arr - mean) / scale).tolist()


def _distribution_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _ensure_finite(values: list[float], *, stage: str) -> None:
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            raise ValueError(
                f"{stage} contains a non-finite value at index {idx}; "
                "ensure the reward_fn returns finite floats or enable reward_transform clip"
            )


__all__ = ["RewardTransformConfig", "RewardTransformMode", "transform_rewards"]