"""Flap suppression — BGP route-flap-damping-style penalty + exponential decay.

Alert fatigue: teams desensitised by a barrage of non-actionable alerts. A
chronically *flapping* entity must not dominate the operator queue. We reuse the
exact BGP/MPLS route-flap-damping logic for alerting (research 07 A2.3):

  * maintain a per-entity **flap penalty** that **increments** on each state change
    / re-fire and **decays exponentially** with a configurable half-life;
  * while ``penalty > suppress_threshold`` the entity is **suppressed** (its risk
    is demoted, not dropped — it remains as evidence);
  * once the penalty decays below ``reuse_threshold`` the entity is re-enabled.

The penalty also yields a multiplicative **risk demotion factor** in [demote_floor, 1]
so a flapping incident's calibrated risk is scaled down smoothly rather than
binary on/off. Deterministic and O(1) per update; pure-Python/NumPy, offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class _FlapState:
    penalty: float = 0.0
    last_update: datetime | None = None
    suppressed: bool = False


@dataclass
class FlapSuppressor:
    """Per-entity flap penalty with exponential decay (RFD-style).

    Parameters mirror BGP route-flap damping:
      * ``penalty_increment`` — penalty added per flap/re-fire (default 1.0).
      * ``half_life_seconds`` — time for the penalty to halve (default 900s/15min).
      * ``suppress_threshold`` — suppress while penalty exceeds this (default 3.0).
      * ``reuse_threshold`` — re-enable once penalty falls below this (default 1.0).
      * ``max_penalty`` — ceiling so penalty cannot grow unbounded (default 16.0).
      * ``demote_floor`` — minimum risk multiplier for a fully-suppressed entity.
    """

    penalty_increment: float = 1.0
    half_life_seconds: float = 900.0
    suppress_threshold: float = 3.0
    reuse_threshold: float = 1.0
    max_penalty: float = 16.0
    demote_floor: float = 0.1
    _states: dict[str, _FlapState] = field(default_factory=dict)

    # -- core update --------------------------------------------------------
    def _decay(self, state: _FlapState, now: datetime) -> None:
        if state.last_update is not None and state.penalty > 0:
            dt = (now - state.last_update).total_seconds()
            if dt > 0:
                # exponential decay: p *= 0.5 ** (dt / half_life)
                state.penalty *= math.pow(0.5, dt / self.half_life_seconds)
        state.last_update = now

    def observe(self, entity_id: str, *, now: datetime | None = None, flaps: int = 1) -> float:
        """Register ``flaps`` state-changes for an entity; return current penalty.

        Decays the existing penalty to ``now`` first, then adds the increment(s),
        and updates the suppress/reuse hysteresis state.
        """
        ts = _as_utc(now) if now else datetime.now(timezone.utc)
        state = self._states.setdefault(entity_id, _FlapState())
        self._decay(state, ts)
        state.penalty = min(self.max_penalty, state.penalty + self.penalty_increment * flaps)
        # hysteresis: enter suppression above suppress_threshold, leave below reuse.
        if state.penalty >= self.suppress_threshold:
            state.suppressed = True
        elif state.penalty < self.reuse_threshold:
            state.suppressed = False
        return round(state.penalty, 6)

    def penalty_of(self, entity_id: str, *, now: datetime | None = None) -> float:
        """Current decayed penalty for an entity without registering a new flap."""
        state = self._states.get(entity_id)
        if state is None:
            return 0.0
        ts = _as_utc(now) if now else datetime.now(timezone.utc)
        # decay a *copy* of the value (don't mutate last_update on a pure read).
        if state.last_update is not None and state.penalty > 0:
            dt = (ts - state.last_update).total_seconds()
            if dt > 0:
                return round(state.penalty * math.pow(0.5, dt / self.half_life_seconds), 6)
        return round(state.penalty, 6)

    def is_suppressed(self, entity_id: str, *, now: datetime | None = None) -> bool:
        """Whether the entity is currently suppressed (penalty above the band)."""
        p = self.penalty_of(entity_id, now=now)
        state = self._states.get(entity_id)
        if state is None:
            return False
        # honour hysteresis: stay suppressed until decayed below reuse_threshold.
        if state.suppressed and p >= self.reuse_threshold:
            return True
        if p >= self.suppress_threshold:
            return True
        return False

    def demotion_factor(self, entity_id: str, *, now: datetime | None = None) -> float:
        """Multiplicative risk demotion in [demote_floor, 1] for a flapping entity.

        1.0 when the entity is calm; falls toward ``demote_floor`` as the penalty
        climbs above the suppress threshold. Above ``suppress_threshold`` the
        factor interpolates down to ``demote_floor`` at ``max_penalty``.
        """
        p = self.penalty_of(entity_id, now=now)
        if p < self.reuse_threshold:
            return 1.0
        if p < self.suppress_threshold:
            # linear taper from 1.0 (reuse) to 0.7 (suppress threshold).
            frac = (p - self.reuse_threshold) / (self.suppress_threshold - self.reuse_threshold)
            return round(1.0 - 0.3 * frac, 4)
        # above suppress threshold: taper 0.7 → demote_floor at max_penalty.
        span = max(self.max_penalty - self.suppress_threshold, 1e-6)
        frac = min(1.0, (p - self.suppress_threshold) / span)
        return round(max(self.demote_floor, 0.7 - (0.7 - self.demote_floor) * frac), 4)

    def reset(self, entity_id: str | None = None) -> None:
        """Clear flap state for one entity (or all)."""
        if entity_id is None:
            self._states.clear()
        else:
            self._states.pop(entity_id, None)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


__all__ = ["FlapSuppressor"]
