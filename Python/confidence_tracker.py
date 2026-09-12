#!/usr/bin/env python3
"""
confidence_tracker.py
─────────────────────
Tracks per-iteration training metrics for the StellaV agent loop.

Stores a log of every iteration and can:
  • Report current / best / trend.
  • Identify which topics produced the worst matches (for Qwen refinement).
  • Detect stagnation (no improvement for N iters) so the agent can
    increase dataset size or shift topic emphasis.
  • Persist history to a JSON log file.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

LOG_PATH = Path("agent_confidence_log.json")
TARGET_CONF  = 0.95
STAGNATION_N = 5          # consider stagnant if best_conf unchanged for N iters


@dataclass
class IterRecord:
    iteration: int
    timestamp: float
    topic: str
    dataset_size: int
    avg_conf: float
    match_rate: float
    reached_target: bool = False


@dataclass
class ConfidenceTracker:
    records: list[IterRecord] = field(default_factory=list)
    target_conf: float = TARGET_CONF

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path = LOG_PATH) -> None:
        data = [asdict(r) for r in self.records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path = LOG_PATH) -> "ConfidenceTracker":
        tracker = cls()
        if not path.exists():
            return tracker
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tracker.records = [IterRecord(**r) for r in data]
        except Exception:
            pass
        return tracker

    # ── record ───────────────────────────────────────────────────────────────

    def record(
        self,
        iteration: int,
        topic: str,
        dataset_size: int,
        avg_conf: float,
        match_rate: float,
    ) -> IterRecord:
        rec = IterRecord(
            iteration=iteration,
            timestamp=time.time(),
            topic=topic,
            dataset_size=dataset_size,
            avg_conf=avg_conf,
            match_rate=match_rate,
            reached_target=(avg_conf >= self.target_conf),
        )
        self.records.append(rec)
        self.save()
        return rec

    # ── queries ──────────────────────────────────────────────────────────────

    @property
    def best_conf(self) -> float:
        return max((r.avg_conf for r in self.records), default=0.0)

    @property
    def latest(self) -> Optional[IterRecord]:
        return self.records[-1] if self.records else None

    def is_stagnant(self) -> bool:
        """True if last STAGNATION_N iters haven't improved best_conf."""
        if len(self.records) < STAGNATION_N:
            return False
        recent = self.records[-STAGNATION_N:]
        return max(r.avg_conf for r in recent) <= self.records[-(STAGNATION_N + 1)].avg_conf if len(self.records) > STAGNATION_N else False

    def worst_topics(self) -> list[str]:
        """Topics with lowest average match_rate (candidates for refinement)."""
        topic_stats: dict[str, list[float]] = {}
        for r in self.records:
            topic_stats.setdefault(r.topic, []).append(r.match_rate)
        averages = {t: sum(v) / len(v) for t, v in topic_stats.items()}
        return sorted(averages, key=averages.get)  # worst first

    # ── display ──────────────────────────────────────────────────────────────

    def report(self) -> str:
        if not self.records:
            return "[tracker] No iterations recorded yet."
        lat = self.records[-1]
        delta = ""
        if len(self.records) > 1:
            prev = self.records[-2]
            d    = lat.avg_conf - prev.avg_conf
            delta = f"  Δconf={d:+.4f}"
        bar_filled = int(lat.avg_conf * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        pct = lat.avg_conf * 100
        return (
            f"\n{'─'*60}\n"
            f"  Iteration : {lat.iteration}\n"
            f"  Topic     : {lat.topic}\n"
            f"  Dataset   : {lat.dataset_size} samples\n"
            f"  Confidence: [{bar}] {pct:5.1f}%{delta}\n"
            f"  Match rate: {lat.match_rate*100:.1f}%\n"
            f"  Best ever : {self.best_conf*100:.1f}%   Target: {self.target_conf*100:.0f}%\n"
            f"  Stagnant  : {'YES — expanding dataset' if self.is_stagnant() else 'No'}\n"
            f"{'─'*60}"
        )
