from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

@dataclass
class DetectorResult:
    video_id: str
    label: int | None
    score: float
    features: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self):
        d = asdict(self)
        d.update(self.features)
        d.pop("features", None)
        return d

class DetectorAdapter:
    name = "base"
    version = "0.1"

    def score_video(self, path: str | Path, **kwargs) -> DetectorResult:
        raise NotImplementedError
