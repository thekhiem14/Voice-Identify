from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class DiarizationSegment:
    cluster: str
    start: float
    end: float
    duration: float
    original_cluster: Optional[str] = None

    def __post_init__(self) -> None:
        self.start = round(float(self.start), 3)
        self.end = round(float(self.end), 3)
        self.duration = round(max(0.0, self.end - self.start), 3)
        if self.original_cluster is None:
            self.original_cluster = self.cluster

    @property
    def speaker(self) -> str:
        """Compatibility alias for older callers."""
        return self.cluster

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimeSlice:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return round(max(0.0, self.end - self.start), 3)

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end, "duration": self.duration}


@dataclass
class SegmentIdentity:
    segment_id: str
    cluster: str
    start: float
    end: float
    duration: float
    speaker_id: Optional[str] = None
    speaker: str = "Unknown"
    provisional_speaker: str = ""
    score: Optional[float] = None
    status: str = "unknown"
    source: str = "segment_verification"
    scores: dict[str, float] = field(default_factory=dict)
    second_best_speaker_id: Optional[str] = None
    second_best_score: Optional[float] = None
    score_margin: Optional[float] = None
    raw_speaker_id: Optional[str] = None
    raw_speaker: str = "Unknown"
    raw_score: Optional[float] = None
    raw_status: str = "unknown"
    text: str = ""
    asr_status: str = "pending"
    asr_error: Optional[str] = None
    identity_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterIdentity:
    cluster: str
    speaker_id: Optional[str]
    speaker: str
    provisional_speaker: str
    score: Optional[float]
    status: str
    source: str
    votes: dict[str, int] = field(default_factory=dict)
    clean_audio_seconds: float = 0.0
    enrollment_file: Optional[str] = None
    dominant_ratio: Optional[float] = None
    evidence_coverage: Optional[float] = None
    candidate_speakers: list[str] = field(default_factory=list)
    profile_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceProfile:
    speaker_id: str
    display_name: str
    embedding: Any = field(repr=False)
    sample_files: list[str] = field(default_factory=list)
    source: str = "database"

    def metadata(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "sample_files": self.sample_files,
            "source": self.source,
        }
