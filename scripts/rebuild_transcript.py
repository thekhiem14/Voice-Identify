from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config.settings import SETTINGS
from core.models import ClusterIdentity, SegmentIdentity, VoiceProfile
from core.pipeline import (
    _build_public_transcript_segments,
    _build_transcript_text,
)
from core.segment_processing import reconcile_speaker_counts, smooth_identities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tạo transcript.json sạch từ một job đã chạy."
    )
    parser.add_argument("job_dir", type=Path)
    parser.add_argument(
        "--speaker-map",
        action="append",
        default=[],
        metavar="LABEL=NAME",
        help="Đổi nhãn output, ví dụ --speaker-map S1=A",
    )
    args = parser.parse_args()
    job_dir = args.job_dir.resolve()

    result = _read(job_dir / "result.json")
    identities = [SegmentIdentity(**row) for row in result.get("segments", [])]
    profiles = [
        VoiceProfile(
            speaker_id=row["speaker_id"],
            display_name=row["display_name"],
            embedding=np.zeros(1, dtype=np.float32),
            sample_files=row.get("sample_files", []),
            source=row.get("source", "database"),
        )
        for row in result.get("profiles", [])
    ]
    references = {
        cluster: ClusterIdentity(**row)
        for cluster, row in result.get("clusters", {}).items()
    }
    identities, summaries = smooth_identities(
        identities,
        profiles,
        cluster_references=references,
        min_evidence_coverage=SETTINGS.cluster_vote_min_evidence_coverage,
    )
    reconciliation = reconcile_speaker_counts(identities, summaries, profiles)
    payload = {
        "segments": _build_public_transcript_segments(
            identities,
            reconciliation,
            max_gap_seconds=2.0,
            max_segment_seconds=20.0,
        )
    }
    speaker_map = _parse_speaker_map(args.speaker_map)
    for segment in payload["segments"]:
        segment["speaker"] = speaker_map.get(segment["speaker"], segment["speaker"])
    destination = job_dir / "transcript.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (job_dir / "transcript.txt").write_text(
        _build_transcript_text(payload), encoding="utf-8"
    )
    print(destination)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_speaker_map(values: list[str]) -> dict[str, str]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Speaker map phải có dạng LABEL=NAME: {value}")
        label, name = value.split("=", 1)
        label, name = label.strip(), name.strip()
        if not label or not name:
            raise ValueError(f"Speaker map không hợp lệ: {value}")
        output[label] = name
    return output


if __name__ == "__main__":
    main()
