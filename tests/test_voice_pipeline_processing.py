from __future__ import annotations

import numpy as np

from core.models import DiarizationSegment, SegmentIdentity, VoiceProfile
from core.segment_processing import (
    merge_segments,
    remap_clusters,
    select_clean_slices,
    split_long_segments,
    smooth_identities,
)


def segment(cluster: str, start: float, end: float) -> DiarizationSegment:
    return DiarizationSegment(cluster, start, end, end - start)


def test_numeric_cluster_mapping_and_merge() -> None:
    original = [segment("SPEAKER_00", 0, 2), segment("SPEAKER_01", 3, 4)]
    remapped = remap_clusters(original, {"1": "0"})
    merged = merge_segments(remapped, max_gap=2.0)

    assert [item.cluster for item in remapped] == ["SPEAKER_00", "SPEAKER_00"]
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0.0, 4.0)


def test_merge_requires_gap_strictly_below_two_seconds() -> None:
    result = merge_segments(
        [segment("SPEAKER_00", 0, 1), segment("SPEAKER_00", 3, 4)],
        max_gap=2.0,
    )
    assert len(result) == 2


def test_clean_enrollment_slices_remove_overlap() -> None:
    segments = [segment("A", 0, 6), segment("B", 2, 4)]
    selected = select_clean_slices("A", segments, target_seconds=4, min_slice_seconds=0.5)

    assert sum(item.duration for item in selected) == 4.0
    assert all(item.end <= 2 or item.start >= 4 for item in selected)


def test_long_under_cluster_segment_is_split_for_identity() -> None:
    windows = split_long_segments([segment("A", 0, 21)], max_duration=8)

    assert len(windows) == 3
    assert all(item.duration <= 8 for item in windows)
    assert windows[0].start == 0
    assert windows[-1].end == 21
    assert all(item.cluster == "A" for item in windows)


def test_majority_vote_and_hard_override() -> None:
    profiles = [
        VoiceProfile("a", "An", np.array([1.0, 0.0])),
        VoiceProfile("b", "Binh", np.array([0.0, 1.0])),
    ]
    identities = [
        SegmentIdentity("s0", "0", 0, 2, 2, raw_speaker_id="a", raw_speaker="An", raw_score=0.8, raw_status="matched"),
        SegmentIdentity("s1", "0", 2, 4, 2, raw_speaker_id="a", raw_speaker="An", raw_score=0.7, raw_status="matched"),
        SegmentIdentity("s2", "0", 4, 5, 1, raw_speaker_id="b", raw_speaker="Binh", raw_score=0.9, raw_status="matched"),
        SegmentIdentity("s3", "1", 5, 6, 1, raw_status="too_short"),
    ]

    smoothed, summaries = smooth_identities(
        identities,
        profiles,
        hard_overrides={"1": "Hung"},
    )

    assert all(item.speaker == "An" for item in smoothed[:3])
    assert summaries["0"].source == "majority_vote"
    assert smoothed[3].speaker == "Hung"
    assert summaries["1"].source == "hard_override"
