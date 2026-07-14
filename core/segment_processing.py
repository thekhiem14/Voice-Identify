from __future__ import annotations

import re
import math
from collections import Counter, defaultdict
from typing import Iterable, Optional

from core.models import ClusterIdentity, DiarizationSegment, SegmentIdentity, TimeSlice, VoiceProfile


def cluster_key(label: str) -> str:
    """Return a forgiving key so `1` also addresses `SPEAKER_01`."""
    value = str(label).strip()
    match = re.search(r"(\d+)$", value)
    return str(int(match.group(1))) if match else value.casefold()


def cluster_sort_key(label: str) -> tuple[int, int | str]:
    key = cluster_key(label)
    return (0, int(key)) if key.isdigit() else (1, key)


def resolve_cluster_mapping(
    labels: Iterable[str], mapping: dict[str, str], resolve_targets: bool = True
) -> dict[str, str]:
    known = list(dict.fromkeys(str(label) for label in labels))
    by_key = {cluster_key(label): label for label in known}
    resolved: dict[str, str] = {}
    for source, target in mapping.items():
        source_label = source if source in known else by_key.get(cluster_key(source))
        if source_label is None:
            continue
        target_label = (
            target
            if not resolve_targets or target in known
            else by_key.get(cluster_key(target), target)
        )
        resolved[source_label] = target_label
    return resolved


def remap_clusters(
    segments: Iterable[DiarizationSegment], mapping: dict[str, str]
) -> list[DiarizationSegment]:
    items = list(segments)
    resolved = resolve_cluster_mapping((item.cluster for item in items), mapping)
    return [
        DiarizationSegment(
            cluster=resolved.get(item.cluster, item.cluster),
            original_cluster=item.original_cluster or item.cluster,
            start=item.start,
            end=item.end,
            duration=item.duration,
        )
        for item in items
    ]


def merge_segments(
    segments: Iterable[DiarizationSegment],
    max_gap: float = 2.0,
    max_merged_duration: float = 30.0,
) -> list[DiarizationSegment]:
    """Merge consecutive same-cluster turns separated by less than max_gap."""
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    if not ordered:
        return []
    merged = [DiarizationSegment(**ordered[0].to_dict())]
    for item in ordered[1:]:
        previous = merged[-1]
        gap = item.start - previous.end
        combined_duration = max(previous.end, item.end) - previous.start
        if (
            item.cluster == previous.cluster
            and gap < max_gap
            and combined_duration <= max_merged_duration
        ):
            previous.end = round(max(previous.end, item.end), 3)
            previous.duration = round(previous.end - previous.start, 3)
        else:
            merged.append(DiarizationSegment(**item.to_dict()))
    return merged


def split_long_segments(
    segments: Iterable[DiarizationSegment], max_duration: float = 8.0
) -> list[DiarizationSegment]:
    """Create bounded identity/ASR windows without changing diarization clusters."""
    if max_duration <= 0:
        raise ValueError("max_duration must be positive.")
    output: list[DiarizationSegment] = []
    for item in segments:
        if item.duration <= max_duration:
            output.append(DiarizationSegment(**item.to_dict()))
            continue
        window_count = max(2, math.ceil(item.duration / max_duration))
        window_duration = item.duration / window_count
        for index in range(window_count):
            start = item.start + index * window_duration
            end = item.end if index == window_count - 1 else item.start + (index + 1) * window_duration
            output.append(
                DiarizationSegment(
                    cluster=item.cluster,
                    original_cluster=item.original_cluster,
                    start=start,
                    end=end,
                    duration=end - start,
                )
            )
    return output


def select_clean_slices(
    cluster: str,
    segments: Iterable[DiarizationSegment],
    target_seconds: float = 10.0,
    min_slice_seconds: float = 0.5,
) -> list[TimeSlice]:
    """Pick longest speech portions after subtracting other-speaker overlaps."""
    items = list(segments)
    candidates: list[TimeSlice] = []
    for item in items:
        if item.cluster != cluster:
            continue
        overlaps = [
            TimeSlice(max(item.start, other.start), min(item.end, other.end))
            for other in items
            if other.cluster != cluster
            and other.start < item.end
            and other.end > item.start
        ]
        clean = _subtract_intervals(TimeSlice(item.start, item.end), overlaps)
        candidates.extend(part for part in clean if part.duration >= min_slice_seconds)

    selected: list[TimeSlice] = []
    remaining = max(0.0, target_seconds)
    for part in sorted(candidates, key=lambda value: value.duration, reverse=True):
        if remaining <= 1e-6:
            break
        duration = min(part.duration, remaining)
        if duration < min_slice_seconds:
            break
        selected.append(TimeSlice(round(part.start, 3), round(part.start + duration, 3)))
        remaining -= duration
    return selected


def smooth_identities(
    identities: list[SegmentIdentity],
    profiles: list[VoiceProfile],
    cluster_references: Optional[dict[str, ClusterIdentity]] = None,
    hard_overrides: Optional[dict[str, str]] = None,
    mixed_secondary_ratio: float = 0.20,
    mixed_secondary_duration: float = 1.5,
    min_evidence_coverage: float = 0.35,
) -> tuple[list[SegmentIdentity], dict[str, ClusterIdentity]]:
    """Majority vote per cluster, with explicit overrides taking precedence."""
    cluster_references = cluster_references or {}
    hard_overrides = hard_overrides or {}
    profile_by_id = {profile.speaker_id: profile for profile in profiles}
    profile_by_name = {profile.display_name.casefold(): profile for profile in profiles}

    grouped: dict[str, list[SegmentIdentity]] = defaultdict(list)
    for identity in identities:
        grouped[identity.cluster].append(identity)
    ordered_clusters = sorted(grouped, key=cluster_sort_key)
    provisional_names = {
        cluster: f"Speaker {index + 1}" for index, cluster in enumerate(ordered_clusters)
    }
    resolved_overrides = resolve_cluster_mapping(
        grouped.keys(), hard_overrides, resolve_targets=False
    )

    summaries: dict[str, ClusterIdentity] = {}
    for cluster, members in grouped.items():
        override_value = resolved_overrides.get(cluster)
        if override_value is not None:
            profile = profile_by_id.get(override_value) or profile_by_name.get(
                override_value.casefold()
            )
            speaker_id = profile.speaker_id if profile else None
            speaker = profile.display_name if profile else override_value
            chosen = ClusterIdentity(
                cluster=cluster,
                speaker_id=speaker_id,
                speaker=speaker,
                provisional_speaker=provisional_names[cluster],
                score=None,
                status="matched",
                source="hard_override",
            )
        else:
            matched = [member for member in members if member.raw_speaker_id]
            vote_counts = Counter(member.raw_speaker_id for member in matched)
            if vote_counts:
                duration_by_id: dict[str, float] = defaultdict(float)
                score_by_id: dict[str, float] = defaultdict(float)
                for member in matched:
                    duration_by_id[member.raw_speaker_id] += member.duration
                    score_by_id[member.raw_speaker_id] += member.raw_score or 0.0
                eligible_duration = sum(
                    member.duration
                    for member in members
                    if member.raw_status not in {"too_short", "embedding_error"}
                )
                matched_duration = sum(duration_by_id.values())
                evidence_coverage = matched_duration / max(1e-9, eligible_duration)
                winner_id = max(
                    vote_counts,
                    key=lambda value: (
                        vote_counts[value],
                        duration_by_id[value],
                        score_by_id[value],
                    ),
                )
                winner_members = [m for m in matched if m.raw_speaker_id == winner_id]
                profile = profile_by_id[winner_id]
                total_duration = sum(duration_by_id.values())
                winner_ratio = duration_by_id[winner_id] / max(1e-9, total_duration)
                secondary_ids = [value for value in vote_counts if value != winner_id]
                mixed = any(
                    duration_by_id[value] / max(1e-9, total_duration)
                    >= mixed_secondary_ratio
                    and (
                        vote_counts[value] >= 2
                        or duration_by_id[value] >= mixed_secondary_duration
                    )
                    for value in secondary_ids
                )
                if evidence_coverage < min_evidence_coverage:
                    mapped_reference = cluster_references.get(cluster)
                    mapped_speaker_id = (
                        mapped_reference.speaker_id if mapped_reference else None
                    )
                    mapped_speaker = (
                        mapped_reference.speaker
                        if mapped_reference and mapped_reference.speaker_id
                        else "Unknown"
                    )
                    candidate_ids = sorted(
                        vote_counts,
                        key=lambda value: (duration_by_id[value], vote_counts[value]),
                        reverse=True,
                    )
                    chosen = ClusterIdentity(
                        cluster=cluster,
                        speaker_id=mapped_speaker_id,
                        speaker=mapped_speaker,
                        provisional_speaker=provisional_names[cluster],
                        score=mapped_reference.score if mapped_reference else None,
                        status="weak_evidence",
                        source=(
                            mapped_reference.source
                            if mapped_speaker_id and mapped_reference
                            else "segment_level_only"
                        ),
                        votes={key: int(value) for key, value in vote_counts.items()},
                        dominant_ratio=round(winner_ratio, 4),
                        evidence_coverage=round(evidence_coverage, 4),
                        candidate_speakers=[
                            profile_by_id[value].display_name for value in candidate_ids
                        ],
                        profile_scores=(
                            mapped_reference.profile_scores if mapped_reference else {}
                        ),
                    )
                elif mixed:
                    candidate_ids = sorted(
                        vote_counts,
                        key=lambda value: (duration_by_id[value], vote_counts[value]),
                        reverse=True,
                    )
                    candidate_names = [profile_by_id[value].display_name for value in candidate_ids]
                    chosen = ClusterIdentity(
                        cluster=cluster,
                        speaker_id=None,
                        speaker="Mixed: " + ", ".join(candidate_names),
                        provisional_speaker=provisional_names[cluster],
                        score=None,
                        status="mixed_cluster",
                        source="segment_level_mixed",
                        votes={key: int(value) for key, value in vote_counts.items()},
                        dominant_ratio=round(winner_ratio, 4),
                        evidence_coverage=round(evidence_coverage, 4),
                        candidate_speakers=candidate_names,
                    )
                else:
                    chosen = ClusterIdentity(
                        cluster=cluster,
                        speaker_id=winner_id,
                        speaker=profile.display_name,
                        provisional_speaker=provisional_names[cluster],
                        score=round(
                            sum((m.raw_score or 0.0) * m.duration for m in winner_members)
                            / max(1e-9, sum(m.duration for m in winner_members)),
                            4,
                        ),
                        status="matched",
                        source="majority_vote",
                        votes={key: int(value) for key, value in vote_counts.items()},
                        dominant_ratio=round(winner_ratio, 4),
                        evidence_coverage=round(evidence_coverage, 4),
                        candidate_speakers=[profile.display_name],
                    )
            elif cluster in cluster_references and cluster_references[cluster].speaker_id:
                chosen = cluster_references[cluster]
                chosen.source = "cluster_enrollment"
            elif not profiles:
                chosen = ClusterIdentity(
                    cluster=cluster,
                    speaker_id=None,
                    speaker=provisional_names[cluster],
                    provisional_speaker=provisional_names[cluster],
                    score=None,
                    status="diarization_only",
                    source="diarization_only",
                )
            else:
                reference = cluster_references.get(cluster)
                chosen = ClusterIdentity(
                    cluster=cluster,
                    speaker_id=None,
                    speaker="Unknown",
                    provisional_speaker=provisional_names[cluster],
                    score=reference.score if reference else None,
                    status="unknown",
                    source="no_match",
                    clean_audio_seconds=(reference.clean_audio_seconds if reference else 0.0),
                    enrollment_file=(reference.enrollment_file if reference else None),
                )

        reference = cluster_references.get(cluster)
        if reference:
            chosen.clean_audio_seconds = reference.clean_audio_seconds
            chosen.enrollment_file = reference.enrollment_file
        chosen.provisional_speaker = provisional_names[cluster]
        summaries[cluster] = chosen

        if chosen.status == "mixed_cluster":
            _apply_mixed_cluster_labels(members, provisional_names[cluster])
            continue

        if chosen.status == "weak_evidence":
            _apply_segment_level_labels(members, provisional_names[cluster])
            continue

        for member in members:
            member.provisional_speaker = provisional_names[cluster]
            if chosen.speaker_id is not None or chosen.source == "hard_override":
                member.speaker_id = chosen.speaker_id
                member.speaker = chosen.speaker
                member.score = chosen.score if chosen.score is not None else member.raw_score
                member.status = "matched"
                member.source = chosen.source
            elif chosen.source == "diarization_only":
                member.speaker_id = None
                member.speaker = chosen.speaker
                member.score = None
                member.status = "diarization_only"
                member.source = "diarization_only"
            else:
                member.speaker_id = member.raw_speaker_id
                member.speaker = member.raw_speaker
                member.score = member.raw_score
                member.status = member.raw_status
                member.source = "segment_verification"
    return identities, summaries


def reconcile_speaker_counts(
    identities: list[SegmentIdentity],
    summaries: dict[str, ClusterIdentity],
    profiles: list[VoiceProfile],
) -> dict:
    profile_to_clusters: dict[str, list[str]] = defaultdict(list)
    for item in identities:
        if item.speaker_id and item.cluster not in profile_to_clusters[item.speaker_id]:
            profile_to_clusters[item.speaker_id].append(item.cluster)
    for cluster, summary in summaries.items():
        if summary.speaker_id and cluster not in profile_to_clusters[summary.speaker_id]:
            profile_to_clusters[summary.speaker_id].append(cluster)
    profile_ids = {profile.speaker_id for profile in profiles}
    seen_ids = set(profile_to_clusters)
    n_profiles = len(profiles)
    k_clusters = len(summaries)
    if not profiles:
        relationship = "no_profiles"
    elif not summaries:
        relationship = "no_speech"
    elif k_clusters > n_profiles:
        relationship = "more_clusters_than_profiles"
    elif k_clusters < n_profiles:
        relationship = "fewer_clusters_than_profiles"
    else:
        relationship = "equal_counts"
    return {
        "registered_profiles": n_profiles,
        "diarized_clusters": k_clusters,
        "relationship": relationship,
        "identified_profiles": sorted(seen_ids),
        "profiles_not_observed": sorted(profile_ids - seen_ids),
        "profile_to_clusters": dict(profile_to_clusters),
        "cluster_speakers": {
            cluster: value.speaker
            for cluster, value in summaries.items()
            if value.speaker_id and value.speaker != "Unknown"
        },
        "split_identity_candidates": {
            key: value for key, value in profile_to_clusters.items() if len(value) > 1
        },
        "mixed_clusters": sorted(
            cluster for cluster, value in summaries.items() if value.status == "mixed_cluster"
        ),
        "weak_evidence_clusters": sorted(
            cluster for cluster, value in summaries.items() if value.status == "weak_evidence"
        ),
        "unknown_clusters": sorted(
            cluster
            for cluster, value in summaries.items()
            if value.status in {"unknown", "weak_evidence"}
        ),
        "diarization_only": not bool(profiles),
        "note": (
            "n profile là tập người có thể nhận diện, không nhất thiết là số người thực sự "
            "xuất hiện; vì vậy n != k là tín hiệu cần kiểm tra, không tự động là lỗi."
        ),
    }


def speaker_stats(segments: Iterable[DiarizationSegment]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"duration": 0.0, "segments": 0}
    )
    for item in segments:
        stats[item.cluster]["duration"] = round(
            stats[item.cluster]["duration"] + item.duration, 3
        )
        stats[item.cluster]["segments"] += 1
    return dict(stats)


def _subtract_intervals(base: TimeSlice, cuts: list[TimeSlice]) -> list[TimeSlice]:
    if not cuts:
        return [base]
    merged: list[TimeSlice] = []
    for cut in sorted(cuts, key=lambda item: item.start):
        start = max(base.start, cut.start)
        end = min(base.end, cut.end)
        if end <= start:
            continue
        if merged and start <= merged[-1].end:
            merged[-1] = TimeSlice(merged[-1].start, max(merged[-1].end, end))
        else:
            merged.append(TimeSlice(start, end))
    output: list[TimeSlice] = []
    cursor = base.start
    for cut in merged:
        if cut.start > cursor:
            output.append(TimeSlice(cursor, cut.start))
        cursor = max(cursor, cut.end)
    if cursor < base.end:
        output.append(TimeSlice(cursor, base.end))
    return output


def _apply_mixed_cluster_labels(
    members: list[SegmentIdentity], provisional_speaker: str
) -> None:
    ordered = sorted(members, key=lambda item: item.start)
    for index, member in enumerate(ordered):
        member.provisional_speaker = provisional_speaker
        if member.raw_speaker_id:
            member.speaker_id = member.raw_speaker_id
            member.speaker = member.raw_speaker
            member.score = member.raw_score
            member.status = "matched"
            member.source = "segment_level_mixed"
            continue
        previous = next(
            (item for item in reversed(ordered[:index]) if item.raw_speaker_id), None
        )
        following = next(
            (item for item in ordered[index + 1 :] if item.raw_speaker_id), None
        )
        if previous and following and previous.raw_speaker_id == following.raw_speaker_id:
            member.speaker_id = previous.raw_speaker_id
            member.speaker = previous.raw_speaker
            member.score = min(previous.raw_score or 0.0, following.raw_score or 0.0)
            member.status = "matched"
            member.source = "neighbor_smoothing"
        else:
            member.speaker_id = None
            member.speaker = "Unknown"
            member.score = member.raw_score
            member.status = member.raw_status
            member.source = "mixed_cluster_unresolved"


def _apply_segment_level_labels(
    members: list[SegmentIdentity], provisional_speaker: str
) -> None:
    """Keep strong per-segment matches without spreading them across a cluster."""
    for member in members:
        member.provisional_speaker = provisional_speaker
        member.speaker_id = member.raw_speaker_id
        member.speaker = member.raw_speaker
        member.score = member.raw_score
        member.status = member.raw_status
        member.source = (
            "segment_verification" if member.raw_speaker_id else "weak_cluster_unresolved"
        )
