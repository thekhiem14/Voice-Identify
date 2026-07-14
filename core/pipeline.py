from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from config.settings import OUTPUT_DIR, SETTINGS, VOICE_DB_PATH, ensure_runtime_dirs
from core.asr_engine import GipformerASR, mark_asr_skipped, mark_asr_unavailable
from core.audio_enhancer import (
    convert_to_16k_mono,
    enhance_audio,
    inspect_audio_quality,
    quick_snr,
)
from core.diarization import run_diarization
from core.models import SegmentIdentity
from core.segment_processing import (
    cluster_key,
    cluster_sort_key,
    merge_segments,
    project_identities_to_segments,
    reconcile_speaker_counts,
    remap_clusters,
    split_long_segments,
    smooth_identities,
    speaker_stats,
)
from core.voice_id import (
    CAMPPlusEmbedder,
    EnrollmentRequest,
    aggregate_cluster_segment_scores,
    assign_cluster_profiles,
    build_cluster_references,
    build_runtime_profiles,
    load_voice_db,
    verify_segments,
)


ProgressCallback = Callable[[int, int, str, str], None]


def _runtime_log(message: str) -> None:
    """Console log used by both CLI and the Flet process."""
    stamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


@dataclass
class PipelineOptions:
    output_dir: str = str(OUTPUT_DIR)
    job_name: Optional[str] = None
    cluster_mapping: dict[str, str] = field(default_factory=dict)
    hard_overrides: dict[str, str] = field(default_factory=dict)
    verification_threshold: float = SETTINGS.verification_threshold
    merge_max_gap_sec: float = SETTINGS.merge_max_gap_sec
    merge_max_duration_sec: float = SETTINGS.merge_max_duration_sec
    cluster_enrollment_target_sec: float = SETTINGS.cluster_enrollment_target_sec
    identity_max_window_sec: float = SETTINGS.identity_max_window_sec
    voice_id_batch_size: int = SETTINGS.voice_id_batch_size
    asr_min_duration_sec: float = SETTINGS.asr_min_duration_sec
    asr_padding_sec: float = SETTINGS.asr_padding_sec
    asr_provider: str = SETTINGS.asr_provider
    asr_batch_size: int = SETTINGS.asr_batch_size
    include_saved_profiles: bool = True
    enhance_audio: bool = False
    skip_identify: bool = False
    skip_asr: bool = False
    allow_partial_results: bool = True
    hf_token: Optional[str] = SETTINGS.hf_token
    device: Optional[str] = SETTINGS.device
    speakerlab_root: Optional[str] = SETTINGS.speakerlab_root
    diarizen_model_id: str = SETTINGS.diarizen_model_id
    campplus_model_id: str = SETTINGS.campplus_model_id


class PipelineExecutionError(RuntimeError):
    def __init__(self, message: str, failure_path: Path) -> None:
        super().__init__(message)
        self.failure_path = failure_path


def run_pipeline(
    audio_path: str | Path,
    options: Optional[PipelineOptions] = None,
    enrollments: Optional[list[EnrollmentRequest]] = None,
    progress: Optional[ProgressCallback] = None,
) -> dict[str, Path]:
    """Run the pipeline and always leave a machine-readable failure artifact."""
    ensure_runtime_dirs()
    options = options or PipelineOptions()
    source_audio = Path(audio_path).expanduser().resolve()
    job_id = _job_id(source_audio, options.job_name)
    job_dir = Path(options.output_dir).expanduser().resolve() / job_id
    failure_path = job_dir / "failure.json"
    started = time.perf_counter()
    try:
        return _execute_pipeline(
            source_audio,
            job_id,
            job_dir,
            options,
            enrollments or [],
            progress,
            started,
        )
    except PipelineExecutionError:
        raise
    except Exception as exc:
        payload = {
            "schema_version": "1.1",
            "status": "failed",
            "job_id": job_id,
            "source_audio": str(source_audio),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "options": _safe_options(options),
        }
        try:
            _write_json(failure_path, payload)
        except Exception:
            pass
        raise PipelineExecutionError(
            f"Pipeline thất bại: {exc}. Chi tiết: {failure_path}", failure_path
        ) from exc
    finally:
        # These files are implementation details and can be very large.
        shutil.rmtree(job_dir / "work", ignore_errors=True)


def _execute_pipeline(
    source_audio: Path,
    job_id: str,
    job_dir: Path,
    options: PipelineOptions,
    enrollments: list[EnrollmentRequest],
    progress: Optional[ProgressCallback],
    started: float,
) -> dict[str, Path]:
    if not source_audio.exists() or not source_audio.is_file():
        raise FileNotFoundError(f"Audio input does not exist: {source_audio}")
    _validate_options(options)
    work_dir = job_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    step_times: dict[str, float] = {}
    warnings: list[dict] = []

    def add_warning(code: str, message: str, severity: str = "warning", **context) -> None:
        warnings.append(
            {"code": code, "severity": severity, "message": message, **context}
        )

    def step(number: int, key: str, message: str) -> float:
        if progress:
            progress(number, 8, key, message)
        return time.perf_counter()

    # 1. Preprocessing and signal validation
    tick = step(1, "preprocessing", "Chuẩn hóa và kiểm tra audio mono 16 kHz")
    normalized_audio = convert_to_16k_mono(
        source_audio, work_dir, output_name="meeting_16k.wav", overwrite=True
    )
    active_audio = normalized_audio
    enhancement = None
    if options.enhance_audio:
        try:
            active_audio, enhancement = enhance_audio(
                normalized_audio,
                work_dir,
                device=options.device,
                log=_runtime_log,
            )
        except (ImportError, RuntimeError) as exc:
            add_warning(
                "enhancement_skipped",
                f"Không thể khử nhiễu nên tiếp tục với audio gốc: {exc}",
            )
    audio_quality = inspect_audio_quality(active_audio)
    duration_seconds = float(audio_quality["duration_seconds"])
    snr_db = quick_snr(active_audio)
    if duration_seconds < 0.25:
        add_warning(
            "audio_too_short",
            "Audio ngắn dưới 0.25 giây; không chạy diarization.",
        )
    elif duration_seconds < 1.0:
        add_warning("very_short_audio", "Audio dưới 1 giây nên kết quả không ổn định.")
    if audio_quality["is_silent"]:
        add_warning("silent_audio", "Audio gần như im lặng; trả về timeline rỗng.")
    elif float(audio_quality["rms_dbfs"]) < -45:
        add_warning("low_signal_level", "Mức âm thanh rất nhỏ, voice ID có thể kém ổn định.")
    if float(audio_quality["clipping_ratio"]) > 0.01:
        add_warning("clipped_audio", "Hơn 1% sample bị clipping.")
    if duration_seconds > 4 * 3600:
        add_warning(
            "very_long_audio",
            "Audio dài hơn 4 giờ; nên chia file để tránh thiếu RAM/VRAM.",
        )
    step_times["preprocessing"] = time.perf_counter() - tick

    # 2. Diarization. Silence and unusably short input are intentionally empty.
    tick = step(2, "diarization", "DiariZen đang phân đoạn và gom cụm người nói")
    if audio_quality["is_silent"] or duration_seconds < 0.25:
        raw_segments = []
    else:
        raw_segments = run_diarization(
            active_audio,
            model_id=options.diarizen_model_id,
            hf_token=options.hf_token,
            device=options.device,
            log=_runtime_log,
        )
    _runtime_log(f"[Pipeline] Diarization trả về {len(raw_segments)} segment thô")
    if not raw_segments:
        add_warning("no_speech_detected", "Không phát hiện segment lời nói nào.", "info")
    step_times["diarization"] = time.perf_counter() - tick

    # 3. Mapping validation, refinement and merge
    tick = step(3, "refinement", "Ánh xạ cluster và ghép các lượt nói gần nhau")
    raw_labels = {item.cluster for item in raw_segments}
    _mapping_warnings(options.cluster_mapping, raw_labels, "cluster_mapping", add_warning)
    remapped_segments = remap_clusters(raw_segments, options.cluster_mapping)
    refined_segments = merge_segments(
        remapped_segments,
        max_gap=options.merge_max_gap_sec,
        max_merged_duration=options.merge_max_duration_sec,
    )
    identity_segments = split_long_segments(
        refined_segments, max_duration=options.identity_max_window_sec
    )
    _runtime_log(
        f"[Pipeline] Sau tinh lọc: {len(refined_segments)} segment; "
        f"{len(identity_segments)} cửa sổ dùng riêng cho Voice ID"
    )
    refined_labels = {item.cluster for item in refined_segments}
    _mapping_warnings(options.hard_overrides, refined_labels, "hard_override", add_warning)
    step_times["refinement"] = time.perf_counter() - tick

    # 4. Build usable profiles. Invalid samples/profiles do not discard valid ones.
    profiles = []
    cluster_references = {}
    has_saved_profiles = False
    if options.include_saved_profiles:
        try:
            has_saved_profiles = bool(load_voice_db(VOICE_DB_PATH).get("speakers"))
        except Exception as exc:
            add_warning("invalid_voice_database", str(exc))
    needs_embedder = not options.skip_identify and (
        bool(enrollments) or has_saved_profiles
    )

    tick = step(4, "enrollment", "Tạo mẫu sạch và embedding CAM++")
    embedder = None
    if needs_embedder:
        try:
            embedder = CAMPPlusEmbedder(
                model_id=options.campplus_model_id,
                device=options.device,
                speakerlab_root=options.speakerlab_root,
            )
            profiles, profile_issues = build_runtime_profiles(
                embedder,
                requests=enrollments,
                voice_db_path=VOICE_DB_PATH,
                workspace=work_dir / "run_samples",
                include_saved=options.include_saved_profiles,
            )
            warnings.extend(profile_issues)
            if profiles and refined_segments:
                cluster_references = build_cluster_references(
                    active_audio,
                    refined_segments,
                    profiles,
                    embedder,
                    work_dir / "cluster_enrollments",
                    threshold=options.verification_threshold,
                    target_seconds=options.cluster_enrollment_target_sec,
                )
                failed_references = [
                    cluster
                    for cluster, reference in cluster_references.items()
                    if reference.status == "embedding_error"
                ]
                if failed_references:
                    add_warning(
                        "cluster_enrollment_errors",
                        "Một số cluster không tạo được enrollment embedding.",
                        clusters=failed_references,
                    )
        except Exception as exc:
            if not options.allow_partial_results:
                raise
            add_warning(
                "identity_stage_unavailable",
                f"Không thể khởi tạo/chuẩn bị CAM++: {exc}",
            )
            embedder = None
            profiles = []
    elif not options.skip_identify:
        add_warning(
            "no_voice_samples",
            "Không có sample hợp lệ; dùng tên tạm Speaker 1, Speaker 2...",
            "info",
        )
    step_times["enrollment"] = time.perf_counter() - tick

    # 5. Per-segment verification
    tick = step(5, "verification", "So khớp cosine từng segment với profile")
    if embedder and profiles:
        identities = verify_segments(
            active_audio,
            identity_segments,
            profiles,
            embedder,
            threshold=options.verification_threshold,
            batch_size=options.voice_id_batch_size,
            log=_runtime_log,
            log_every=SETTINGS.progress_log_every,
        )
        embedding_errors = [
            item.segment_id for item in identities if item.status == "embedding_error"
        ]
        if embedding_errors:
            add_warning(
                "segment_embedding_errors",
                "Một số segment không trích xuất được embedding.",
                segments=embedding_errors,
            )
        ambiguous = [
            item.segment_id
            for item in identities
            if item.score_margin is not None
            and item.score_margin < SETTINGS.verification_ambiguity_warning_margin
        ]
        if ambiguous:
            add_warning(
                "ambiguous_identity_scores",
                f"Best và second-best cosine quá gần; đã áp dụng ngưỡng {options.verification_threshold:.2f} nhưng cần kiểm tra.",
                "info",
                segments=ambiguous,
            )
        cluster_references = aggregate_cluster_segment_scores(
            cluster_references,
            identities,
            top_k=5,
        )
        cluster_references = assign_cluster_profiles(
            cluster_references,
            profiles,
            threshold=options.verification_threshold,
        )
        for cluster, reference in cluster_references.items():
            _runtime_log(
                f"[AutoMap] cluster {cluster} -> {reference.speaker} "
                f"(cosine={reference.score})"
            )
    else:
        identities = _unknown_identities(
            identity_segments,
            "disabled" if options.skip_identify else "no_profiles",
        )
    step_times["verification"] = time.perf_counter() - tick

    # 6. Mixed-cluster-aware smoothing and n/k reconciliation
    tick = step(6, "smoothing", "Phát hiện cluster trộn, vote và hard override")
    identities, cluster_summaries = smooth_identities(
        identities,
        profiles,
        cluster_references=cluster_references,
        hard_overrides=options.hard_overrides,
        mixed_secondary_ratio=SETTINGS.mixed_cluster_secondary_ratio,
        mixed_secondary_duration=SETTINGS.mixed_cluster_secondary_duration_sec,
        min_evidence_coverage=SETTINGS.cluster_vote_min_evidence_coverage,
    )
    reconciliation = reconcile_speaker_counts(identities, cluster_summaries, profiles)
    _reconciliation_warnings(reconciliation, add_warning)
    step_times["smoothing"] = time.perf_counter() - tick

    # Match the notebook's ASR timeline: Gipformer receives merged diarization
    # turns instead of the shorter windows needed by CAM++.
    identity_windows = identities
    identities = project_identities_to_segments(refined_segments, identity_windows)
    for identity_window in identity_windows:
        identity_window.asr_status = "not_applicable_identity_window"
    _runtime_log(
        f"[Pipeline] ASR dùng {len(identities)} segment đã merge "
        "(không cắt theo cửa sổ Voice ID)"
    )

    # Drop the CAM++ model before ASR. If CUDA Gipformer is explicitly
    # selected, also release PyTorch's cache before ONNX reserves its own arena.
    if not options.skip_asr and identities:
        embedder = None
        if options.asr_provider.casefold() == "cuda":
            _release_torch_cuda_memory(options.device, _runtime_log)

    # 7. ASR is recoverable: model-level and segment-level failures stay in output.
    tick = step(7, "asr", "Gipformer đang nhận dạng tiếng Việt theo từng segment")
    if options.skip_asr:
        identities = mark_asr_skipped(identities)
        asr_skipped = len(identities)
    elif not identities:
        asr_skipped = 0
    else:
        try:
            asr = GipformerASR(
                provider=options.asr_provider,
                device=options.device,
                log=_runtime_log,
            )
            identities, asr_skipped = asr.transcribe_segments(
                # Gipformer was trained for difficult/noisy speech. The
                # notebook showed that denoising can distort Vietnamese tones,
                # so ASR always uses only the normalized raw recording.
                normalized_audio,
                identities,
                min_duration=options.asr_min_duration_sec,
                padding=options.asr_padding_sec,
                batch_size=options.asr_batch_size,
                log_every=SETTINGS.progress_log_every,
            )
        except Exception as exc:
            if not options.allow_partial_results:
                raise
            identities = mark_asr_unavailable(identities, str(exc))
            asr_skipped = 0
            add_warning("asr_unavailable", f"Gipformer không khả dụng: {exc}")
    asr_errors = [item.segment_id for item in identities if item.asr_status == "error"]
    if asr_errors:
        add_warning(
            "asr_segment_errors",
            "Một số segment ASR lỗi; các segment còn lại vẫn được giữ.",
            segments=asr_errors,
        )
    step_times["asr"] = time.perf_counter() - tick

    # 8. Structured output
    tick = step(8, "output", "Đang ghi JSON kết quả, warning và artifact")
    paths = {
        "result": job_dir / "result.json",
        "transcript": job_dir / "transcript.json",
        "transcript_text": job_dir / "transcript.txt",
    }
    run_status = "complete" if not warnings else "completed_with_warnings"
    clusters_payload = {
        cluster: summary.to_dict() for cluster, summary in cluster_summaries.items()
    }
    result_payload = {
        "schema_version": "1.1",
        "status": run_status,
        "job_id": job_id,
        "source_audio": str(source_audio),
        "duration_seconds": round(duration_seconds, 3),
        "sample_rate": SETTINGS.sample_rate,
        "verification_threshold": options.verification_threshold,
        "reconciliation": reconciliation,
        "warnings": warnings,
        "clusters": clusters_payload,
        "profiles": [profile.metadata() for profile in profiles],
        "segments": [item.to_dict() for item in identities],
        # Preserve segment-level CAM++ evidence, especially for mixed or
        # under-clustered speakers, without forcing ASR onto short windows.
        "identity_windows": [item.to_dict() for item in identity_windows],
        "audio_quality": audio_quality,
    }
    transcript_payload = {
        "segments": _build_public_transcript_segments(
            identities,
            reconciliation,
            max_gap_seconds=2.0,
            max_segment_seconds=20.0,
        )
    }
    _write_json(paths["transcript"], transcript_payload)
    paths["transcript_text"].write_text(
        _build_transcript_text(transcript_payload), encoding="utf-8"
    )
    step_times["output"] = time.perf_counter() - tick
    metrics_payload = {
        "job_id": job_id,
        "status": run_status,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "step_runtime_seconds": {
            key: round(value, 3) for key, value in step_times.items()
        },
        "audio_quality": audio_quality,
        "snr_db_estimate": round(snr_db, 2),
        "enhancement": enhancement,
        "raw_segments": len(raw_segments),
        "refined_segments": len(refined_segments),
        "identity_segments": len(identity_windows),
        "asr_segments": len(identities),
        # Backward-compatible metric name retained for older consumers.
        "identity_asr_segments": len(identities),
        "clusters": len(refined_labels),
        "speaker_stats": speaker_stats(refined_segments),
        "profiles": len(profiles),
        "reconciliation": reconciliation,
        "matched_segments": sum(
            1 for item in identity_windows if item.status == "matched"
        ),
        "diarization_only_segments": sum(
            1 for item in identity_windows if item.status == "diarization_only"
        ),
        "unknown_segments": sum(
            1 for item in identity_windows if item.speaker == "Unknown"
        ),
        "asr_skipped_segments": asr_skipped,
        "asr_error_segments": len(asr_errors),
        "warning_count": len(warnings),
        "warnings": warnings,
        "options": _safe_options(options),
    }
    result_payload["metrics"] = metrics_payload
    _write_json(paths["result"], result_payload)
    return paths


def _unknown_identities(segments, status: str) -> list[SegmentIdentity]:
    return [
        SegmentIdentity(
            segment_id=f"seg_{index:04d}",
            cluster=item.cluster,
            start=item.start,
            end=item.end,
            duration=item.duration,
            status=status,
            raw_status=status,
        )
        for index, item in enumerate(segments)
    ]


def _release_torch_cuda_memory(
    device: Optional[str], log: Callable[[str], None] = _runtime_log
) -> None:
    if not str(device or "").startswith("cuda"):
        return
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log("[GPU] Đã giải phóng cache PyTorch trước khi chạy Gipformer")
    except Exception as exc:
        # Cache cleanup is an optimization. Model initialization will still
        # report a useful error if CUDA is genuinely unavailable.
        log(f"[GPU] Không thể giải phóng cache PyTorch: {exc}")


def _build_clean_transcript(
    *,
    job_id: str,
    source_audio: Path,
    duration_seconds: float,
    identities: list[SegmentIdentity],
    reconciliation: dict,
    warnings: list[dict],
) -> dict:
    """Build a user-facing ASR artifact from the consolidated result data."""
    utterances = []
    asr_counts: dict[str, int] = {}
    speaker_rows: dict[str, dict] = {}
    for item in identities:
        asr_counts[item.asr_status] = asr_counts.get(item.asr_status, 0) + 1
        text = item.text.strip()
        if not text:
            continue
        speaker = item.speaker or "Unknown"
        utterances.append(
            {
                "id": item.segment_id,
                "start": item.start,
                "end": item.end,
                "duration": item.duration,
                "speaker": speaker,
                "speaker_id": item.speaker_id,
                "cluster": item.cluster,
                "text": text,
                "asr_status": item.asr_status,
                "identity_status": item.status,
                "identity_confidence": item.score,
            }
        )
        row = speaker_rows.setdefault(
            speaker,
            {"speaker": speaker, "utterances": 0, "speaking_seconds": 0.0},
        )
        row["utterances"] += 1
        row["speaking_seconds"] += item.duration

    speakers = []
    for row in speaker_rows.values():
        row["speaking_seconds"] = round(row["speaking_seconds"], 3)
        speakers.append(row)
    speakers.sort(key=lambda value: value["speaking_seconds"], reverse=True)

    important_warnings = [
        {"code": item.get("code"), "message": item.get("message")}
        for item in warnings
        if item.get("severity") in {"warning", "error"}
    ]
    return {
        "schema_version": "1.0",
        "type": "clean_transcript",
        "job_id": job_id,
        "source_audio": str(source_audio),
        "language": "vi",
        "duration_seconds": round(duration_seconds, 3),
        "summary": {
            "registered_voice_profiles": reconciliation.get("registered_profiles", 0),
            "detected_speaker_clusters": reconciliation.get("diarized_clusters", 0),
            "transcribed_utterances": len(utterances),
            "segments_without_text": len(identities) - len(utterances),
            "asr_status_counts": asr_counts,
            "speakers": speakers,
        },
        "warnings": important_warnings,
        "utterances": utterances,
    }


def _build_transcript_text(payload: dict) -> str:
    segments = payload.get("segments", payload.get("utterances", []))
    lines = [
        "Biên bản ASR",
        f"Số lượt nói có nội dung: {len(segments)}",
        "",
    ]
    for row in segments:
        lines.append(
            f"[{_clock(row.get('start', 0.0))} - {_clock(row.get('end', 0.0))}] "
            f"{row.get('speaker', 'Unknown')}: {row.get('text', '')}"
        )
    return "\n".join(lines) + "\n"


def _build_public_transcript_segments(
    identities: list[SegmentIdentity],
    reconciliation: dict,
    *,
    max_gap_seconds: float,
    max_segment_seconds: float,
) -> list[dict]:
    """Return the minimal public contract requested by UI/API consumers."""
    clusters = sorted({item.cluster for item in identities}, key=cluster_sort_key)
    cluster_labels = {cluster: f"S{index + 1}" for index, cluster in enumerate(clusters)}
    weak_clusters = set(reconciliation.get("weak_evidence_clusters", []))
    cluster_speakers = reconciliation.get("cluster_speakers", {})
    rows: list[dict] = []
    for item in sorted(identities, key=lambda value: (value.start, value.end)):
        text = item.text.strip()
        if not text:
            continue
        if item.cluster in cluster_speakers:
            speaker = cluster_speakers[item.cluster]
        elif item.cluster in weak_clusters or not item.speaker or item.speaker == "Unknown":
            speaker = cluster_labels[item.cluster]
        elif item.speaker.startswith("Speaker "):
            speaker = cluster_labels[item.cluster]
        else:
            speaker = item.speaker
        row = {
            "speaker": speaker,
            "start": round(item.start, 3),
            "end": round(item.end, 3),
            "text": text,
        }
        if rows:
            previous = rows[-1]
            gap = row["start"] - previous["end"]
            merged_duration = max(previous["end"], row["end"]) - previous["start"]
            if (
                previous["speaker"] == row["speaker"]
                and gap <= max_gap_seconds
                and merged_duration <= max_segment_seconds
            ):
                previous["end"] = round(max(previous["end"], row["end"]), 3)
                previous["text"] = _join_asr_text(previous["text"], row["text"])
                continue
        rows.append(row)
    return rows


def _join_asr_text(left: str, right: str) -> str:
    """Join padded ASR windows while removing repeated boundary words."""
    left_words = left.split()
    right_words = right.split()
    max_overlap = min(10, len(left_words), len(right_words))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if [word.casefold() for word in left_words[-size:]] == [
            word.casefold() for word in right_words[:size]
        ]:
            overlap = size
            break
    return " ".join(left_words + right_words[overlap:]).strip()


def _clock(seconds: float) -> str:
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


def _validate_options(options: PipelineOptions) -> None:
    if not 0.0 <= options.verification_threshold <= 1.0:
        raise ValueError("verification_threshold must be between 0 and 1.")
    if options.merge_max_gap_sec < 0 or options.merge_max_duration_sec <= 0:
        raise ValueError("Merge duration/gap configuration is invalid.")
    if options.asr_min_duration_sec < 0 or options.asr_padding_sec < 0:
        raise ValueError("ASR duration/padding configuration is invalid.")
    if not options.asr_provider.strip() or options.asr_batch_size <= 0:
        raise ValueError("ASR provider/batch configuration is invalid.")
    if options.identity_max_window_sec <= 0:
        raise ValueError("identity_max_window_sec must be positive.")
    if options.voice_id_batch_size <= 0:
        raise ValueError("voice_id_batch_size must be positive.")


def _mapping_warnings(mapping, labels, kind, add_warning) -> None:
    label_keys = {cluster_key(label) for label in labels}
    for source in mapping:
        if cluster_key(source) not in label_keys:
            add_warning(
                f"unused_{kind}",
                f"{kind} cho cluster '{source}' không được dùng vì cluster không tồn tại.",
                "info",
                source=source,
            )
    if kind == "cluster_mapping":
        keyed = {cluster_key(key): cluster_key(value) for key, value in mapping.items()}
        cycles = sorted(
            key for key, value in keyed.items() if key != value and keyed.get(value) == key
        )
        if cycles:
            add_warning(
                "cluster_mapping_cycle",
                "Mapping hai chiều sẽ hoán đổi cluster thay vì gộp; hãy kiểm tra cấu hình.",
                clusters=cycles,
            )


def _reconciliation_warnings(report: dict, add_warning) -> None:
    relationship = report["relationship"]
    if relationship in {"more_clusters_than_profiles", "fewer_clusters_than_profiles"}:
        add_warning(
            "profile_cluster_count_mismatch",
            (
                f"Có {report['registered_profiles']} profile dùng được và "
                f"{report['diarized_clusters']} cluster. Đây không tự động là lỗi."
            ),
            "info",
            relationship=relationship,
        )
    if report["split_identity_candidates"]:
        add_warning(
            "possible_over_clustering",
            "Một người được gán cho nhiều cluster; giữ nguyên vì có thể DiariZen đã tách dư.",
            "info",
            profiles=report["split_identity_candidates"],
        )
    if report["mixed_clusters"]:
        add_warning(
            "possible_under_clustering",
            "Một cluster chứa bằng chứng đáng kể của nhiều người; đã giữ nhãn segment-level.",
            clusters=report["mixed_clusters"],
        )
    if report.get("weak_evidence_clusters"):
        add_warning(
            "weak_cluster_identity_evidence",
            "Một số cluster có quá ít thời lượng vượt ngưỡng; chỉ giữ các nhãn segment đủ tin cậy, không ép tên toàn cluster.",
            "info",
            clusters=report["weak_evidence_clusters"],
        )
    if report["profiles_not_observed"]:
        add_warning(
            "profiles_not_observed",
            "Một số người đã đăng ký không xuất hiện hoặc không vượt ngưỡng.",
            "info",
            speaker_ids=report["profiles_not_observed"],
        )


def _job_id(audio_path: Path, requested: Optional[str]) -> str:
    base = requested or f"{audio_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", base).strip("._")
    return safe or "meeting"


def _safe_options(options: PipelineOptions) -> dict:
    payload = asdict(options)
    if payload.get("hf_token"):
        payload["hf_token"] = "***redacted***"
    return payload


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)
