from __future__ import annotations

import json
import hashlib
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from config.settings import CAMPPLUS_DIR, SETTINGS, VOICE_DB_PATH
from core.audio_enhancer import (
    convert_to_16k_mono,
    inspect_audio_quality,
    load_mono_audio,
    write_concatenated_slices,
)
from core.models import ClusterIdentity, DiarizationSegment, SegmentIdentity, VoiceProfile
from core.segment_processing import select_clean_slices


CAMPPLUS_MODELS: dict[str, dict[str, Any]] = {
    "iic/speech_campplus_sv_zh_en_16k-common_advanced": {
        "revision": "v1.0.0",
        "model_file": "campplus_cn_en_common.pt",
        "model_class": "speakerlab.models.campplus.DTDNN.CAMPPlus",
        "model_args": {"feat_dim": 80, "embedding_size": 192},
    },
    "iic/speech_campplus_sv_zh-cn_16k-common": {
        "revision": "master",
        "model_file": "campplus_cn_common.bin",
        "model_class": "speakerlab.models.campplus.DTDNN.CAMPPlus",
        "model_args": {"feat_dim": 80, "embedding_size": 192},
    },
}


@dataclass(frozen=True)
class EnrollmentRequest:
    speaker_id: str
    display_name: str
    sample_paths: tuple[str, ...]


class CAMPPlusEmbedder:
    """CAM++ feature extractor backed by the official 3D-Speaker code."""

    def __init__(
        self,
        model_id: str = SETTINGS.campplus_model_id,
        model_cache_dir: str | Path = CAMPPLUS_DIR,
        device: Optional[str] = SETTINGS.device,
        speakerlab_root: Optional[str | Path] = SETTINGS.speakerlab_root,
    ) -> None:
        if model_id not in CAMPPLUS_MODELS:
            raise ValueError(f"Unsupported CAM++ model id: {model_id}")
        root = speakerlab_root or os.environ.get("SPEAKERLAB_ROOT")
        if root:
            resolved = str(Path(root).resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

        try:
            import torch
            from speakerlab.process.processor import FBank
            from speakerlab.utils.builder import dynamic_import
        except ImportError as exc:
            raise RuntimeError(
                "CAM++ cần torch và speakerlab của 3D-Speaker. "
                "Hãy chạy scripts/setup.ps1 hoặc đặt SPEAKERLAB_ROOT."
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.sample_rate = SETTINGS.sample_rate
        requested_device = device or "cuda:0"
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CAM++ đã được cấu hình CUDA nhưng PyTorch không nhận GPU. "
                f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}."
            )
        self.device = torch.device(requested_device)
        if self.device.type == "cuda":
            print(
                f"[GPU] CAM++ -> {self.device} | "
                f"{torch.cuda.get_device_name(self.device)}",
                flush=True,
            )
        else:
            print("[DEVICE] CAM++ -> CPU", flush=True)
        self.feature_extractor = FBank(80, sample_rate=self.sample_rate, mean_nor=True)

        config = CAMPPLUS_MODELS[model_id]
        model_path = ensure_campplus_weights(model_id, model_cache_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"CAM++ weight file not found: {model_path}")

        self.model = dynamic_import(config["model_class"])(**config["model_args"])
        state = torch.load(str(model_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def embed_file(self, audio_path: str | Path) -> np.ndarray:
        waveform, sample_rate = load_mono_audio(audio_path)
        return self.embed_waveform(waveform, sample_rate)

    def embed_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.embed_waveforms([waveform], sample_rate, batch_size=1)[0]

    def embed_waveforms(
        self,
        waveforms: list[np.ndarray],
        sample_rate: int,
        batch_size: int = SETTINGS.voice_id_batch_size,
    ) -> list[np.ndarray]:
        """Embed many windows without padding different feature lengths.

        CAM++ supports batches, but padding a shorter utterance changes its
        statistics-pooled embedding. Grouping exact frame lengths preserves the
        single-window result while using the GPU efficiently for repeated
        lengths produced by ``split_long_segments``.
        """
        if batch_size <= 0:
            raise ValueError("CAM++ batch_size must be positive.")
        if not waveforms:
            return []
        features = [
            self._extract_features(waveform, sample_rate) for waveform in waveforms
        ]
        grouped: dict[int, list[int]] = {}
        for index, feature in enumerate(features):
            grouped.setdefault(int(feature.shape[0]), []).append(index)

        embeddings: list[Optional[np.ndarray]] = [None] * len(features)
        torch = self.torch
        with torch.inference_mode():
            for indices in grouped.values():
                for offset in range(0, len(indices), batch_size):
                    batch_indices = indices[offset : offset + batch_size]
                    batch = torch.stack(
                        [features[index] for index in batch_indices]
                    ).to(self.device)
                    values = self.model(batch).detach().cpu().numpy()
                    for index, value in zip(batch_indices, values):
                        embeddings[index] = normalize_embedding(value)
        if any(value is None for value in embeddings):
            raise RuntimeError("CAM++ không trả đủ embedding cho batch.")
        return [value for value in embeddings if value is not None]

    def _extract_features(self, waveform: np.ndarray, sample_rate: int):
        torch = self.torch
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise ValueError("Không có waveform để trích xuất embedding.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Waveform chứa NaN hoặc Infinity.")
        if sample_rate != self.sample_rate:
            try:
                import torchaudio

                tensor = torch.from_numpy(samples).unsqueeze(0)
                tensor = torchaudio.functional.resample(tensor, sample_rate, self.sample_rate)
                samples = tensor.squeeze(0).numpy()
            except Exception as exc:
                raise RuntimeError("Không thể resample audio cho CAM++.") from exc
        minimum = int(0.5 * self.sample_rate)
        if len(samples) < minimum:
            samples = np.pad(samples, (0, minimum - len(samples)))
        wav = torch.from_numpy(samples).unsqueeze(0)
        return self.feature_extractor(wav)


def enroll_speaker(
    speaker_id: str,
    display_name: str,
    sample_paths: list[str | Path],
    embedder: CAMPPlusEmbedder,
    voice_db_path: str | Path = VOICE_DB_PATH,
) -> dict[str, Any]:
    """Persist one named CAM++ profile in the model-specific voice database."""
    safe_id = safe_speaker_id(speaker_id or display_name)
    request = EnrollmentRequest(
        speaker_id=safe_id,
        display_name=display_name.strip(),
        sample_paths=tuple(str(path) for path in sample_paths),
    )
    db_path = Path(voice_db_path)
    profile = _profile_from_request(
        request,
        embedder,
        db_path.parent / "samples",
        "database",
        strict=True,
        issues=[],
    )
    database = load_voice_db(db_path)
    existing_model = database.get("model_id") or database.get("model")
    if database.get("speakers") and existing_model != embedder.model_id:
        raise ValueError(
            f"Voice DB dùng model {existing_model}; không thể trộn với {embedder.model_id}. "
            "Hãy dùng database riêng cho từng model rồi enroll lại."
        )
    speakers = [
        item for item in database.get("speakers", []) if item.get("speaker_id") != safe_id
    ]
    speakers.append(
        {
            **profile.metadata(),
            "embedding": np.asarray(profile.embedding).round(8).tolist(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    payload = {
        "version": 1,
        "model_id": embedder.model_id,
        "sample_rate": embedder.sample_rate,
        "speakers": sorted(speakers, key=lambda item: item["speaker_id"]),
    }
    save_voice_db(payload, db_path)
    return payload


def build_runtime_profiles(
    embedder: CAMPPlusEmbedder,
    requests: Optional[list[EnrollmentRequest]] = None,
    voice_db_path: str | Path = VOICE_DB_PATH,
    workspace: str | Path = CAMPPLUS_DIR / "runtime_samples",
    include_saved: bool = True,
) -> tuple[list[VoiceProfile], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    profiles: list[VoiceProfile] = []
    if include_saved:
        try:
            profiles = load_voice_profiles(
                voice_db_path, embedder.model_id, issues=issues
            )
        except Exception as exc:
            issues.append(
                {
                    "code": "saved_profiles_unavailable",
                    "severity": "warning",
                    "message": str(exc),
                }
            )
    by_id = {profile.speaker_id: profile for profile in profiles}
    for request in requests or []:
        normalized = EnrollmentRequest(
            speaker_id=safe_speaker_id(request.speaker_id or request.display_name),
            display_name=request.display_name.strip(),
            sample_paths=tuple(request.sample_paths),
        )
        try:
            by_id[normalized.speaker_id] = _profile_from_request(
                normalized,
                embedder,
                Path(workspace),
                "run_sample",
                strict=False,
                issues=issues,
            )
        except Exception as exc:
            issues.append(
                {
                    "code": "profile_rejected",
                    "severity": "warning",
                    "speaker_id": normalized.speaker_id,
                    "message": str(exc),
                }
            )
    display_names: dict[str, list[str]] = {}
    for profile in by_id.values():
        display_names.setdefault(profile.display_name.casefold(), []).append(profile.speaker_id)
    for name, speaker_ids in display_names.items():
        if len(speaker_ids) > 1:
            issues.append(
                {
                    "code": "duplicate_display_name",
                    "severity": "warning",
                    "speaker_ids": speaker_ids,
                    "message": "Nhiều speaker_id có cùng tên hiển thị; nên đổi tên để đọc output rõ hơn.",
                }
            )
    return sorted(by_id.values(), key=lambda item: item.speaker_id), issues


def verify_segments(
    audio_path: str | Path,
    segments: list[DiarizationSegment],
    profiles: list[VoiceProfile],
    embedder: CAMPPlusEmbedder,
    threshold: float = SETTINGS.verification_threshold,
    min_duration: float = SETTINGS.verification_min_segment_sec,
    log: Callable[[str], None] = print,
    log_every: int = SETTINGS.progress_log_every,
    batch_size: int = SETTINGS.voice_id_batch_size,
) -> list[SegmentIdentity]:
    waveform, sample_rate = load_mono_audio(audio_path)
    output: list[SegmentIdentity] = []
    candidates: list[tuple[SegmentIdentity, np.ndarray]] = []
    total = len(segments)
    log(
        f"[CAM++] Bắt đầu định danh {total} segment trên {embedder.device} "
        f"(batch tối đa {batch_size}, chỉ ghép cùng độ dài)"
    )
    for index, segment in enumerate(segments):
        identity = SegmentIdentity(
            segment_id=f"seg_{index:04d}",
            cluster=segment.cluster,
            start=segment.start,
            end=segment.end,
            duration=segment.duration,
        )
        if not profiles:
            identity.raw_status = identity.status = "no_profiles"
        elif segment.duration < min_duration:
            identity.raw_status = identity.status = "too_short"
        else:
            start_i = max(0, int(segment.start * sample_rate))
            end_i = min(len(waveform), int(segment.end * sample_rate))
            candidates.append((identity, waveform[start_i:end_i]))
        output.append(identity)

    if candidates:
        try:
            embeddings = embedder.embed_waveforms(
                [candidate[1] for candidate in candidates],
                sample_rate,
                batch_size=batch_size,
            )
            for (identity, _), embedding in zip(candidates, embeddings):
                _apply_embedding_match(identity, embedding, profiles, threshold)
        except Exception:
            # Recover the previous segment-level error behavior if one item or
            # one GPU batch fails, rather than losing every identity window.
            for identity, samples in candidates:
                try:
                    embedding = embedder.embed_waveform(samples, sample_rate)
                    _apply_embedding_match(identity, embedding, profiles, threshold)
                except Exception as exc:
                    identity.raw_status = identity.status = "embedding_error"
                    identity.identity_error = str(exc)

    for completed in range(1, total + 1):
        if completed % log_every == 0 or completed == total:
            log(f"[CAM++] Đã xử lý {completed}/{total} segment")
    return output


def _apply_embedding_match(
    identity: SegmentIdentity,
    embedding: np.ndarray,
    profiles: list[VoiceProfile],
    threshold: float,
) -> None:
    profile, score, scores = best_profile_match(embedding, profiles)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1:
        identity.second_best_speaker_id = ranked[1][0]
        identity.second_best_score = ranked[1][1]
        identity.score_margin = round(ranked[0][1] - ranked[1][1], 4)
    matched = profile is not None and score is not None and score >= threshold
    identity.scores = scores
    identity.raw_score = identity.score = (
        round(score, 4) if score is not None else None
    )
    identity.raw_status = identity.status = "matched" if matched else "below_threshold"
    if matched and profile:
        identity.raw_speaker_id = identity.speaker_id = profile.speaker_id
        identity.raw_speaker = identity.speaker = profile.display_name


def build_cluster_references(
    audio_path: str | Path,
    segments: list[DiarizationSegment],
    profiles: list[VoiceProfile],
    embedder: CAMPPlusEmbedder,
    output_dir: str | Path,
    threshold: float = SETTINGS.verification_threshold,
    target_seconds: float = SETTINGS.cluster_enrollment_target_sec,
    min_clean_seconds: float = SETTINGS.cluster_enrollment_min_clean_sec,
) -> dict[str, ClusterIdentity]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    references: dict[str, ClusterIdentity] = {}
    for cluster in sorted({item.cluster for item in segments}):
        provisional = f"Cluster {cluster}"
        selected = select_clean_slices(
            cluster,
            segments,
            target_seconds=target_seconds,
            min_slice_seconds=min_clean_seconds,
        )
        if not selected:
            references[cluster] = ClusterIdentity(
                cluster=cluster,
                speaker_id=None,
                speaker="Unknown",
                provisional_speaker=provisional,
                score=None,
                status="insufficient_clean_audio",
                source="cluster_enrollment",
            )
            continue
        destination = output_root / f"{safe_filename(cluster)}.wav"
        try:
            enrollment_file, used_seconds = write_concatenated_slices(
                audio_path, selected, destination
            )
            embedding = embedder.embed_file(enrollment_file)
            profile, score, scores = best_profile_match(embedding, profiles)
            matched = profile is not None and score is not None and score >= threshold
            references[cluster] = ClusterIdentity(
                cluster=cluster,
                speaker_id=profile.speaker_id if matched and profile else None,
                speaker=profile.display_name if matched and profile else "Unknown",
                provisional_speaker=provisional,
                score=round(score, 4) if score is not None else None,
                status="matched" if matched else "below_threshold",
                source="cluster_enrollment",
                clean_audio_seconds=used_seconds,
                enrollment_file=str(enrollment_file),
                profile_scores=scores,
            )
        except Exception as exc:
            references[cluster] = ClusterIdentity(
                cluster=cluster,
                speaker_id=None,
                speaker="Unknown",
                provisional_speaker=provisional,
                score=None,
                status="embedding_error",
                source="cluster_enrollment",
                candidate_speakers=[str(exc)],
            )
    return references


def assign_cluster_profiles(
    references: dict[str, ClusterIdentity],
    profiles: list[VoiceProfile],
    threshold: float = SETTINGS.verification_threshold,
) -> dict[str, ClusterIdentity]:
    """Globally map cluster embeddings to named samples using cosine scores."""
    if not references or not profiles:
        return references
    from scipy.optimize import linear_sum_assignment

    profile_by_id = {profile.speaker_id: profile for profile in profiles}
    profile_ids = list(profile_by_id)
    clusters = [
        cluster for cluster, reference in references.items() if reference.profile_scores
    ]
    if not clusters:
        return references

    matrix = np.asarray(
        [
            [references[cluster].profile_scores.get(profile_id, -1.0) for profile_id in profile_ids]
            for cluster in clusters
        ],
        dtype=np.float32,
    )
    row_indices, column_indices = linear_sum_assignment(-matrix)

    for cluster in clusters:
        reference = references[cluster]
        reference.speaker_id = None
        reference.speaker = "Unknown"
        reference.status = "below_threshold"
        reference.source = "global_cluster_mapping"

    assigned_clusters = set()
    for row, column in zip(row_indices, column_indices):
        cluster = clusters[int(row)]
        profile_id = profile_ids[int(column)]
        score = float(matrix[int(row), int(column)])
        if score < threshold:
            continue
        reference = references[cluster]
        profile = profile_by_id[profile_id]
        reference.speaker_id = profile_id
        reference.speaker = profile.display_name
        reference.score = round(score, 4)
        reference.status = "matched"
        reference.source = "global_cluster_mapping"
        assigned_clusters.add(cluster)

    # k may be greater than n when DiariZen splits one person into multiple
    # clusters. Allow a repeated identity only when its own best score passes.
    for cluster in clusters:
        if cluster in assigned_clusters:
            continue
        reference = references[cluster]
        best_id, best_score = max(
            reference.profile_scores.items(), key=lambda item: item[1]
        )
        if best_score < threshold:
            continue
        profile = profile_by_id[best_id]
        reference.speaker_id = best_id
        reference.speaker = profile.display_name
        reference.score = round(best_score, 4)
        reference.status = "matched"
        reference.source = "global_cluster_mapping_repeated"
    return references


def aggregate_cluster_segment_scores(
    references: dict[str, ClusterIdentity],
    identities: list[SegmentIdentity],
    top_k: int = 5,
) -> dict[str, ClusterIdentity]:
    """Add a robust top-k segment score to each cluster/profile score matrix."""
    grouped: dict[str, list[SegmentIdentity]] = {}
    for identity in identities:
        if identity.scores:
            grouped.setdefault(identity.cluster, []).append(identity)
    for cluster, members in grouped.items():
        reference = references.get(cluster)
        if reference is None:
            continue
        profile_ids = set(reference.profile_scores)
        for member in members:
            profile_ids.update(member.scores)
        for profile_id in profile_ids:
            values = sorted(
                (
                    float(member.scores[profile_id])
                    for member in members
                    if profile_id in member.scores
                ),
                reverse=True,
            )
            if not values:
                continue
            robust_score = float(np.mean(values[: max(1, top_k)]))
            reference.profile_scores[profile_id] = round(
                max(reference.profile_scores.get(profile_id, -1.0), robust_score),
                4,
            )
    return references


def best_profile_match(
    embedding: np.ndarray, profiles: list[VoiceProfile]
) -> tuple[Optional[VoiceProfile], Optional[float], dict[str, float]]:
    if not profiles:
        return None, None, {}
    scored = {
        profile.speaker_id: float(
            np.dot(normalize_embedding(embedding), normalize_embedding(profile.embedding))
        )
        for profile in profiles
    }
    best_id = max(scored, key=scored.get)
    profile_by_id = {profile.speaker_id: profile for profile in profiles}
    return (
        profile_by_id[best_id],
        scored[best_id],
        {key: round(value, 4) for key, value in sorted(scored.items())},
    )


def ensure_campplus_weights(
    model_id: str = SETTINGS.campplus_model_id,
    model_cache_dir: str | Path = CAMPPLUS_DIR,
) -> Path:
    """Download the one checkpoint needed for inference from ModelScope.

    ModelScope's concurrent SDK downloader can stall on some Windows networks.
    The official repository endpoint supports a normal streamed download, which
    is both deterministic and sufficient because the architecture lives in
    3D-Speaker.
    """
    if model_id not in CAMPPLUS_MODELS:
        raise ValueError(f"Unsupported CAM++ model id: {model_id}")
    config = CAMPPLUS_MODELS[model_id]
    namespace, model_name = model_id.split("/", 1)
    model_dir = Path(model_cache_dir) / namespace / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / config["model_file"]
    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        return model_path

    query = urllib.parse.urlencode(
        {"Revision": config["revision"], "FilePath": config["model_file"]}
    )
    url = f"https://modelscope.cn/api/v1/models/{model_id}/repo?{query}"
    temporary = model_path.with_suffix(model_path.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if temporary.stat().st_size <= 1_000_000:
            raise RuntimeError("CAM++ checkpoint download is unexpectedly small.")
        temporary.replace(model_path)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            f"Không tải được CAM++ từ ModelScope: {url}. "
            "Có thể tải thủ công file checkpoint vào " + str(model_path)
        ) from exc
    return model_path


def load_voice_profiles(
    voice_db_path: str | Path = VOICE_DB_PATH,
    expected_model_id: Optional[str] = SETTINGS.campplus_model_id,
    issues: Optional[list[dict[str, Any]]] = None,
) -> list[VoiceProfile]:
    issues = issues if issues is not None else []
    database = load_voice_db(voice_db_path)
    speakers = database.get("speakers", [])
    model_id = database.get("model_id") or database.get("model")
    if speakers and expected_model_id and model_id != expected_model_id:
        raise ValueError(
            f"Voice DB hiện dùng {model_id}, nhưng pipeline dùng {expected_model_id}. "
            "Cần enroll lại mẫu bằng CAM++."
        )
    profiles: list[VoiceProfile] = []
    expected_dimension = None
    if expected_model_id in CAMPPLUS_MODELS:
        expected_dimension = CAMPPLUS_MODELS[expected_model_id]["model_args"]["embedding_size"]
    seen_ids: set[str] = set()
    for item in speakers:
        embedding = item.get("embedding")
        speaker_id = item.get("speaker_id")
        if not speaker_id or speaker_id in seen_ids:
            issues.append(
                {
                    "code": "invalid_or_duplicate_profile_id",
                    "severity": "warning",
                    "speaker_id": speaker_id,
                    "message": "Profile thiếu ID hoặc bị trùng và đã được bỏ qua.",
                }
            )
            continue
        try:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if expected_dimension and vector.size != expected_dimension:
                raise ValueError(
                    f"embedding dimension {vector.size}, expected {expected_dimension}"
                )
            profiles.append(
                VoiceProfile(
                    speaker_id=speaker_id,
                    display_name=item.get("display_name", speaker_id),
                    embedding=normalize_embedding(vector),
                    sample_files=list(item.get("sample_files", [])),
                    source=item.get("source", "database"),
                )
            )
            seen_ids.add(speaker_id)
        except Exception as exc:
            issues.append(
                {
                    "code": "invalid_saved_embedding",
                    "severity": "warning",
                    "speaker_id": speaker_id,
                    "message": str(exc),
                }
            )
    return profiles


def load_voice_db(voice_db_path: str | Path = VOICE_DB_PATH) -> dict[str, Any]:
    path = Path(voice_db_path)
    if not path.exists():
        return {
            "version": 1,
            "model_id": SETTINGS.campplus_model_id,
            "sample_rate": SETTINGS.sample_rate,
            "speakers": [],
        }
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Voice database JSON bị hỏng tại {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("speakers", []), list):
        raise ValueError(f"Voice database không đúng schema tại {path}.")
    return payload


def save_voice_db(payload: dict[str, Any], voice_db_path: str | Path = VOICE_DB_PATH) -> None:
    path = Path(voice_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    value = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("Embedding rỗng hoặc chứa giá trị không hữu hạn.")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Embedding có norm bằng 0.")
    return value / norm


def safe_speaker_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        # Vietnamese-only names still need a stable ASCII-safe storage key.
        normalized = "speaker_" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return normalized


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._") or "cluster"


def _profile_from_request(
    request: EnrollmentRequest,
    embedder: CAMPPlusEmbedder,
    samples_root: Path,
    source: str,
    strict: bool,
    issues: list[dict[str, Any]],
) -> VoiceProfile:
    if not request.display_name.strip():
        raise ValueError("Tên người nói không được để trống.")
    if not request.sample_paths:
        raise ValueError(f"{request.display_name} chưa có sample giọng.")
    speaker_dir = samples_root / request.speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)
    embeddings: list[np.ndarray] = []
    normalized_files: list[str] = []
    for index, raw_path in enumerate(request.sample_paths):
        source_path = Path(raw_path)
        try:
            if not source_path.exists():
                raise FileNotFoundError(f"Voice sample does not exist: {source_path}")
            output_name = f"{index:02d}_{safe_filename(source_path.stem)}_16k.wav"
            normalized = convert_to_16k_mono(
                source_path,
                speaker_dir,
                output_name=output_name,
                overwrite=True,
            )
            quality = inspect_audio_quality(normalized)
            if float(quality["duration_seconds"]) < SETTINGS.enrollment_min_duration_sec:
                raise ValueError(
                    f"Sample phải dài ít nhất {SETTINGS.enrollment_min_duration_sec:.1f} giây."
                )
            if quality["is_silent"]:
                raise ValueError("Sample gần như im lặng.")
            embedding = embedder.embed_file(normalized)
            normalized_files.append(str(normalized))
            embeddings.append(embedding)
        except Exception as exc:
            if strict:
                raise
            issues.append(
                {
                    "code": "invalid_enrollment_sample",
                    "severity": "warning",
                    "speaker_id": request.speaker_id,
                    "sample": str(source_path),
                    "message": str(exc),
                }
            )
    if not embeddings:
        raise ValueError(f"Không có sample hợp lệ cho {request.display_name}.")
    profile_embedding = normalize_embedding(np.mean(np.stack(embeddings), axis=0))
    return VoiceProfile(
        speaker_id=request.speaker_id,
        display_name=request.display_name,
        embedding=profile_embedding,
        sample_files=normalized_files,
        source=source,
    )
