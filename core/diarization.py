from __future__ import annotations

import functools
import os
import time
from pathlib import Path
from typing import Callable, Optional

import soundfile as sf

from config.settings import DIARIZEN_DIR, SETTINGS
from core.models import DiarizationSegment


def run_diarization(
    audio_path: str | Path,
    model_id: str = SETTINGS.diarizen_model_id,
    hf_token: Optional[str] = SETTINGS.hf_token,
    device: Optional[str] = SETTINGS.device,
    log: Callable[[str], None] = print,
) -> list[DiarizationSegment]:
    """Run DiariZen on a normalized 16 kHz mono WAV file."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio input does not exist: {path}")

    _login_huggingface(hf_token)
    _patch_legacy_audio_dependencies()

    import torch
    from diarizen.pipelines.inference import DiariZenPipeline

    requested_device = device or "cuda:0"
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Đã yêu cầu CUDA nhưng PyTorch hiện tại không nhận GPU. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}. "
            "Hãy chạy scripts/setup.ps1 sau khi đóng app cũ."
        )
    runtime_device = torch.device(requested_device)
    if runtime_device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(runtime_device)
        vram_gb = torch.cuda.get_device_properties(runtime_device).total_memory / 1024**3
        log(f"[GPU] DiariZen -> {runtime_device} | {gpu_name} | VRAM {vram_gb:.1f} GB")
    else:
        log("[DEVICE] DiariZen -> CPU")
    local_model = DIARIZEN_DIR / model_id.replace("/", "__")
    local_embedding = (
        DIARIZEN_DIR
        / "pyannote__wespeaker-voxceleb-resnet34-LM"
        / "pytorch_model.bin"
    )
    if (local_model / "pytorch_model.bin").exists() and local_embedding.exists():
        pipeline = DiariZenPipeline(
            diarizen_hub=local_model,
            embedding_model=str(local_embedding),
        )
    else:
        # No cache_dir is passed here because DiariZen treats any cache_dir as
        # local-only.  Hugging Face then uses its normal cache and can download.
        pipeline = DiariZenPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(runtime_device)

    if runtime_device.type == "cuda":
        # The checkpoint defaults to 32, which is too large for an RTX 3060 6 GB.
        pipeline._segmentation.batch_size = SETTINGS.diarization_gpu_batch_size
        log(
            "[GPU] DiariZen segmentation batch_size="
            f"{SETTINGS.diarization_gpu_batch_size} (tối ưu cho VRAM 6 GB)"
        )
    else:
        pipeline._segmentation.batch_size = SETTINGS.diarization_cpu_batch_size
        log(
            "[CPU] DiariZen segmentation batch_size="
            f"{SETTINGS.diarization_cpu_batch_size} (giới hạn RAM)"
        )

    _attach_progress_logging(pipeline, log, SETTINGS.progress_log_every)

    started = time.perf_counter()
    annotation = pipeline(str(path))
    segments: list[DiarizationSegment] = []
    for index, (turn, _, speaker) in enumerate(
        annotation.itertracks(yield_label=True), 1
    ):
        start = round(float(turn.start), 3)
        end = round(float(turn.end), 3)
        if end <= start:
            continue
        cluster = str(speaker)
        segments.append(
            DiarizationSegment(
                cluster=cluster,
                original_cluster=cluster,
                start=start,
                end=end,
                duration=end - start,
            )
        )
        if index % SETTINGS.progress_log_every == 0:
            log(f"[DiariZen] Đã xuất {index} segment diarization")
    log(
        f"[DiariZen] Hoàn tất {len(segments)} segment trong "
        f"{time.perf_counter() - started:.1f}s"
    )
    return sorted(segments, key=lambda item: (item.start, item.end))


def _attach_progress_logging(pipeline, log: Callable[[str], None], every: int) -> None:
    """Pass pyannote's internal hooks through DiariZen's custom pipeline."""
    original_segmentations = pipeline.get_segmentations
    original_embeddings = pipeline.get_embeddings
    next_report = {"segmentation": every, "embeddings": every}

    def progress_hook(stage, _artifact=None, completed=0, total=0, **_kwargs):
        completed = min(int(completed), int(total)) if total else int(completed)
        total = int(total)
        threshold = next_report.setdefault(stage, every)
        if completed == 0:
            log(f"[DiariZen:{stage}] Bắt đầu, tổng {total} batch/chunk")
            return
        if completed >= threshold or (total and completed >= total):
            percent = (100.0 * completed / total) if total else 0.0
            log(f"[DiariZen:{stage}] {completed}/{total} ({percent:.1f}%)")
            while next_report[stage] <= completed:
                next_report[stage] += every

    def tracked_segmentations(file, hook=None, soft=False):
        return original_segmentations(file, hook=progress_hook, soft=soft)

    def tracked_embeddings(
        file, binary_segmentations, exclude_overlap=False, hook=None
    ):
        return original_embeddings(
            file,
            binary_segmentations,
            exclude_overlap=exclude_overlap,
            hook=progress_hook,
        )

    pipeline.get_segmentations = tracked_segmentations
    pipeline.get_embeddings = tracked_embeddings


def _login_huggingface(hf_token: Optional[str]) -> None:
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception:
        # Public models can still be loaded without a token.  Authentication
        # errors will surface later with a more useful model-download message.
        return


def _patch_legacy_audio_dependencies() -> None:
    """Bridge DiariZen's pinned pyannote code to NumPy 2 / torchaudio 2.11+."""
    import numpy as np
    import torch
    import torchaudio

    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    if not hasattr(np, "NAN"):
        np.NAN = np.nan

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            def __init__(self, *args, **kwargs):
                pass

        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None

    def soundfile_load(
        uri,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format=None,
        buffer_size: int = 4096,
        backend=None,
    ):
        frames = -1 if num_frames is None or num_frames < 0 else num_frames
        data, sample_rate = sf.read(
            str(uri),
            start=frame_offset,
            frames=frames,
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(data.T if channels_first else data)
        return waveform, int(sample_rate)

    torchaudio.load = soundfile_load

    if not getattr(torch.load, "_meeting_insight_patched", False):
        original_load = torch.load.__wrapped__ if hasattr(torch.load, "__wrapped__") else torch.load

        @functools.wraps(original_load)
        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        patched_load._meeting_insight_patched = True
        torch.load = patched_load

    try:
        from torch.torch_version import TorchVersion

        torch.serialization.add_safe_globals([TorchVersion])
    except Exception:
        return
