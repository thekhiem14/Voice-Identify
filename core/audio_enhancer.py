from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf

from config.settings import CACHE_DIR, SETTINGS
from core.models import TimeSlice


def convert_to_16k_mono(
    input_path: str | Path,
    output_dir: str | Path = CACHE_DIR,
    output_name: Optional[str] = None,
    overwrite: bool = True,
) -> Path:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Audio input does not exist: {src}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (output_name or f"{src.stem}_16k.wav")
    if dst.exists() and not overwrite:
        return dst

    args = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-ar",
        str(SETTINGS.sample_rate),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        "-vn",
        str(dst),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}: {result.stderr[-800:]}")
    return dst


def probe_duration(audio_path: str | Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except OSError:
        # ffprobe is optional for WAV/FLAC files readable by soundfile.
        pass
    return float(sf.info(str(audio_path)).duration)


def load_mono_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sample_rate)


def inspect_audio_quality(audio_path: str | Path) -> dict[str, float | int | bool]:
    """Return cheap signal checks used to reject empty/corrupt input safely."""
    y, sample_rate = load_mono_audio(audio_path)
    if sample_rate <= 0 or y.size == 0:
        raise ValueError("Audio không có sample hợp lệ.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Audio chứa NaN hoặc Infinity.")
    absolute = np.abs(y)
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    peak = float(absolute.max(initial=0.0))
    clipping_ratio = float(np.mean(absolute >= 0.999))
    return {
        "sample_rate": int(sample_rate),
        "samples": int(y.size),
        "duration_seconds": round(y.size / sample_rate, 3),
        "rms": rms,
        "rms_dbfs": round(20 * np.log10(max(rms, 1e-12)), 2),
        "peak": peak,
        "clipping_ratio": round(clipping_ratio, 6),
        "is_silent": rms < SETTINGS.silence_rms_threshold,
    }


def write_concatenated_slices(
    audio_path: str | Path,
    slices: list[TimeSlice],
    output_path: str | Path,
    silence_seconds: float = 0.08,
) -> tuple[Path, float]:
    """Concatenate selected clean portions into one speaker enrollment WAV."""
    y, sample_rate = load_mono_audio(audio_path)
    chunks: list[np.ndarray] = []
    used_seconds = 0.0
    silence = np.zeros(max(0, int(silence_seconds * sample_rate)), dtype=np.float32)
    for selected in slices:
        start_i = max(0, int(selected.start * sample_rate))
        end_i = min(len(y), int(selected.end * sample_rate))
        if end_i <= start_i:
            continue
        if chunks and len(silence):
            chunks.append(silence)
        chunk = y[start_i:end_i]
        chunks.append(chunk)
        used_seconds += len(chunk) / sample_rate
    if not chunks:
        raise ValueError("No clean audio slices were available for enrollment.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), np.concatenate(chunks), sample_rate, subtype="PCM_16")
    return destination, round(used_seconds, 3)


def quick_snr(audio_path: str | Path) -> float:
    y, sr = sf.read(str(audio_path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    frame = max(1, int(sr * 0.025))
    energies = np.array(
        [
            np.sqrt(np.mean(y[i : i + frame] ** 2))
            for i in range(0, max(0, len(y) - frame), frame)
        ]
    )
    if len(energies) == 0:
        return 0.0
    threshold = np.percentile(energies, 50)
    speech = energies[energies > threshold]
    noise = energies[energies <= threshold]
    if len(speech) == 0 or len(noise) == 0:
        return 0.0
    return float(20 * np.log10(speech.mean() / (noise.mean() + 1e-9)))


def enhance_audio(
    audio_path: str | Path,
    output_dir: str | Path = CACHE_DIR,
    blend_alpha: float = SETTINGS.enhancement_blend_alpha,
    prop_decrease: float = SETTINGS.enhancement_prop_decrease,
    device: Optional[str] = SETTINGS.device,
    log: Callable[[str], None] = print,
) -> tuple[Path, dict[str, Any]]:
    try:
        import noisereduce as nr
        from nara_wpe.wpe import wpe
    except ImportError as exc:
        raise RuntimeError(
            "Enhancement requires optional packages: nara_wpe and noisereduce."
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wpe_path = out_dir / "meeting_dereverbed.wav"
    clean_path = out_dir / "meeting_clean.wav"

    y_in, sr = sf.read(str(audio_path))
    y_mono = y_in if y_in.ndim == 1 else y_in.mean(axis=1)
    # nara-wpe is NumPy-based. Spectral gating below is the expensive stage and
    # has a native PyTorch/CUDA implementation.
    log("[Enhancement] WPE khử vang (CPU/NumPy)")
    y_wpe = wpe(
        y_mono[np.newaxis, :].astype(np.float64),
        taps=10,
        delay=3,
        iterations=5,
        statistics_mode="full",
    )[0].astype(np.float32)
    sf.write(str(wpe_path), y_wpe, sr)

    noise_sample = y_wpe[: 5 * sr]
    runtime_device = device or "cuda:0"
    use_cuda = str(runtime_device).startswith("cuda")
    if use_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Khử nhiễu đã yêu cầu CUDA nhưng PyTorch không nhận GPU."
            )
        log(f"[GPU] NoiseReduce spectral gate -> {runtime_device}")
    else:
        log("[DEVICE] NoiseReduce spectral gate -> CPU")
    y_denoised = nr.reduce_noise(
        y=y_wpe,
        sr=sr,
        y_noise=noise_sample,
        prop_decrease=prop_decrease,
        stationary=False,
        n_jobs=1 if use_cuda else -1,
        use_torch=use_cuda,
        device=runtime_device,
    )

    n = min(len(y_denoised), len(y_wpe))
    blended = (
        blend_alpha * y_denoised[:n] + (1.0 - blend_alpha) * y_wpe[:n]
    ).astype(np.float32)
    sf.write(str(clean_path), blended, sr)
    return clean_path, {
        "wpe_path": str(wpe_path),
        "clean_path": str(clean_path),
        "blend_alpha": blend_alpha,
        "prop_decrease": prop_decrease,
    }
