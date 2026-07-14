from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from huggingface_hub import hf_hub_download

from config.settings import GIPFORMER_DIR, SETTINGS
from core.audio_enhancer import load_mono_audio
from core.models import SegmentIdentity


class GipformerASR:
    """Vietnamese Gipformer RNNT inference through sherpa-onnx."""

    FILES = (
        "encoder-epoch-35-avg-6.int8.onnx",
        "decoder-epoch-35-avg-6.int8.onnx",
        "joiner-epoch-35-avg-6.int8.onnx",
        "tokens.txt",
    )

    def __init__(
        self,
        model_repo: str = SETTINGS.gipformer_repo_id,
        model_dir: str | Path = GIPFORMER_DIR,
        num_threads: int = SETTINGS.asr_num_threads,
        provider: str = SETTINGS.asr_provider,
        device: Optional[str] = SETTINGS.device,
        log: Callable[[str], None] = print,
    ) -> None:
        self._torch_dll_dir = None
        if provider == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "Gipformer yêu cầu CUDA nhưng PyTorch không nhận GPU. "
                        "Hãy cài PyTorch CUDA và khởi động lại app."
                    )
                log(
                    f"[GPU] Gipformer yêu cầu sherpa-onnx provider=cuda trên "
                    f"{torch.cuda.get_device_name(torch.device(device or 'cuda:0'))}"
                )
                torch_lib = Path(torch.__file__).resolve().parent / "lib"
                if torch_lib.exists():
                    import os

                    if hasattr(os, "add_dll_directory"):
                        self._torch_dll_dir = os.add_dll_directory(str(torch_lib))
            except ImportError as exc:
                raise RuntimeError("Không thể kiểm tra CUDA cho Gipformer.") from exc
        else:
            log(f"[DEVICE] Gipformer -> provider={provider}")

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("Gipformer cần package sherpa-onnx.") from exc

        if provider == "cuda":
            sherpa_lib = Path(sherpa_onnx.__file__).resolve().parent / "lib"
            cuda_provider = sherpa_lib / "onnxruntime_providers_cuda.dll"
            if not cuda_provider.exists():
                raise RuntimeError(
                    "sherpa-onnx đang là bản CPU. Hãy đóng app rồi chạy "
                    "scripts/install_gpu.ps1 để cài DLL CUDA chính thức."
                )

        model_root = Path(model_dir)
        model_root.mkdir(parents=True, exist_ok=True)
        for filename in self.FILES:
            if not (model_root / filename).exists():
                hf_hub_download(repo_id=model_repo, filename=filename, local_dir=model_root)

        self.provider = provider
        self.log = log
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_root / self.FILES[0]),
            decoder=str(model_root / self.FILES[1]),
            joiner=str(model_root / self.FILES[2]),
            tokens=str(model_root / self.FILES[3]),
            num_threads=max(1, int(num_threads)),
            sample_rate=SETTINGS.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=provider,
        )

    def transcribe_segments(
        self,
        audio_path: str | Path,
        segments: list[SegmentIdentity],
        min_duration: float = SETTINGS.asr_min_duration_sec,
        padding: float = SETTINGS.asr_padding_sec,
        batch_size: int = SETTINGS.asr_batch_size,
        log_every: int = SETTINGS.progress_log_every,
    ) -> tuple[list[SegmentIdentity], int]:
        waveform, sample_rate = load_mono_audio(audio_path)
        if sample_rate != SETTINGS.sample_rate:
            raise ValueError(
                f"Gipformer expects {SETTINGS.sample_rate} Hz audio, got {sample_rate} Hz."
            )
        audio_duration = len(waveform) / sample_rate
        skipped = 0
        total = len(segments)
        pending: list[tuple[SegmentIdentity, object]] = []
        completed = 0
        self.log(
            f"[Gipformer] Bắt đầu ASR {total} segment | "
            f"provider={self.provider} | batch_size={batch_size}"
        )

        def report_progress(force: bool = False) -> None:
            if force or completed % log_every == 0 or completed == total:
                self.log(f"[Gipformer] Đã xử lý {completed}/{total} segment")

        def decode_pending() -> None:
            nonlocal completed
            if not pending:
                return
            batch = list(pending)
            pending.clear()
            try:
                self.recognizer.decode_streams([stream for _, stream in batch])
                for segment, stream in batch:
                    segment.text = stream.result.text.strip()
                    segment.asr_status = "transcribed"
            except Exception as batch_exc:
                # Recover per segment so one malformed chunk does not discard a batch.
                for segment, stream in batch:
                    try:
                        self.recognizer.decode_streams([stream])
                        segment.text = stream.result.text.strip()
                        segment.asr_status = "transcribed"
                    except Exception as exc:
                        segment.text = ""
                        segment.asr_status = "error"
                        segment.asr_error = f"{batch_exc}; retry: {exc}"
            completed += len(batch)
            report_progress()

        for segment in segments:
            if segment.duration < min_duration:
                segment.text = ""
                segment.asr_status = "skipped_too_short"
                skipped += 1
                completed += 1
                report_progress()
                continue
            try:
                padded_start = max(0.0, segment.start - padding)
                padded_end = min(audio_duration, segment.end + padding)
                chunk = waveform[
                    int(padded_start * sample_rate) : int(padded_end * sample_rate)
                ].astype(np.float32)
                if chunk.size == 0:
                    raise ValueError("ASR chunk rỗng sau khi cắt.")
                stream = self.recognizer.create_stream()
                stream.accept_waveform(sample_rate, chunk)
                pending.append((segment, stream))
                if len(pending) >= max(1, int(batch_size)):
                    decode_pending()
            except Exception as exc:
                segment.text = ""
                segment.asr_status = "error"
                segment.asr_error = str(exc)
                completed += 1
                report_progress()
        decode_pending()
        report_progress(force=True)
        return segments, skipped


def mark_asr_skipped(segments: list[SegmentIdentity]) -> list[SegmentIdentity]:
    for segment in segments:
        segment.asr_status = "disabled"
        segment.text = ""
    return segments


def mark_asr_unavailable(
    segments: list[SegmentIdentity], error: str
) -> list[SegmentIdentity]:
    for segment in segments:
        segment.asr_status = "unavailable"
        segment.asr_error = error
        segment.text = ""
    return segments
