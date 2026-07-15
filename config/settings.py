from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DATABASE_DIR = DATA_DIR / "database"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

VOICE_DB_PATH = DATABASE_DIR / "voice_db_campplus.json"
CAMPPLUS_DIR = MODELS_DIR / "cam_plus"
GIPFORMER_DIR = MODELS_DIR / "gipformer"
DIARIZEN_DIR = MODELS_DIR / "diarizen"

# The repository is already present in older copies of this project.  Keeping
# this fallback avoids forcing users to clone a second 3D-Speaker checkout.
VENDORED_3D_SPEAKER_DIR = MODELS_DIR / "3D-Speaker"
LEGACY_3D_SPEAKER_DIR = CAMPPLUS_DIR / "3D-Speaker"


def load_env_file(env_path: Path = ENV_PATH) -> None:
    """Load a small .env file without adding python-dotenv as a dependency."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


def _default_speakerlab_root() -> Optional[str]:
    configured = os.environ.get("SPEAKERLAB_ROOT")
    if configured:
        return configured
    if VENDORED_3D_SPEAKER_DIR.exists():
        return str(VENDORED_3D_SPEAKER_DIR)
    if LEGACY_3D_SPEAKER_DIR.exists():
        return str(LEGACY_3D_SPEAKER_DIR)
    return None


@dataclass(frozen=True)
class AISettings:
    sample_rate: int = 16_000
    # CPU is the portable initial mode. The UI can switch the complete pipeline
    # to cuda:0 after scripts/install_gpu.ps1 has installed the GPU runtime.
    device: Optional[str] = os.environ.get("DEVICE") or "cpu"
    # The notebook uses sherpa-onnx's default CPU provider. It is faster for
    # sequential Gipformer streams on this machine, reproduces the reference
    # transcript more closely, and does not compete with PyTorch for VRAM.
    asr_provider: str = os.environ.get("ASR_PROVIDER") or "cpu"
    hf_token: Optional[str] = os.environ.get("HF_TOKEN") or None
    speakerlab_root: Optional[str] = _default_speakerlab_root()

    diarizen_model_id: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    # Bilingual CAM++ keeps the fast 6.85M-parameter architecture while being
    # less Mandarin-specific for Vietnamese meetings containing English terms.
    campplus_model_id: str = "iic/speech_campplus_sv_zh_en_16k-common_advanced"
    gipformer_repo_id: str = "g-group-ai-lab/gipformer-65M-rnnt"

    # User-selected operating point. Scores below this remain an anonymous S-label.
    # Default published with the selected CAM++ Chinese-English checkpoint.
    verification_threshold: float = 0.33
    verification_ambiguity_warning_margin: float = 0.03
    verification_min_segment_sec: float = 0.50
    cluster_enrollment_target_sec: float = 10.0
    cluster_enrollment_min_clean_sec: float = 0.50

    merge_max_gap_sec: float = 2.0
    # Match the operating point that produced the stronger notebook transcript:
    # keep enough context for Gipformer, but cap each utterance at 20 seconds.
    merge_max_duration_sec: float = 20.0
    asr_min_duration_sec: float = 1.5
    asr_padding_sec: float = 0.5
    asr_num_threads: int = max(1, min(8, (os.cpu_count() or 4) // 2))
    # The reference notebook decodes one stream at a time. With 20-second
    # segments, batching 32 encoder attention graphs can request >1 GB in one
    # ONNX allocation and exhaust a 6 GB GPU.
    asr_batch_size: int = max(1, int(os.environ.get("ASR_BATCH_SIZE", "1")))
    progress_log_every: int = max(1, int(os.environ.get("PROGRESS_LOG_EVERY", "100")))
    # WavLM Large with 16-second windows does not fit batch_size=32 in 6 GB VRAM.
    diarization_gpu_batch_size: int = max(
        1, int(os.environ.get("DIARIZATION_GPU_BATCH_SIZE", "2"))
    )
    # WavLM Large is memory-heavy on CPU as well. A small batch avoids large
    # RAM spikes when the UI is switched away from CUDA.
    diarization_cpu_batch_size: int = max(
        1, int(os.environ.get("DIARIZATION_CPU_BATCH_SIZE", "1"))
    )

    enrollment_min_duration_sec: float = 1.0
    silence_rms_threshold: float = 1e-5
    mixed_cluster_secondary_ratio: float = 0.20
    mixed_cluster_secondary_duration_sec: float = 1.5
    cluster_vote_min_evidence_coverage: float = 0.35
    identity_max_window_sec: float = 8.0
    # CAM++ can safely batch windows with exactly equal feature lengths. This
    # avoids padding-induced embedding drift while improving GPU utilization.
    voice_id_batch_size: int = max(
        1, int(os.environ.get("VOICE_ID_BATCH_SIZE", "16"))
    )

    enhancement_blend_alpha: float = 0.7
    enhancement_prop_decrease: float = 0.75


SETTINGS = AISettings()


def ensure_runtime_dirs() -> None:
    for path in (
        DATA_DIR,
        CACHE_DIR,
        DATABASE_DIR,
        MODELS_DIR,
        CAMPPLUS_DIR,
        GIPFORMER_DIR,
        DIARIZEN_DIR,
        OUTPUT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

    if not VOICE_DB_PATH.exists():
        VOICE_DB_PATH.write_text(
            '{\n'
            '  "version": 1,\n'
            f'  "model_id": "{SETTINGS.campplus_model_id}",\n'
            f'  "sample_rate": {SETTINGS.sample_rate},\n'
            '  "speakers": []\n'
            '}\n',
            encoding="utf-8",
        )
