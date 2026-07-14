from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config.settings import (  # noqa: E402
    CAMPPLUS_DIR,
    DIARIZEN_DIR,
    GIPFORMER_DIR,
    SETTINGS,
    ensure_runtime_dirs,
)
from core.voice_id import ensure_campplus_weights  # noqa: E402


GIPFORMER_FILES = (
    "encoder-epoch-35-avg-6.int8.onnx",
    "decoder-epoch-35-avg-6.int8.onnx",
    "joiner-epoch-35-avg-6.int8.onnx",
    "tokens.txt",
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Tải checkpoint cho Voice Identity Studio")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["gipformer", "diarizen", "campplus"],
        default=["gipformer", "diarizen", "campplus"],
    )
    parser.add_argument("--hf-token", default=SETTINGS.hf_token)
    args = parser.parse_args()
    ensure_runtime_dirs()
    if "gipformer" in args.only:
        download_gipformer()
    if "diarizen" in args.only:
        download_diarizen(args.hf_token)
    if "campplus" in args.only:
        download_campplus()


def download_gipformer() -> None:
    from huggingface_hub import hf_hub_download

    print("Downloading Gipformer...")
    for filename in GIPFORMER_FILES:
        path = hf_hub_download(
            repo_id=SETTINGS.gipformer_repo_id,
            filename=filename,
            local_dir=GIPFORMER_DIR,
        )
        print(f"  ok {Path(path).name}")


def download_diarizen(hf_token: str | None) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    print("Downloading DiariZen...")
    target = DIARIZEN_DIR / SETTINGS.diarizen_model_id.replace("/", "__")
    snapshot_download(
        repo_id=SETTINGS.diarizen_model_id,
        local_dir=target,
        token=hf_token or None,
    )
    embedding_dir = DIARIZEN_DIR / "pyannote__wespeaker-voxceleb-resnet34-LM"
    hf_hub_download(
        repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        filename="pytorch_model.bin",
        local_dir=embedding_dir,
        token=hf_token or None,
    )
    print(f"  ok {target}")


def download_campplus() -> None:
    print("Downloading CAM++...")
    path = ensure_campplus_weights(SETTINGS.campplus_model_id, CAMPPLUS_DIR)
    print(f"  ok {path}")


if __name__ == "__main__":
    main()
