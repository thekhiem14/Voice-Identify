from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from config.settings import OUTPUT_DIR, SETTINGS, VOICE_DB_PATH, ensure_runtime_dirs
from core.pipeline import PipelineOptions, run_pipeline
from core.voice_id import (
    CAMPPlusEmbedder,
    EnrollmentRequest,
    enroll_speaker,
    load_voice_db,
    safe_speaker_id,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    ensure_runtime_dirs()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cluster_mapping = _parse_assignments(args.cluster_map)
        hard_overrides = _parse_assignments(args.force_speaker)
        requests = _parse_enrollment_requests(args.sample)
        paths = run_pipeline(
            args.audio,
            PipelineOptions(
                output_dir=args.output_dir,
                job_name=args.job_name,
                cluster_mapping=cluster_mapping,
                hard_overrides=hard_overrides,
                verification_threshold=args.threshold,
                include_saved_profiles=not args.no_saved_profiles,
                enhance_audio=args.enhance,
                skip_identify=args.skip_identify,
                skip_asr=args.skip_asr,
                allow_partial_results=not args.strict,
                hf_token=args.hf_token,
                device=args.device,
                speakerlab_root=args.speakerlab_root,
            ),
            enrollments=requests,
            progress=_console_progress,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    if args.command == "enroll":
        embedder = CAMPPlusEmbedder(
            device=args.device,
            speakerlab_root=args.speakerlab_root,
        )
        enroll_speaker(
            args.speaker_id,
            args.display_name,
            args.samples,
            embedder,
            VOICE_DB_PATH,
        )
        print(f"Đã đăng ký {args.display_name} vào {VOICE_DB_PATH}")
        return

    if args.command == "profiles":
        database = load_voice_db(VOICE_DB_PATH)
        speakers = database.get("speakers", [])
        if not speakers:
            print("Chưa có mẫu giọng đã lưu.")
        for item in speakers:
            print(
                f"{item['speaker_id']}: {item.get('display_name', item['speaker_id'])} "
                f"({len(item.get('sample_files', []))} sample)"
            )
        return

    from ui.main_layout import run_flet_app

    run_flet_app()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nhận diện người nói tiếng Việt: DiariZen + CAM++ + Gipformer"
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Chạy pipeline đầy đủ")
    run_parser.add_argument("--audio", required=True, help="File cuộc họp đầu vào")
    run_parser.add_argument(
        "--sample",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Mẫu dùng trong lần chạy; lặp lại option để thêm file/người",
    )
    run_parser.add_argument(
        "--cluster-map",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Gộp cụm, ví dụ --cluster-map 1=0",
    )
    run_parser.add_argument(
        "--force-speaker",
        action="append",
        default=[],
        metavar="CLUSTER=NAME",
        help="Ép nhãn cụm, ví dụ --force-speaker 2=Hưng",
    )
    run_parser.add_argument("--threshold", type=float, default=SETTINGS.verification_threshold)
    run_parser.add_argument("--job-name")
    run_parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    run_parser.add_argument("--no-saved-profiles", action="store_true")
    run_parser.add_argument("--enhance", action="store_true")
    run_parser.add_argument("--skip-identify", action="store_true")
    run_parser.add_argument("--skip-asr", action="store_true")
    run_parser.add_argument(
        "--strict",
        action="store_true",
        help="Dừng toàn bộ job nếu CAM++ hoặc Gipformer lỗi thay vì xuất partial result",
    )
    run_parser.add_argument("--hf-token", default=SETTINGS.hf_token)
    run_parser.add_argument("--device", default=SETTINGS.device)
    run_parser.add_argument("--speakerlab-root", default=SETTINGS.speakerlab_root)

    enroll_parser = subparsers.add_parser("enroll", help="Lưu mẫu giọng vào database")
    enroll_parser.add_argument("--speaker-id", required=True)
    enroll_parser.add_argument("--display-name", required=True)
    enroll_parser.add_argument("--samples", nargs="+", required=True)
    enroll_parser.add_argument("--device", default=SETTINGS.device)
    enroll_parser.add_argument("--speakerlab-root", default=SETTINGS.speakerlab_root)

    subparsers.add_parser("profiles", help="Liệt kê mẫu giọng đã lưu")
    return parser


def _parse_assignments(values: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Cú pháp phải là KEY=VALUE: {value}")
        key, target = value.split("=", 1)
        key, target = key.strip(), target.strip()
        if not key or not target:
            raise ValueError(f"Cú pháp phải là KEY=VALUE: {value}")
        output[key] = target
    return output


def _parse_enrollment_requests(values: list[str]) -> list[EnrollmentRequest]:
    grouped: dict[str, list[str]] = defaultdict(list)
    names: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Sample phải có cú pháp NAME=PATH: {value}")
        display_name, path = value.split("=", 1)
        display_name, path = display_name.strip(), path.strip()
        if not display_name or not path:
            raise ValueError(f"Sample phải có cú pháp NAME=PATH: {value}")
        speaker_id = safe_speaker_id(display_name)
        names[speaker_id] = display_name
        grouped[speaker_id].append(str(Path(path).expanduser()))
    return [
        EnrollmentRequest(
            speaker_id=speaker_id,
            display_name=names[speaker_id],
            sample_paths=tuple(paths),
        )
        for speaker_id, paths in grouped.items()
    ]


def _console_progress(number: int, total: int, key: str, message: str) -> None:
    print(f"[{number}/{total}] {message}")


if __name__ == "__main__":
    main()
