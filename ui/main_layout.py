from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from config.settings import SETTINGS, VOICE_DB_PATH, ensure_runtime_dirs
from core.audio_enhancer import probe_duration
from core.pipeline import PipelineOptions, run_pipeline
from core.voice_id import (
    CAMPPlusEmbedder,
    EnrollmentRequest,
    enroll_speaker,
    load_voice_db,
    safe_speaker_id,
)
from ui.components.transcript_view import build_transcript_view, load_result


AUDIO_EXTENSIONS = ["wav", "mp3", "m4a", "flac", "ogg", "aac", "mp4"]
TIMING_LABELS = {
    "preprocessing": "Preprocess",
    "diarization": "DiariZen",
    "refinement": "Merge/refine",
    "enrollment": "CAM++ enrollment",
    "verification": "CAM++ verification",
    "smoothing": "Voice-ID smoothing",
    "asr": "Gipformer ASR",
    "output": "Ghi output",
}


def runtime_inference_options(use_gpu: bool) -> dict[str, object]:
    """Return one coherent device preset for every inference stage."""
    configured = str(SETTINGS.device or "cuda:0")
    gpu_device = configured if configured.startswith("cuda") else "cuda:0"
    return {
        "mode": "gpu" if use_gpu else "cpu",
        "device": gpu_device if use_gpu else "cpu",
        "asr_provider": "cuda" if use_gpu else "cpu",
        # Gipformer's attention allocation is unsafe with larger CUDA batches;
        # keeping one also mirrors the notebook in CPU mode.
        "asr_batch_size": 1,
    }


def runtime_inference_label(use_gpu: bool) -> str:
    if use_gpu:
        return "GPU: DiariZen, CAM++ và Gipformer đều chạy CUDA"
    return "CPU: toàn bộ pipeline chạy trên CPU, không sử dụng CUDA"


def format_timing_summary(metrics: dict) -> list[str]:
    """Create compact, copyable timing rows for CPU/GPU comparison."""
    mode = str(metrics.get("runtime_mode", "unknown")).upper()
    runtime = float(metrics.get("runtime_seconds", 0.0) or 0.0)
    audio = float(metrics.get("audio_duration_seconds", 0.0) or 0.0)
    rtf = metrics.get("realtime_factor")
    speed = metrics.get("audio_seconds_per_runtime_second")
    summary = f"[{mode}] Tổng: {runtime:.3f}s • Audio: {audio:.3f}s"
    if rtf is not None:
        summary += f" • RTF: {float(rtf):.4f}"
    if speed is not None:
        summary += f" • Tốc độ: {float(speed):.2f}x realtime"
    rows = [summary]
    step_times = metrics.get("step_runtime_seconds", {})
    for key, label in TIMING_LABELS.items():
        if key in step_times:
            rows.append(f"{label}: {float(step_times[key]):.3f}s")
    return rows


def format_runtime_comparison(cpu: dict, gpu: dict) -> list[str]:
    """Compare two runs of the same audio, using CPU as the baseline."""
    cpu_total = float(cpu.get("runtime_seconds", 0.0) or 0.0)
    gpu_total = float(gpu.get("runtime_seconds", 0.0) or 0.0)
    if cpu_total <= 0 or gpu_total <= 0:
        return []
    ratio = cpu_total / gpu_total
    winner = (
        f"GPU nhanh hơn {ratio:.2f}x"
        if ratio >= 1.0
        else f"CPU nhanh hơn {1.0 / ratio:.2f}x"
    )
    rows = [
        f"Tổng: CPU {cpu_total:.3f}s • GPU {gpu_total:.3f}s • {winner}"
    ]
    cpu_steps = cpu.get("step_runtime_seconds", {})
    gpu_steps = gpu.get("step_runtime_seconds", {})
    for key, label in TIMING_LABELS.items():
        if key not in cpu_steps or key not in gpu_steps:
            continue
        cpu_time = float(cpu_steps[key])
        gpu_time = float(gpu_steps[key])
        if cpu_time <= 0 or gpu_time <= 0:
            continue
        rows.append(
            f"{label}: CPU {cpu_time:.3f}s • GPU {gpu_time:.3f}s • "
            f"CPU/GPU {cpu_time / gpu_time:.2f}x"
        )
    return rows


def run_flet_app() -> None:
    try:
        import flet as ft
    except ImportError as exc:
        raise RuntimeError("Hãy cài Flet bằng: pip install -r requirements.txt") from exc

    async def main(page: ft.Page) -> None:
        ensure_runtime_dirs()
        page.title = "Voice Identity Studio"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
        page.padding = 18
        try:
            page.window.width = 1280
            page.window.height = 860
            page.window.min_width = 920
            page.window.min_height = 680
        except Exception:
            pass

        picker = ft.FilePicker()
        page.services.append(picker)
        state = {
            "audio_path": None,
            "pending_samples": [],
            "runtime_profiles": [],
            "db_samples": [],
            "last_result": None,
            "audio_duration": None,
            "timing_by_mode": {},
        }
        run_lock = threading.Lock()

        audio_label = ft.Text("Chưa chọn file cuộc họp", color=ft.Colors.GREY_600)
        sample_label = ft.Text("Chưa chọn sample", size=12, color=ft.Colors.GREY_600)
        runtime_name = ft.TextField(label="Tên người nói", hint_text="Ví dụ: Hưng", expand=True)
        runtime_list = ft.Column(spacing=6)
        cluster_map = ft.TextField(
            label="Gộp cluster",
            hint_text="1=0, 4=0",
            helper="Có thể dùng số 1 thay cho SPEAKER_01",
            expand=True,
        )
        hard_override = ft.TextField(
            label="Ép tên cluster",
            hint_text="2=Hưng, 3=B",
            helper="Hard override được áp dụng sau majority vote",
            expand=True,
        )
        threshold = ft.TextField(
            label="Cosine threshold",
            value=str(SETTINGS.verification_threshold),
            width=165,
        )
        include_saved = ft.Switch(label="Dùng kho mẫu đã lưu", value=True)
        enhance = ft.Switch(label="Khử nhiễu", value=False)
        skip_asr = ft.Switch(label="Bỏ qua ASR", value=False)
        use_gpu = ft.Switch(
            label="Dùng GPU cho toàn pipeline",
            value=str(SETTINGS.device or "").startswith("cuda"),
        )
        runtime_hint = ft.Text(
            runtime_inference_label(bool(use_gpu.value)),
            size=12,
            color=ft.Colors.BLUE_700,
        )
        progress_bar = ft.ProgressBar(value=0, visible=False)
        status = ft.Text("Sẵn sàng", color=ft.Colors.GREY_700)
        run_button = ft.FilledButton("Chạy pipeline", icon=ft.Icons.PLAY_ARROW)
        result_path = ft.Text("", size=12, selectable=True, color=ft.Colors.GREY_700)
        results = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True, spacing=10)

        db_name = ft.TextField(label="Tên hiển thị", hint_text="Ví dụ: Hưng", expand=True)
        db_id = ft.TextField(label="Speaker ID", hint_text="Tự tạo nếu để trống", expand=True)
        db_sample_label = ft.Text("Chưa chọn sample", size=12, color=ft.Colors.GREY_600)
        db_status = ft.Text("", color=ft.Colors.GREY_700)
        db_list = ft.Column(spacing=8)
        enroll_button = ft.FilledButton("Lưu mẫu giọng", icon=ft.Icons.PERSON_ADD)

        def render_runtime_profiles() -> None:
            controls = []
            for index, item in enumerate(state["runtime_profiles"]):
                controls.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon(ft.Icons.RECORD_VOICE_OVER, color=ft.Colors.INDIGO),
                                ft.Column(
                                    [
                                        ft.Text(item["display_name"], weight=ft.FontWeight.W_600),
                                        ft.Text(
                                            f"{len(item['sample_paths'])} sample dùng cho lần chạy này",
                                            size=11,
                                            color=ft.Colors.GREY_600,
                                        ),
                                    ],
                                    spacing=2,
                                    expand=True,
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.DELETE_OUTLINE,
                                    tooltip="Xóa",
                                    on_click=lambda _, i=index: remove_runtime_profile(i),
                                ),
                            ]
                        ),
                        padding=8,
                        border=ft.Border.all(1, ft.Colors.GREY_300),
                        border_radius=8,
                    )
                )
            runtime_list.controls = controls or [
                ft.Text(
                    "Bạn có thể dùng kho mẫu đã lưu, hoặc thêm sample chỉ cho lần chạy này.",
                    size=12,
                    color=ft.Colors.GREY_600,
                )
            ]

        def remove_runtime_profile(index: int) -> None:
            state["runtime_profiles"].pop(index)
            render_runtime_profiles()
            page.update()

        def render_database() -> None:
            database = load_voice_db(VOICE_DB_PATH)
            controls = []
            for item in database.get("speakers", []):
                controls.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon(ft.Icons.PERSON, color=ft.Colors.TEAL_700),
                                ft.Column(
                                    [
                                        ft.Text(
                                            item.get("display_name", item["speaker_id"]),
                                            weight=ft.FontWeight.W_600,
                                        ),
                                        ft.Text(
                                            f"ID: {item['speaker_id']} • {len(item.get('sample_files', []))} sample",
                                            size=11,
                                            color=ft.Colors.GREY_600,
                                        ),
                                    ],
                                    spacing=2,
                                ),
                            ]
                        ),
                        padding=10,
                        border=ft.Border.all(1, ft.Colors.GREY_300),
                        border_radius=8,
                    )
                )
            db_list.controls = controls or [ft.Text("Kho mẫu hiện đang trống.")]

        async def pick_audio(_):
            files = await picker.pick_files(
                dialog_title="Chọn file cuộc họp",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=AUDIO_EXTENSIONS,
                allow_multiple=False,
            )
            if files and files[0].path:
                state["audio_path"] = files[0].path
                audio_label.value = files[0].path
                page.update()

        async def pick_runtime_samples(_):
            files = await picker.pick_files(
                dialog_title="Chọn sample giọng",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=AUDIO_EXTENSIONS,
                allow_multiple=True,
            )
            state["pending_samples"] = [item.path for item in files if item.path]
            sample_label.value = (
                f"Đã chọn {len(state['pending_samples'])} sample"
                if state["pending_samples"]
                else "Chưa chọn sample"
            )
            page.update()

        def add_runtime_profile(_):
            name = runtime_name.value.strip()
            if not name or not state["pending_samples"]:
                status.value = "Cần nhập tên và chọn ít nhất một sample."
                status.color = ft.Colors.RED_700
                page.update()
                return
            speaker_id = safe_speaker_id(name)
            state["runtime_profiles"] = [
                item for item in state["runtime_profiles"] if item["speaker_id"] != speaker_id
            ]
            state["runtime_profiles"].append(
                {
                    "speaker_id": speaker_id,
                    "display_name": name,
                    "sample_paths": list(state["pending_samples"]),
                }
            )
            state["pending_samples"] = []
            runtime_name.value = ""
            sample_label.value = "Chưa chọn sample"
            status.value = f"Đã thêm sample tạm cho {name}."
            status.color = ft.Colors.GREY_700
            render_runtime_profiles()
            page.update()

        def set_busy(busy: bool) -> None:
            run_button.disabled = busy
            use_gpu.disabled = busy
            progress_bar.visible = busy
            if not busy:
                progress_bar.value = 1
            page.update()

        def on_progress(number: int, total: int, key: str, message: str) -> None:
            progress_bar.value = max(0, min(1, (number - 1) / total))
            duration = state.get("audio_duration")
            long_audio_note = ""
            if (
                key == "diarization"
                and not use_gpu.value
                and duration
                and duration >= 10 * 60
            ):
                long_audio_note = (
                    f" • Audio {duration / 60:.1f} phút; chạy CPU có thể mất khá lâu. "
                    "App vẫn đang xử lý nếu CPU còn hoạt động."
                )
            status.value = f"Bước {number}/{total}: {message}{long_audio_note}"
            status.color = ft.Colors.INDIGO_700
            page.update()

        def run_worker() -> None:
            try:
                value = float(threshold.value.strip())
                runtime = runtime_inference_options(bool(use_gpu.value))
                requests = [
                    EnrollmentRequest(
                        speaker_id=item["speaker_id"],
                        display_name=item["display_name"],
                        sample_paths=tuple(item["sample_paths"]),
                    )
                    for item in state["runtime_profiles"]
                ]
                paths = run_pipeline(
                    state["audio_path"],
                    PipelineOptions(
                        cluster_mapping=parse_assignments(cluster_map.value),
                        hard_overrides=parse_assignments(hard_override.value),
                        verification_threshold=value,
                        include_saved_profiles=include_saved.value,
                        enhance_audio=enhance.value,
                        skip_asr=skip_asr.value,
                        device=str(runtime["device"]),
                        asr_provider=str(runtime["asr_provider"]),
                        asr_batch_size=int(runtime["asr_batch_size"]),
                    ),
                    enrollments=requests,
                    progress=on_progress,
                )
                payload = load_result(paths["result"])
                transcript = load_result(paths["transcript"])
                metrics = payload.get("metrics", {})
                timing_rows = format_timing_summary(metrics)
                timing_mode = metrics.get("runtime_mode")
                if timing_mode in {"cpu", "gpu"}:
                    state["timing_by_mode"][timing_mode] = {
                        "source_audio": payload.get("source_audio"),
                        "metrics": metrics,
                    }
                cpu_run = state["timing_by_mode"].get("cpu")
                gpu_run = state["timing_by_mode"].get("gpu")
                comparison_rows = []
                if (
                    cpu_run
                    and gpu_run
                    and cpu_run["source_audio"] == gpu_run["source_audio"]
                ):
                    comparison_rows = format_runtime_comparison(
                        cpu_run["metrics"], gpu_run["metrics"]
                    )
                state["last_result"] = str(paths["result"])
                clusters = payload.get("clusters", {})
                summary = " • ".join(
                    f"{key}: {value.get('speaker', 'Unknown')}"
                    for key, value in clusters.items()
                )
                reconciliation = payload.get("reconciliation", {})
                relationship_labels = {
                    "no_profiles": "Không có mẫu giọng – dùng tên Speaker tạm",
                    "no_speech": "Không phát hiện lời nói",
                    "more_clusters_than_profiles": "Số cụm nhiều hơn số mẫu đã đăng ký",
                    "fewer_clusters_than_profiles": "Số cụm ít hơn số mẫu đã đăng ký",
                    "equal_counts": "Số cụm bằng số mẫu đã đăng ký",
                }
                count_summary = (
                    f"{reconciliation.get('registered_profiles', 0)} mẫu giọng đã đăng ký • "
                    f"{reconciliation.get('diarized_clusters', 0)} cụm được phát hiện\n"
                    f"{relationship_labels.get(reconciliation.get('relationship'), 'Cần kiểm tra số người nói')}"
                )
                warning_rows = []
                for item in payload.get("warnings", []):
                    if item.get("severity") == "info":
                        continue
                    color = (
                        ft.Colors.RED_700
                        if item.get("severity") == "error"
                        else ft.Colors.ORANGE_800
                        if item.get("severity") == "warning"
                        else ft.Colors.BLUE_700
                    )
                    warning_rows.append(
                        ft.Text(
                            f"[{item.get('code', 'warning')}] {item.get('message', '')}",
                            size=12,
                            color=color,
                            selectable=True,
                        )
                    )
                results.controls = [
                    ft.Text("Kết quả", size=22, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        runtime_inference_label(runtime["mode"] == "gpu"),
                        color=ft.Colors.GREEN_700,
                    ),
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("Thời gian inference", weight=ft.FontWeight.BOLD),
                                *[
                                    ft.Text(row, size=12, selectable=True)
                                    for row in timing_rows
                                ],
                                ft.Text(
                                    "RTF càng thấp càng nhanh. Timing gồm cả tải model; "
                                    "nên chạy mỗi mode hai lần trên cùng file để so công bằng.",
                                    size=11,
                                    color=ft.Colors.GREY_600,
                                ),
                            ],
                            spacing=3,
                        ),
                        padding=10,
                        bgcolor=ft.Colors.BLUE_50,
                        border_radius=8,
                    ),
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("So sánh CPU và GPU", weight=ft.FontWeight.BOLD),
                                *[
                                    ft.Text(row, size=12, selectable=True)
                                    for row in comparison_rows
                                ],
                            ],
                            spacing=3,
                        ),
                        padding=10,
                        bgcolor=ft.Colors.GREEN_50,
                        border_radius=8,
                        visible=bool(comparison_rows),
                    ),
                    ft.Text(count_summary, color=ft.Colors.INDIGO_700),
                    ft.Text(
                        summary or "Không phát hiện người nói",
                        color=ft.Colors.GREY_700,
                    ),
                    ft.Container(
                        content=ft.Column(warning_rows, spacing=4),
                        padding=10,
                        bgcolor=ft.Colors.AMBER_50,
                        border_radius=8,
                        visible=bool(warning_rows),
                    ),
                    ft.Text(
                        f"Biên bản ASR • {len(transcript.get('segments', []))} lượt nói",
                        size=18,
                        weight=ft.FontWeight.BOLD,
                    ),
                    build_transcript_view(ft, transcript.get("segments", [])),
                    ft.Text(
                        "Thông tin kỹ thuật và cảnh báo được gom trong result.json.",
                        size=11,
                        color=ft.Colors.GREY_600,
                    ),
                ]
                result_path.value = (
                    f"Transcript TXT: {paths['transcript_text']}\n"
                    f"Transcript JSON: {paths['transcript']}\n"
                    f"Thông tin kỹ thuật: {paths['result']}"
                )
                has_important_warnings = bool(warning_rows)
                mode_name = str(runtime["mode"]).upper()
                status.value = (
                    f"Hoàn tất bằng {mode_name}, có cảnh báo cần kiểm tra."
                    if has_important_warnings
                    else f"Hoàn tất bằng {mode_name}. Kết quả đã được ghi ra thư mục job."
                )
                status.color = (
                    ft.Colors.ORANGE_800
                    if has_important_warnings
                    else ft.Colors.GREEN_700
                )
            except Exception as exc:
                status.value = f"Lỗi: {exc}"
                status.color = ft.Colors.RED_700
            finally:
                state["audio_duration"] = None
                if run_lock.locked():
                    run_lock.release()
                set_busy(False)

        def start_run(_):
            if not run_lock.acquire(blocking=False):
                status.value = "Pipeline đang chạy; không thể tạo thêm một job trùng."
                status.color = ft.Colors.ORANGE_800
                page.update()
                return
            if not state["audio_path"]:
                status.value = "Hãy chọn file cuộc họp trước."
                status.color = ft.Colors.RED_700
                page.update()
                run_lock.release()
                return
            try:
                float(threshold.value.strip())
                parse_assignments(cluster_map.value)
                parse_assignments(hard_override.value)
                try:
                    state["audio_duration"] = probe_duration(state["audio_path"])
                except Exception:
                    state["audio_duration"] = None
            except ValueError as exc:
                status.value = f"Cấu hình không hợp lệ: {exc}"
                status.color = ft.Colors.RED_700
                page.update()
                run_lock.release()
                return
            set_busy(True)
            progress_bar.value = 0
            try:
                page.run_thread(run_worker)
            except Exception:
                run_lock.release()
                set_busy(False)
                raise

        async def pick_db_samples(_):
            files = await picker.pick_files(
                dialog_title="Chọn sample để lưu",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=AUDIO_EXTENSIONS,
                allow_multiple=True,
            )
            state["db_samples"] = [item.path for item in files if item.path]
            db_sample_label.value = (
                f"Đã chọn {len(state['db_samples'])} sample"
                if state["db_samples"]
                else "Chưa chọn sample"
            )
            page.update()

        def enroll_worker() -> None:
            try:
                runtime = runtime_inference_options(bool(use_gpu.value))
                embedder = CAMPPlusEmbedder(device=str(runtime["device"]))
                identifier = db_id.value.strip() or safe_speaker_id(db_name.value)
                enroll_speaker(
                    identifier,
                    db_name.value.strip(),
                    state["db_samples"],
                    embedder,
                    VOICE_DB_PATH,
                )
                db_status.value = (
                    f"Đã lưu mẫu cho {db_name.value.strip()} bằng "
                    f"{str(runtime['mode']).upper()}."
                )
                db_status.color = ft.Colors.GREEN_700
                state["db_samples"] = []
                db_sample_label.value = "Chưa chọn sample"
                render_database()
            except Exception as exc:
                db_status.value = f"Lỗi enroll: {exc}"
                db_status.color = ft.Colors.RED_700
            finally:
                enroll_button.disabled = False
                page.update()

        def start_enroll(_):
            if not db_name.value.strip() or not state["db_samples"]:
                db_status.value = "Cần nhập tên và chọn ít nhất một sample."
                db_status.color = ft.Colors.RED_700
                page.update()
                return
            enroll_button.disabled = True
            db_status.value = "Đang trích xuất embedding CAM++..."
            db_status.color = ft.Colors.INDIGO_700
            page.update()
            page.run_thread(enroll_worker)

        def open_output(_):
            if not state["last_result"]:
                return
            folder = str(Path(state["last_result"]).parent)
            try:
                os.startfile(folder)
            except Exception:
                result_path.value = folder
                page.update()

        run_button.on_click = start_run
        enroll_button.on_click = start_enroll

        def update_runtime_mode(_=None) -> None:
            runtime_hint.value = runtime_inference_label(bool(use_gpu.value))
            runtime_hint.color = (
                ft.Colors.GREEN_700 if use_gpu.value else ft.Colors.BLUE_700
            )
            page.update()

        use_gpu.on_change = update_runtime_mode

        analysis_controls = ft.Column(
            [
                ft.ResponsiveRow(
                    [
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text("1. Audio cuộc họp", size=18, weight=ft.FontWeight.W_600),
                                    ft.Button(
                                        "Chọn audio",
                                        icon=ft.Icons.AUDIO_FILE,
                                        on_click=pick_audio,
                                    ),
                                    audio_label,
                                    ft.Divider(),
                                    ft.Text("2. Sample người nói", size=18, weight=ft.FontWeight.W_600),
                                    ft.Text(
                                        "Không bắt buộc. Nếu không có sample, app vẫn diarize + ASR và dùng tên tạm Speaker 1, Speaker 2...",
                                        size=12,
                                        color=ft.Colors.BLUE_700,
                                    ),
                                    ft.Row([runtime_name]),
                                    ft.Row(
                                        [
                                            ft.Button(
                                                "Chọn sample",
                                                icon=ft.Icons.UPLOAD_FILE,
                                                on_click=pick_runtime_samples,
                                            ),
                                            ft.FilledButton(
                                                "Thêm người",
                                                icon=ft.Icons.ADD,
                                                on_click=add_runtime_profile,
                                            ),
                                        ],
                                        wrap=True,
                                    ),
                                    sample_label,
                                    runtime_list,
                                ],
                                spacing=10,
                            ),
                            padding=16,
                            border=ft.Border.all(1, ft.Colors.GREY_300),
                            border_radius=12,
                            col={"sm": 12, "md": 5},
                        ),
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text("3. Tinh lọc & định danh", size=18, weight=ft.FontWeight.W_600),
                                    ft.Row([use_gpu]),
                                    runtime_hint,
                                    ft.Row([cluster_map, hard_override]),
                                    ft.Row([threshold, include_saved], wrap=True),
                                    ft.Row([enhance, skip_asr], wrap=True),
                                    ft.Divider(),
                                    run_button,
                                    progress_bar,
                                    status,
                                    ft.Row(
                                        [
                                            result_path,
                                            ft.IconButton(
                                                icon=ft.Icons.FOLDER_OPEN,
                                                tooltip="Mở thư mục output",
                                                on_click=open_output,
                                            ),
                                        ]
                                    ),
                                ],
                                spacing=12,
                            ),
                            padding=16,
                            border=ft.Border.all(1, ft.Colors.GREY_300),
                            border_radius=12,
                            col={"sm": 12, "md": 7},
                        ),
                    ],
                    spacing=12,
                    run_spacing=12,
                ),
                ft.Divider(),
                results,
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=12,
        )

        database_controls = ft.Column(
            [
                ft.Text("Kho mẫu giọng CAM++", size=22, weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Mẫu lưu ở đây sẽ tự động được dùng ở các lần phân tích tiếp theo.",
                    color=ft.Colors.GREY_600,
                ),
                ft.Row([db_name, db_id]),
                ft.Row(
                    [
                        ft.Button(
                            "Chọn sample",
                            icon=ft.Icons.UPLOAD_FILE,
                            on_click=pick_db_samples,
                        ),
                        enroll_button,
                    ],
                    wrap=True,
                ),
                db_sample_label,
                db_status,
                ft.Divider(),
                db_list,
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=12,
        )

        render_runtime_profiles()
        render_database()
        page.add(
            ft.Column(
                [
                    ft.Text("Voice Identity Studio", size=28, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "DiariZen diarization • CAM++ verification • Gipformer tiếng Việt",
                        color=ft.Colors.GREY_600,
                    ),
                    ft.Tabs(
                        content=ft.Column(
                            [
                                ft.TabBar(
                                    tabs=[
                                        ft.Tab(label="Phân tích", icon=ft.Icons.GRAPHIC_EQ),
                                        ft.Tab(label="Kho mẫu giọng", icon=ft.Icons.PEOPLE),
                                    ]
                                ),
                                ft.TabBarView(
                                    controls=[analysis_controls, database_controls],
                                    expand=True,
                                ),
                            ],
                            expand=True,
                        ),
                        length=2,
                        expand=True,
                    ),
                ],
                expand=True,
            )
        )

    ft.run(main)


def parse_assignments(value: str) -> dict[str, str]:
    output: dict[str, str] = {}
    if not value or not value.strip():
        return output
    for item in re.split(r"[,;\n]+", value):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"'{item}' phải có dạng FROM=TO")
        source, target = (part.strip() for part in item.split("=", 1))
        if not source or not target:
            raise ValueError(f"'{item}' phải có dạng FROM=TO")
        output[source] = target
    return output
