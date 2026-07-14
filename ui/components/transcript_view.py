from __future__ import annotations

import json
from pathlib import Path


def load_result(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_transcript_view(ft, rows: list[dict]):
    if not rows:
        return ft.Container(
            content=ft.Text("Không có segment nào trong kết quả."),
            padding=20,
        )
    controls = []
    palette = [
        ft.Colors.BLUE_700,
        ft.Colors.TEAL_700,
        ft.Colors.PURPLE_700,
        ft.Colors.ORANGE_800,
        ft.Colors.PINK_700,
        ft.Colors.INDIGO_700,
    ]
    clusters: dict[str, int] = {}
    for row in rows[:1000]:
        cluster = str(row.get("cluster", "")).strip()
        color_key = cluster or str(row.get("speaker", "Unknown"))
        if color_key not in clusters:
            clusters[color_key] = len(clusters)
        color = palette[clusters[color_key] % len(palette)]
        score = row.get("identity_confidence", row.get("score"))
        status = row.get("identity_status") or row.get("source") or row.get("status", "")
        meta = f"{_format_time(float(row.get('start', 0)))} – {_format_time(float(row.get('end', 0)))}"
        if cluster:
            meta += f"  •  cluster {cluster}"
        if score is not None and row.get("speaker") != "Unknown":
            meta += f"  •  độ tin cậy {float(score):.3f}"
        if row.get("asr_status") and row.get("asr_status") != "transcribed":
            meta += f"  •  ASR: {row.get('asr_status')}"
        provisional = str(row.get("provisional_speaker", "")).strip()
        if row.get("speaker") == "Unknown" and provisional:
            meta += f"  •  tên tạm: {provisional}"
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        controls.append(
            ft.Container(
                content=ft.Row(
                    [
                        ft.Container(width=5, bgcolor=color, border_radius=4),
                        ft.Column(
                            [
                                ft.Text(
                                    str(row.get("speaker", "Unknown")),
                                    size=16,
                                    weight=ft.FontWeight.W_600,
                                    color=color,
                                ),
                                ft.Text(meta, size=11, color=ft.Colors.GREY_600),
                                ft.Text(text, size=14, selectable=True),
                            ],
                            spacing=4,
                            expand=True,
                        ),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
                padding=12,
                border=ft.Border.all(1, ft.Colors.GREY_300),
                border_radius=10,
            )
        )
    return ft.Column(controls, spacing=8)


def _format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"
