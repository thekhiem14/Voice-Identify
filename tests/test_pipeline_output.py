from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from core.models import DiarizationSegment, SegmentIdentity
from core.pipeline import (
    PipelineOptions,
    _build_public_transcript_segments,
    run_pipeline,
)
from config.settings import SETTINGS, VOICE_DB_PATH
from ui.main_layout import runtime_inference_label, runtime_inference_options


def test_default_voice_verification_threshold_matches_campplus_checkpoint() -> None:
    assert SETTINGS.verification_threshold == 0.33
    assert SETTINGS.campplus_model_id == (
        "iic/speech_campplus_sv_zh_en_16k-common_advanced"
    )
    assert VOICE_DB_PATH.name == "voice_db_campplus.json"
    assert SETTINGS.voice_id_batch_size == 16


def test_notebook_asr_operating_point_is_the_default() -> None:
    assert SETTINGS.merge_max_gap_sec == 2.0
    assert SETTINGS.merge_max_duration_sec == 20.0
    assert SETTINGS.asr_min_duration_sec == 1.5
    assert SETTINGS.asr_provider == "cpu"
    assert SETTINGS.asr_batch_size == 1
    options = PipelineOptions()
    assert options.asr_provider == "cpu"
    assert options.asr_batch_size == 1


def test_ui_runtime_toggle_switches_every_inference_stage() -> None:
    cpu = runtime_inference_options(False)
    gpu = runtime_inference_options(True)

    assert cpu == {
        "mode": "cpu",
        "device": "cpu",
        "asr_provider": "cpu",
        "asr_batch_size": 1,
    }
    assert gpu["mode"] == "gpu"
    assert str(gpu["device"]).startswith("cuda")
    assert gpu["asr_provider"] == "cuda"
    assert gpu["asr_batch_size"] == 1
    assert "không sử dụng CUDA" in runtime_inference_label(False)
    assert "đều chạy CUDA" in runtime_inference_label(True)
    assert SETTINGS.diarization_cpu_batch_size == 1
    assert SETTINGS.device == "cpu"


def test_public_transcript_has_minimal_shape_and_merges_same_speaker() -> None:
    rows = [
        SegmentIdentity(
            "s0", "1", 0, 5, 5, speaker="Unknown",
            provisional_speaker="Speaker 2", text="XIN CHÀO MỌI NGƯỜI",
            asr_status="transcribed",
        ),
        SegmentIdentity(
            "s1", "1", 5.5, 9, 3.5, speaker="Unknown",
            provisional_speaker="Speaker 2", text="MỌI NGƯỜI HÔM NAY",
            asr_status="transcribed",
        ),
        SegmentIdentity(
            "s2", "0", 10, 12, 2, speaker="Unknown",
            provisional_speaker="Speaker 1", text="VÂNG",
            asr_status="transcribed",
        ),
    ]
    output = _build_public_transcript_segments(
        rows,
        {"weak_evidence_clusters": ["1"]},
        max_gap_seconds=2,
        max_segment_seconds=20,
    )

    assert output == [
        {
            "speaker": "S2",
            "start": 0.0,
            "end": 9.0,
            "text": "XIN CHÀO MỌI NGƯỜI HÔM NAY",
        },
        {"speaker": "S1", "start": 10.0, "end": 12.0, "text": "VÂNG"},
    ]


def test_pipeline_writes_final_json_without_heavy_models(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "meeting.wav"
    t = np.arange(16_000 * 3, dtype=np.float32) / 16_000
    sf.write(audio, 0.1 * np.sin(2 * np.pi * 220 * t), 16_000)

    monkeypatch.setattr(
        "core.pipeline.convert_to_16k_mono",
        lambda *args, **kwargs: audio,
    )
    monkeypatch.setattr(
        "core.pipeline.run_diarization",
        lambda *args, **kwargs: [
            DiarizationSegment("SPEAKER_00", 0, 1.2, 1.2),
            DiarizationSegment("SPEAKER_00", 1.5, 2.8, 1.3),
        ],
    )

    paths = run_pipeline(
        audio,
        PipelineOptions(
            output_dir=str(tmp_path / "outputs"),
            job_name="unit-test",
            hard_overrides={"0": "Hưng"},
            skip_identify=True,
            skip_asr=True,
        ),
    )
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))

    assert paths["result"].exists()
    assert paths["transcript"].exists()
    assert paths["transcript_text"].exists()
    assert set(paths) == {"result", "transcript", "transcript_text"}
    assert not (paths["result"].parent / "work").exists()
    assert payload["schema_version"] == "1.1"
    assert len(payload["segments"]) == 1
    assert payload["segments"][0]["speaker"] == "Hưng"
    assert payload["segments"][0]["asr_status"] == "disabled"
    transcript = json.loads(paths["transcript"].read_text(encoding="utf-8"))
    assert transcript == {"segments": []}


def test_pipeline_keeps_voice_id_windows_separate_from_asr_segments(
    tmp_path, monkeypatch
) -> None:
    audio = tmp_path / "meeting.wav"
    enhanced_audio = tmp_path / "meeting_enhanced.wav"
    t = np.arange(16_000 * 12, dtype=np.float32) / 16_000
    sf.write(audio, 0.1 * np.sin(2 * np.pi * 220 * t), 16_000)
    sf.write(enhanced_audio, 0.08 * np.sin(2 * np.pi * 220 * t), 16_000)
    asr_calls: list[list[float]] = []
    asr_audio_calls = []

    monkeypatch.setattr(
        "core.pipeline.convert_to_16k_mono",
        lambda *args, **kwargs: audio,
    )
    monkeypatch.setattr(
        "core.pipeline.run_diarization",
        lambda *args, **kwargs: [
            DiarizationSegment("0", 0, 6, 6),
            DiarizationSegment("0", 6.2, 12, 5.8),
        ],
    )
    monkeypatch.setattr(
        "core.pipeline.enhance_audio",
        lambda *args, **kwargs: (enhanced_audio, {"enabled": True}),
    )

    class FakeASR:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe_segments(self, _audio, segments, **kwargs):
            asr_audio_calls.append(_audio)
            asr_calls.append([item.duration for item in segments])
            for item in segments:
                item.text = "MỘT CÂU ĐỦ NGỮ CẢNH"
                item.asr_status = "transcribed"
            return segments, 0

    monkeypatch.setattr("core.pipeline.GipformerASR", FakeASR)

    paths = run_pipeline(
        audio,
        PipelineOptions(
            output_dir=str(tmp_path / "outputs"),
            job_name="separate-timelines",
            skip_identify=True,
            enhance_audio=True,
        ),
    )
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))

    assert asr_calls == [[12.0]]
    assert asr_audio_calls == [audio]
    assert len(payload["segments"]) == 1
    assert len(payload["identity_windows"]) == 2
    assert payload["segments"][0]["text"] == "MỘT CÂU ĐỦ NGỮ CẢNH"
    assert {
        item["asr_status"] for item in payload["identity_windows"]
    } == {"not_applicable_identity_window"}
    assert payload["metrics"]["identity_segments"] == 2
    assert payload["metrics"]["asr_segments"] == 1
    assert payload["metrics"]["options"]["asr_provider"] == "cpu"
    assert payload["metrics"]["options"]["asr_batch_size"] == 1
