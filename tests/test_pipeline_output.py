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
from config.settings import SETTINGS


def test_default_voice_verification_threshold_is_point_four() -> None:
    assert SETTINGS.verification_threshold == 0.40


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
