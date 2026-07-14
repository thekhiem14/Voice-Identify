from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from core.models import ClusterIdentity, DiarizationSegment, SegmentIdentity, VoiceProfile
from core.pipeline import PipelineExecutionError, PipelineOptions, run_pipeline
from core.segment_processing import reconcile_speaker_counts, smooth_identities
from core.voice_id import (
    EnrollmentRequest,
    aggregate_cluster_segment_scores,
    assign_cluster_profiles,
    build_runtime_profiles,
    load_voice_db,
    load_voice_profiles,
    verify_segments,
)


def _identity(segment_id, cluster, start, end, profile, name, score=0.8):
    return SegmentIdentity(
        segment_id,
        cluster,
        start,
        end,
        end - start,
        raw_speaker_id=profile,
        raw_speaker=name,
        raw_score=score,
        raw_status="matched",
    )


def test_no_profiles_uses_stable_provisional_speaker_names() -> None:
    identities = [
        SegmentIdentity("s0", "10", 0, 1, 1, raw_status="no_profiles"),
        SegmentIdentity("s1", "2", 1, 2, 1, raw_status="no_profiles"),
    ]
    output, summaries = smooth_identities(identities, [])

    assert summaries["2"].speaker == "Speaker 1"
    assert summaries["10"].speaker == "Speaker 2"
    assert all(item.status == "diarization_only" for item in output)


def test_under_cluster_keeps_segment_level_identities() -> None:
    profiles = [
        VoiceProfile("a", "An", np.array([1.0, 0.0])),
        VoiceProfile("b", "Binh", np.array([0.0, 1.0])),
    ]
    identities = [
        _identity("s0", "0", 0, 4, "a", "An"),
        _identity("s1", "0", 4, 6, "b", "Binh"),
    ]
    output, summaries = smooth_identities(identities, profiles)

    assert summaries["0"].status == "mixed_cluster"
    assert [item.speaker for item in output] == ["An", "Binh"]
    assert all(item.source == "segment_level_mixed" for item in output)
    report = reconcile_speaker_counts(output, summaries, profiles)
    assert report["profiles_not_observed"] == []


def test_over_cluster_allows_one_profile_on_multiple_clusters() -> None:
    profile = VoiceProfile("a", "An", np.array([1.0, 0.0]))
    identities = [
        _identity("s0", "0", 0, 2, "a", "An"),
        _identity("s1", "1", 2, 4, "a", "An"),
    ]
    output, summaries = smooth_identities(identities, [profile])
    report = reconcile_speaker_counts(output, summaries, [profile])

    assert report["relationship"] == "more_clusters_than_profiles"
    assert report["split_identity_candidates"] == {"a": ["0", "1"]}


def test_sparse_matches_do_not_label_an_entire_cluster() -> None:
    profile = VoiceProfile("a", "An", np.array([1.0, 0.0]))
    identities = [
        _identity("matched", "0", 0, 2, "a", "An"),
        SegmentIdentity(
            "unknown-1", "0", 2, 10, 8, raw_status="below_threshold", raw_score=0.4
        ),
        SegmentIdentity(
            "unknown-2", "0", 10, 18, 8, raw_status="below_threshold", raw_score=0.3
        ),
    ]

    output, summaries = smooth_identities(identities, [profile])

    assert summaries["0"].status == "weak_evidence"
    assert summaries["0"].evidence_coverage < 0.35
    assert [item.speaker for item in output] == ["An", "Unknown", "Unknown"]


def test_global_cluster_mapping_uses_named_audio_profiles() -> None:
    profiles = [
        VoiceProfile("an", "An", np.array([1.0, 0.0])),
        VoiceProfile("binh", "Bình", np.array([0.0, 1.0])),
    ]
    references = {
        "0": ClusterIdentity(
            "0", None, "Unknown", "Speaker 1", None, "unknown",
            "cluster_enrollment", profile_scores={"an": 0.82, "binh": 0.31},
        ),
        "1": ClusterIdentity(
            "1", None, "Unknown", "Speaker 2", None, "unknown",
            "cluster_enrollment", profile_scores={"an": 0.44, "binh": 0.79},
        ),
    }

    assigned = assign_cluster_profiles(references, profiles, threshold=0.55)

    assert assigned["0"].speaker == "An"
    assert assigned["1"].speaker == "Bình"
    assert assigned["0"].source == "global_cluster_mapping"


def test_cluster_mapping_aggregates_best_segment_evidence() -> None:
    reference = ClusterIdentity(
        "0", None, "Unknown", "Speaker 1", None, "pending",
        "cluster_enrollment", profile_scores={"a": 0.45},
    )
    identities = [
        SegmentIdentity(
            f"s{i}", "0", i, i + 1, 1, scores={"a": score, "b": 0.3}
        )
        for i, score in enumerate([0.58, 0.57, 0.56, 0.55, 0.54, 0.2])
    ]

    output = aggregate_cluster_segment_scores(
        {"0": reference}, identities, top_k=5
    )

    assert output["0"].profile_scores["a"] == 0.56
    assert output["0"].profile_scores["b"] == 0.3


def test_weak_segment_votes_can_still_use_cluster_audio_mapping() -> None:
    profile = VoiceProfile("a", "An", np.array([1.0, 0.0]))
    identities = [
        _identity("matched", "0", 0, 1, "a", "An"),
        SegmentIdentity(
            "unknown", "0", 1, 10, 9, raw_status="below_threshold", raw_score=0.4
        ),
    ]
    reference = ClusterIdentity(
        "0", "a", "An", "Speaker 1", 0.72, "matched",
        "global_cluster_mapping", profile_scores={"a": 0.72},
    )

    output, summaries = smooth_identities(
        identities, [profile], cluster_references={"0": reference}
    )
    report = reconcile_speaker_counts(output, summaries, [profile])

    assert summaries["0"].status == "weak_evidence"
    assert summaries["0"].speaker == "An"
    assert report["cluster_speakers"] == {"0": "An"}
    assert output[1].speaker == "Unknown"


def test_invalid_runtime_sample_is_reported_not_fatal(tmp_path, monkeypatch) -> None:
    short_sample = tmp_path / "short.wav"
    sf.write(short_sample, np.ones(8000, dtype=np.float32) * 0.1, 16000)
    monkeypatch.setattr(
        "core.voice_id.convert_to_16k_mono", lambda path, *args, **kwargs: short_sample
    )

    class FakeEmbedder:
        model_id = "iic/speech_campplus_sv_zh_en_16k-common_advanced"
        sample_rate = 16000

        def embed_file(self, path):
            return np.array([1.0, 0.0], dtype=np.float32)

    profiles, issues = build_runtime_profiles(
        FakeEmbedder(),
        [EnrollmentRequest("short", "Short", (str(short_sample),))],
        include_saved=False,
        workspace=tmp_path,
    )

    assert profiles == []
    assert {item["code"] for item in issues} >= {
        "invalid_enrollment_sample",
        "profile_rejected",
    }


def test_campplus_voice_db_rejects_eres2net_embeddings(tmp_path) -> None:
    missing = load_voice_db(tmp_path / "missing.json")
    assert missing["model_id"] == (
        "iic/speech_campplus_sv_zh_en_16k-common_advanced"
    )

    old_database = tmp_path / "voice_db_eres2net.json"
    old_database.write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": "iic/speech_eres2net_sv_zh-cn_16k-common",
                "sample_rate": 16000,
                "speakers": [
                    {
                        "speaker_id": "old",
                        "display_name": "Old profile",
                        "embedding": [0.0] * 192,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="enroll lại mẫu bằng CAM\\+\\+"):
        load_voice_profiles(old_database)


def test_voice_verification_uses_campplus_batch_api(tmp_path) -> None:
    audio = tmp_path / "meeting.wav"
    sf.write(audio, np.ones(16000 * 2, dtype=np.float32) * 0.1, 16000)
    calls = []

    class FakeCAMPlus:
        device = "cpu"

        def embed_waveforms(self, waveforms, sample_rate, batch_size):
            calls.append((len(waveforms), sample_rate, batch_size))
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in waveforms]

        def embed_waveform(self, *args, **kwargs):
            pytest.fail("segment fallback should not run for a successful batch")

    identities = verify_segments(
        audio,
        [
            DiarizationSegment("0", 0.0, 1.0, 1.0),
            DiarizationSegment("0", 1.0, 2.0, 1.0),
        ],
        [VoiceProfile("an", "An", np.array([1.0, 0.0], dtype=np.float32))],
        FakeCAMPlus(),
        threshold=0.33,
        batch_size=16,
        log=lambda *_: None,
    )

    assert calls == [(2, 16000, 16)]
    assert [item.speaker for item in identities] == ["An", "An"]


def test_silent_audio_skips_diarization_and_returns_empty_result(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "silent.wav"
    sf.write(audio, np.zeros(16000 * 2, dtype=np.float32), 16000)
    monkeypatch.setattr("core.pipeline.convert_to_16k_mono", lambda *a, **k: audio)
    monkeypatch.setattr(
        "core.pipeline.run_diarization",
        lambda *a, **k: pytest.fail("DiariZen must not run for silent audio"),
    )

    paths = run_pipeline(
        audio,
        PipelineOptions(
            output_dir=str(tmp_path / "out"),
            job_name="silent",
            skip_asr=True,
        ),
    )
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))

    assert payload["segments"] == []
    assert {item["code"] for item in payload["warnings"]} >= {
        "silent_audio",
        "no_speech_detected",
    }


def test_missing_optional_enhancement_falls_back_to_original_audio(
    tmp_path, monkeypatch
) -> None:
    audio = tmp_path / "silent.wav"
    sf.write(audio, np.zeros(16000 * 2, dtype=np.float32), 16000)
    monkeypatch.setattr("core.pipeline.convert_to_16k_mono", lambda *a, **k: audio)
    monkeypatch.setattr(
        "core.pipeline.enhance_audio",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("optional enhancement packages are missing")
        ),
    )

    paths = run_pipeline(
        audio,
        PipelineOptions(
            output_dir=str(tmp_path / "out"),
            job_name="enhancement-fallback",
            enhance_audio=True,
            skip_asr=True,
        ),
    )
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))

    assert payload["status"] == "completed_with_warnings"
    assert "enhancement_skipped" in {
        item["code"] for item in payload["warnings"]
    }


def test_fatal_error_writes_failure_json(tmp_path) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(PipelineExecutionError) as caught:
        run_pipeline(
            missing,
            PipelineOptions(output_dir=str(tmp_path / "out"), job_name="failed"),
        )

    failure_path = caught.value.failure_path
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_type"] == "FileNotFoundError"
    assert not (failure_path.parent / "work").exists()


def test_identity_model_failure_falls_back_to_diarization_only(
    tmp_path, monkeypatch
) -> None:
    audio = tmp_path / "speech.wav"
    t = np.arange(16000 * 2, dtype=np.float32) / 16000
    sf.write(audio, 0.1 * np.sin(2 * np.pi * 220 * t), 16000)
    monkeypatch.setattr("core.pipeline.convert_to_16k_mono", lambda *a, **k: audio)
    monkeypatch.setattr(
        "core.pipeline.run_diarization",
        lambda *a, **k: [DiarizationSegment("0", 0, 2, 2)],
    )
    monkeypatch.setattr(
        "core.pipeline.CAMPPlusEmbedder",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    paths = run_pipeline(
        audio,
        PipelineOptions(
            output_dir=str(tmp_path / "out"),
            job_name="identity-partial",
            skip_asr=True,
            include_saved_profiles=False,
        ),
        enrollments=[EnrollmentRequest("a", "An", (str(audio),))],
    )
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))

    assert payload["segments"][0]["speaker"] == "Speaker 1"
    assert "identity_stage_unavailable" in {
        item["code"] for item in payload["warnings"]
    }


def test_asr_model_failure_keeps_identity_timeline(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "speech.wav"
    t = np.arange(16000 * 2, dtype=np.float32) / 16000
    sf.write(audio, 0.1 * np.sin(2 * np.pi * 220 * t), 16000)
    monkeypatch.setattr("core.pipeline.convert_to_16k_mono", lambda *a, **k: audio)
    monkeypatch.setattr(
        "core.pipeline.run_diarization",
        lambda *a, **k: [DiarizationSegment("0", 0, 2, 2)],
    )
    monkeypatch.setattr(
        "core.pipeline.GipformerASR",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ASR unavailable")),
    )

    paths = run_pipeline(
        audio,
        PipelineOptions(
            output_dir=str(tmp_path / "out"),
            job_name="asr-partial",
            include_saved_profiles=False,
        ),
    )
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))

    assert payload["segments"][0]["asr_status"] == "unavailable"
    assert payload["segments"][0]["speaker"] == "Speaker 1"
    assert "asr_unavailable" in {item["code"] for item in payload["warnings"]}
