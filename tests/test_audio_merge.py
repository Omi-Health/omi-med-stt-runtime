from __future__ import annotations

import numpy as np
import soundfile as sf

from omi_stt.audio import linear_resample, make_chunks, normalize_to_16k_mono
from omi_stt.merge import merge_transcripts


def test_linear_resample_changes_length() -> None:
    audio = np.linspace(-0.25, 0.25, 8000, dtype=np.float32)
    out = linear_resample(audio, 8000, 16000)

    assert out.dtype == np.float32
    assert len(out) == 16000


def test_make_chunks_uses_overlap(tmp_path) -> None:
    sample_rate = 16000
    audio = np.zeros(sample_rate * 10, dtype=np.float32)
    wav = tmp_path / "sample.wav"
    sf.write(wav, audio, sample_rate)

    chunks = make_chunks(wav, chunk_seconds=4, overlap=1, tmp_dir=tmp_path)

    assert [round(c.duration, 2) for c in chunks] == [4.0, 4.0, 4.0]
    assert all(c.path.exists() for c in chunks)


def test_normalize_to_16k_mono(tmp_path) -> None:
    sample_rate = 8000
    audio = np.column_stack([
        np.zeros(sample_rate, dtype=np.float32),
        np.ones(sample_rate, dtype=np.float32) * 0.1,
    ])
    wav = tmp_path / "stereo.wav"
    sf.write(wav, audio, sample_rate)

    info = normalize_to_16k_mono(wav, tmp_path)

    assert info.sample_rate == 16000
    assert round(info.duration, 2) == 1.0
    data, sr = sf.read(info.path, dtype="float32")
    assert sr == 16000
    assert data.ndim == 1


def test_merge_transcripts_removes_exact_overlap() -> None:
    merged = merge_transcripts([
        "the patient takes bisoprolol once daily",
        "bisoprolol once daily and uses an inhaler",
    ])

    assert merged == "the patient takes bisoprolol once daily and uses an inhaler"
