from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from omi_stt import audio
from omi_stt.audio import AudioDecodeError, find_ffmpeg, read_audio

HAVE_FFMPEG = find_ffmpeg() is not None


def _write_wav(path, *, sr: int = 22050, seconds: float = 0.5, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sr)
    return data


def _raise_sf_error(*_args, **_kwargs):
    # Build a catchable LibsndfileError instance without invoking its int-code __init__.
    raise audio._SOUNDFILE_ERROR.__new__(audio._SOUNDFILE_ERROR)


def test_wav_uses_soundfile_fast_path(tmp_path, monkeypatch) -> None:
    """WAV reads via libsndfile and must never invoke the ffmpeg fallback."""
    wav = tmp_path / "a.wav"
    _write_wav(wav, sr=16000)

    def _boom(*_a, **_k):
        raise AssertionError("ffmpeg fallback should not run for a WAV file")

    monkeypatch.setattr(audio, "_decode_with_ffmpeg", _boom)
    mono, sr = read_audio(wav)

    assert sr == 16000
    assert mono.dtype == np.float32 and mono.ndim == 1


def test_missing_ffmpeg_raises_actionable_error(tmp_path, monkeypatch) -> None:
    """If libsndfile can't open it and ffmpeg is absent, the error says how to fix it."""
    fake = tmp_path / "voice.m4a"
    fake.write_bytes(b"not really audio")

    monkeypatch.setattr(audio.sf, "read", _raise_sf_error)
    monkeypatch.setattr(audio, "find_ffmpeg", lambda: None)

    with pytest.raises(AudioDecodeError) as excinfo:
        read_audio(fake)

    msg = str(excinfo.value)
    assert "ffmpeg" in msg.lower()
    assert "brew install ffmpeg" in msg


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not available")
def test_ffmpeg_fallback_decodes_to_16k_mono(tmp_path, monkeypatch) -> None:
    """When libsndfile fails, ffmpeg decodes the file to 16 kHz mono float32 in [-1, 1]."""
    wav = tmp_path / "src.wav"
    _write_wav(wav, sr=22050, seconds=0.5)

    monkeypatch.setattr(audio.sf, "read", _raise_sf_error)  # force the fallback
    mono, sr = read_audio(wav)

    assert sr == 16000
    assert mono.dtype == np.float32 and mono.ndim == 1
    assert abs(len(mono) - 8000) < 600  # ~0.5 s at 16 kHz
    peak = float(np.max(np.abs(mono)))
    assert 0.01 < peak <= 1.0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not available")
def test_corrupt_file_raises_decode_error(tmp_path) -> None:
    """A non-audio file with an audio extension fails cleanly (soundfile + ffmpeg both reject)."""
    bad = tmp_path / "broken.m4a"
    bad.write_bytes(b"\x00\x01\x02 this is definitely not audio " * 16)

    with pytest.raises(AudioDecodeError):
        read_audio(bad)
