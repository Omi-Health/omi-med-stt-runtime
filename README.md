# Omi Med STT Runtime

[![PyPI](https://img.shields.io/pypi/v/omi-med-stt)](https://pypi.org/project/omi-med-stt/)
[![Tests](https://github.com/Omi-Health/omi-med-stt-runtime/actions/workflows/test.yml/badge.svg)](https://github.com/Omi-Health/omi-med-stt-runtime/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Command-line runtime for **Omi Med STT v1**, an English medical speech-to-text
model built from NVIDIA Parakeet TDT 0.6B v2.

The package downloads the right model artifact for your machine and transcribes
audio locally.

## Install

```bash
pip install -U omi-med-stt
```

Apple Silicon:

```bash
pip install -U "omi-med-stt[mlx]"
```

NVIDIA CUDA / NeMo:

```bash
pip install -U "omi-med-stt[nemo]"
```

## Run

```bash
omi-med-stt audio.wav
```

Useful options:

```bash
omi-med-stt audio.wav --json
omi-med-stt audio.wav --runtime mlx
omi-med-stt audio.wav --runtime nemo
omi-med-stt audio.wav --runtime cpp
omi-med-stt check
```

**Audio formats.** WAV, FLAC, OGG and other libsndfile formats are read directly. Other
inputs — `.m4a` (iPhone Voice Memos / QuickTime), `.mp3`, `.aac`, `.mp4`, `.mov`, `.wma`,
`.opus`, `.webm`, … — are decoded with **ffmpeg**, which ships with the package, so there's
nothing extra to install. If a system `ffmpeg` is on your `PATH` it's used instead (e.g. a
newer build). Whatever the input, audio is downmixed to mono and resampled to 16 kHz
automatically.

## Runtime Choices

| Platform | Default runtime | Model artifact |
|---|---|---|
| Apple Silicon | `mlx` | [`omi-health/omi-med-stt-v1-mlx-q8`](https://huggingface.co/omi-health/omi-med-stt-v1-mlx-q8) |
| NVIDIA CUDA | `nemo` | [`omi-health/omi-med-stt-v1`](https://huggingface.co/omi-health/omi-med-stt-v1) |
| Linux/Windows CPU | `cpp` | [`omi-health/omi-med-stt-v1-gguf`](https://huggingface.co/omi-health/omi-med-stt-v1-gguf) |

The canonical model is the NeMo checkpoint. MLX and GGUF are runtime exports.

CPU setup:

```bash
omi-med-stt install-cpp --cpp-backend cpu
omi-med-stt audio.wav --runtime cpp
```

The CPU path uses a patched `parakeet.cpp` runtime and downloads the q8_0 GGUF
artifact only. It does not download the NeMo or MLX weights.

## Runtime Quality

| Artifact | WER | M-WER | Drug M-WER | Medical Recall |
|---|---:|---:|---:|---:|
| **NeMo canonical** | **6.54%** | 2.23% | 4.75% | 97.77% |
| **MLX q8** | 6.65% | **2.12%** | **4.52%** | **97.88%** |

These numbers compare the released runtime artifacts against each other on the
same internal benchmark. Visit [omi.health](https://omi.health) for the broader
model evaluation and product context.

See the [Apple MLX quality recipe](docs/APPLE_MLX_QUALITY_RECIPE.md) for the
exact inference settings, environment, speed, memory, and reproducibility notes.

## Model Repositories

- Canonical NeMo: [`omi-health/omi-med-stt-v1`](https://huggingface.co/omi-health/omi-med-stt-v1)
- Apple Silicon q8: [`omi-health/omi-med-stt-v1-mlx-q8`](https://huggingface.co/omi-health/omi-med-stt-v1-mlx-q8)
- Apple Silicon full precision: [`omi-health/omi-med-stt-v1-mlx`](https://huggingface.co/omi-health/omi-med-stt-v1-mlx)
- Linux/Windows CPU GGUF: [`omi-health/omi-med-stt-v1-gguf`](https://huggingface.co/omi-health/omi-med-stt-v1-gguf)

If the model repositories are private before launch, authenticate first:

```bash
huggingface-cli login
```

## CUDA Note

If `--runtime nemo` fails with a CUDA driver mismatch, install a PyTorch wheel
matching your driver before installing the NeMo extra. For example, on CUDA 12.8
hosts:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -U "omi-med-stt[nemo]"
```

## Development

```bash
git clone https://github.com/Omi-Health/omi-med-stt-runtime
cd omi-med-stt-runtime
pip install -e ".[dev]"
python scripts/prepublish_check.py --skip-build
python -m pytest -q tests
```

## Safety

Omi Med STT v1 is speech-to-text only. It is not a diagnostic, triage,
prescribing, or clinical decision model, and it is not clinically validated.
Transcripts must be reviewed before any clinical use.

## License And Attribution

Runtime code is MIT licensed.

Model weights are CC-BY-4.0 and are derived from
[`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2).
Omi Med STT v1 is not an NVIDIA model.

The CPU runtime uses [`parakeet.cpp`](https://github.com/mudler/parakeet.cpp).
