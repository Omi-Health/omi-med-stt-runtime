# Omi Med STT Runtime

[![PyPI](https://img.shields.io/pypi/v/omi-med-stt)](https://pypi.org/project/omi-med-stt/)
[![Tests](https://github.com/Omi-Health/omi-med-stt-runtime/actions/workflows/test.yml/badge.svg)](https://github.com/Omi-Health/omi-med-stt-runtime/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Command-line runtime for **Omi Med STT v1**, an English medical speech-to-text
model built from NVIDIA Parakeet TDT 0.6B v2.

The package downloads the right model artifact for your machine and transcribes
audio locally.

**0.2.1** refreshes the public evaluation text across PyPI and the model cards.
The runtime code and published model weights are unchanged from **0.2.0**.

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

The NVIDIA adapter applies the qualified GPU recipe automatically: NeMo 3.0,
BF16, local `[256,256]` attention, greedy-batch TDT decoding with
`max_symbols=10`, timestamps disabled, and duration-sorted batches capped at
eight files or 900 audio-seconds. Inputs are normalized through FFmpeg to mono
16 kHz PCM16. A BF16-capable NVIDIA GPU is required; no inference flags are
needed beyond `--runtime nemo`.

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
artifact only. It does not download the NeMo or MLX weights. Unknown tokens are
rendered as the same U+2047 marker the NeMo and MLX runtimes emit (rendering
parity, not transcript correction).

## Runtime Quality

| Artifact | WER | M-WER | Drug M-WER | Medical Recall |
|---|---:|---:|---:|---:|
| **NeMo canonical** | **6.54%** | 2.23% | 4.75% | 97.77% |
| **MLX q8** | 6.65% | **2.12%** | **4.52%** | **97.88%** |
| **GGUF q8_0 / CPU** | 7.10% | 2.16% | **4.30%** | 97.84% |

These numbers compare the unchanged runtime artifacts on the same frozen
1,513-clip, 7.18-hour medical benchmark and scorer, using the runtime recipes
shipped in this package. No dictionary, custom vocabulary, contextual bias, or
transcript correction was used.
The CPU row uses the silence-aware long-audio chunking shipped in `0.1.25`.
Its lower drug-error count in this draw is not a statistically established
ranking over GPU or MLX; the GPU remains the best overall WER and throughput
path, while MLX q8 is the selected Apple runtime.

Compared with the open-model rows on Omi's standing 30-system board, the CUDA
and MLX q8 runtimes have the lowest observed WER, while MLX q8 has the
second-lowest observed M-WER. These are positions in this benchmark draw, not a
universal ranking.

See the [full benchmark](https://omi.health/benchmark) and
[runtime-specific results](https://omi.health/research/omi-med-stt#runtime-results)
for the broader evaluation and product context.

Runtime recipes and checks:

- [NVIDIA GPU quality recipe](docs/NVIDIA_GPU_QUALITY_RECIPE.md)
- [Apple MLX q8 quality recipe](docs/APPLE_MLX_QUALITY_RECIPE.md)
- [CPU GGUF quality recipe](docs/CPU_GGUF_QUALITY_RECIPE.md)

The recipe pages include the exact runtime settings, verification commands,
and paths to the tests that enforce them.

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
