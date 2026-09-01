# Apple Silicon MLX quality recipe

Omi Med STT v1's q8 MLX runtime now reproduces the canonical GPU runtime within
0.11 absolute WER points while using the same model family and greedy decoder.
No dictionary, contextual biasing, or transcript correction was used.

## Measured result

Both rows below use the same frozen 1,513-file, 7.18-hour internal benchmark and
the same scorer.

| Runtime | WER | M-WER | Occurrence M-WER | Drug M-WER | Medical recall |
|---|---:|---:|---:|---:|---:|
| Canonical GPU, BF16 | 6.5408% | 2.2284% | 1.9150% | 4.7511% (21/442) | 97.7716% |
| Apple MLX q8 | 6.6473% | 2.1240% | 1.8802% | 4.5249% (20/442) | 97.8760% |

The Apple draw ran on an M4 Max with 64 GB unified memory using macOS 26.5.2,
Python 3.12.9, MLX 0.31.2, and parakeet-mlx 0.4.1. It processed 25,853.85
seconds of audio in 308.44 seconds wall time (83.82x realtime including file
loading and evaluation overhead; 112.10x across model calls) and peaked at
5,797.89 MiB of MLX memory.

## Exact runtime recipe

- Published `omi-health/omi-med-stt-v1-mlx-q8` weights (8-bit, group size 64)
- BF16 activations
- Greedy decoding
- Local relative-position attention with a 256/256 context
- 16 kHz mono input
- 400-sample periodic Hann window centered in a 512-point FFT
- Existing MLX magnitude, log, and per-feature normalization convention
- MLX cache cleared between files
- SentencePiece unknown tokens rendered with the reference-runtime convention

The implementation is in
[`src/omi_stt/mlx_runtime.py`](../src/omi_stt/mlx_runtime.py). From a source
checkout, the complete user-facing command remains:

```bash
pip install -e .
omi-med-stt recording.wav --runtime mlx
```

The cache clear is correctness-critical for sequential long-file workloads. In
the frozen draw, omitting it caused two long files to emit repeated unknown
tokens. With it enabled, all 1,513 files completed and the largest unknown-token
count in any transcript was 17.

## Full-attention ceiling

Full attention scored 6.6176% WER, 2.2284% M-WER, and 21/442 drug errors. It is
not the default because the 855.56-second longest benchmark file peaked near
19.8 GiB, versus about 5.8 GiB for the selected bounded-attention recipe.

These measurements are for runtime reproducibility, not clinical validation.
Transcripts still require review before clinical use.
