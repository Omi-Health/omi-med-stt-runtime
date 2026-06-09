# Omi Med STT Runtime - Agent Notes

This is the public runtime repository for **Omi Med STT v1**.

Keep this repo focused on the runtime package:

- Python CLI package: `omi-med-stt`
- runtime adapters for MLX, NeMo, and parakeet.cpp/GGUF
- public examples and unit tests
- minimal public runtime documentation

Do not add private research notes, benchmark manifests, local machine paths,
cloud resource details, API keys, private transcripts, or raw evaluation output.

## Model Repositories

Use the public launch name **Omi Med STT v1**.

| Purpose | Repository |
|---|---|
| Canonical NeMo checkpoint | `omi-health/omi-med-stt-v1` |
| Default Apple Silicon MLX q8 | `omi-health/omi-med-stt-v1-mlx-q8` |
| Full precision MLX export | `omi-health/omi-med-stt-v1-mlx` |
| GGUF q8_0/f16 export | `omi-health/omi-med-stt-v1-gguf` |

Do not reintroduce the retired name `omi-health/omi-stt-v1`.

## Runtime Defaults

`omi-med-stt audio.wav` auto-selects:

- Apple Silicon: `mlx`
- Linux/Windows CPU: `cpp`
- Linux with NVIDIA GPU: `nemo`

Manual runtime selection:

```bash
omi-med-stt audio.wav --runtime mlx
omi-med-stt audio.wav --runtime cpp
omi-med-stt audio.wav --runtime nemo
```

## Public Documentation Rule

This repo should show only the small runtime-artifact comparison needed by users:

| Artifact | WER | M-WER | Drug M-WER | Medical Recall |
|---|---:|---:|---:|---:|
| NeMo canonical | 8.30% | 2.37% | 4.75% | 97.95% |
| MLX full precision | 8.59% | 2.65% | 5.20% | 97.70% |
| MLX q8 | 8.61% | 2.75% | 5.20% | 97.63% |
| GGUF q8_0 | 9.12% | 3.20% | 6.33% | 97.53% |

For broader benchmark comparisons, point users to the Omi website. Do not add
large benchmark tables or internal evaluation history to this repo.

## Checks Before Publishing

```bash
pip install -e ".[dev]"
python scripts/prepublish_check.py --skip-build
pytest -q tests
```

Do not trigger model-download smoke tests for README-only or metadata-only edits.
