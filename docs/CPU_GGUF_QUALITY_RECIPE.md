# CPU GGUF quality recipe

`omi-med-stt` 0.1.25 uses the unchanged public q8_0 GGUF artifact through
`parakeet.cpp`. Files up to 240 seconds are decoded whole. Longer files use
overlapping 180-second chunks whose cuts snap to the quietest 300 ms window in
the 12 seconds before each nominal boundary. The operation is deterministic,
uses only the input waveform, and does not alter transcript text after decode.

## Qualified result

The 0.1.25 CPU path was run on the same internal 1,513-file board and scorer as
the CUDA and MLX runtime recipes:

| WER | Canonical M-WER | Occurrence M-WER | Medical errors | Drug errors | Medical recall |
|---:|---:|---:|---:|---:|---:|
| 7.10% | 2.16% | 1.71% | 49 / 2,872 | 19 / 442 | 97.84% |

The prior fixed-boundary CPU path scored 7.17% WER, 2.12% canonical M-WER,
52 medical errors, and 21 drug errors. Silence-aware cuts improved WER and
occurrence counts, while canonical M-WER moved by one borderline term. On a
separate non-board long-form holdout, silence-aware cuts also improved WER
(7.32% to 7.25%) and canonical M-WER (0.91% to 0.45%) without a speed penalty.

Do not interpret the 49/19 counts as proof that CPU is clinically better than
CUDA or MLX. Paired occurrence comparisons were not significant in this draw
(CPU versus CUDA: p=0.21; CPU versus MLX: p=0.40). The honest runtime choice is:

- CUDA for the best WER and throughput;
- MLX q8 for the best balanced Apple Silicon path and canonical M-WER;
- CPU GGUF when portability matters and no supported accelerator is present.

## Why optimization stops here

The remaining roughly 0.35 WER-point gap to feasible whole-file CPU decoding
comes from context lost across chunk boundaries. `parakeet.cpp`'s current
limited-attention mode still materializes the full quadratic attention matrix,
so changing its mask would not reduce memory or speed and would introduce an
unqualified attention geometry. Closing the residual gap requires genuine
sparse/windowed attention in the C++ engine, not another runtime flag.

## Verification

```bash
python -m pip install -U "omi-med-stt==0.2.0"
omi-med-stt install-cpp --cpp-backend cpu
omi-med-stt check
omi-med-stt consultation.wav --runtime cpp
```

For a source checkout, ensure tests import the checkout rather than an older
globally installed package:

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=src python -m pytest -q tests
```

The release gate is 61 passing tests. Model weights did not change; only the
runtime and long-audio boundary policy changed.
