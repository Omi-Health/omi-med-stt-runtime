# NVIDIA GPU quality recipe

Omi Med STT v1's NeMo adapter applies the selected GPU recipe automatically.
The public model checkpoint is unchanged; the settings below are runtime
configuration, not additional model weights or transcript post-processing.

## Run

Install the NVIDIA runtime extra and select `nemo` explicitly, or allow `auto`
to select it when `nvidia-smi` reports an NVIDIA GPU:

```bash
pip install -U "omi-med-stt[nemo]"
omi-med-stt recording.wav --runtime nemo
```

The exact benchmark environment used NeMo 3.0.0, PyTorch 2.8.0, and CUDA 12.8.
NeMo 3.0.0 is pinned by the package extra. Install the matching CUDA-enabled
PyTorch build for the host driver before installing the extra when necessary.

## Settings applied automatically

- BF16 inference on a BF16-capable NVIDIA GPU
- local relative-position attention with `[256,256]` context
- greedy-batch TDT decoding with `max_symbols=10`
- CUDA graph decoding disabled
- batches of at most eight inputs
- no more than 900 decoded audio-seconds in a formed batch
- duration sorting, followed by restoration of caller input order
- timestamps disabled
- the evaluated model repository revision pinned
- FFmpeg normalization to mono 16 kHz signed-PCM16 WAV
- complete-recording inference for the normal `transcribe` command

The complete-recording behavior matters: the older CLI automatically split a
NeMo recording after 240 seconds and merged the pieces. That was not the
qualified recipe and could alter words at every artificial boundary. Explicit
`transcribe-long` remains available when a caller deliberately wants chunks.

The implementation is in
[`src/omi_stt/nemo_runtime.py`](../src/omi_stt/nemo_runtime.py), with NVIDIA
input normalization selected in
[`src/omi_stt/cli.py`](../src/omi_stt/cli.py) and implemented in
[`src/omi_stt/audio.py`](../src/omi_stt/audio.py).

## Checks

Focused source checks:

```bash
PYTHONPATH=src pytest -q \
  tests/test_nemo_runtime.py \
  tests/test_audio_decode.py \
  tests/test_cli.py
```

Complete package check:

```bash
PYTHONPATH=src python scripts/prepublish_check.py --skip-build
python -m build --wheel
```

Check locations:

- GPU configuration, BF16 enforcement, batching, and order restoration:
  [`tests/test_nemo_runtime.py`](../tests/test_nemo_runtime.py)
- exact PCM16 normalization:
  [`tests/test_audio_decode.py`](../tests/test_audio_decode.py)
- runtime routing and no implicit NeMo chunking:
  [`tests/test_cli.py`](../tests/test_cli.py)
- package-wide verification entrypoint:
  [`scripts/prepublish_check.py`](../scripts/prepublish_check.py)

The implementation closeout passed 96/96 source tests, the prepublish check,
and wheel construction. Those checks were run on Apple Silicon and validate
configuration, control flow, packaging, and audio normalization. The next
release check on an NVIDIA host must additionally run a real CUDA transcription
smoke before publication.

No dictionary, contextual biasing, language-model rescoring, or transcript
rewrite is enabled by this runtime recipe.
