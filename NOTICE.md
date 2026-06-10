# Notices

This repository contains runtime code for Omi Med STT v1. It does not contain
model weights.

## NVIDIA NeMo / Parakeet

Omi Med STT v1 is derived from `nvidia/parakeet-tdt-0.6b-v2`.

The model weights are governed by the upstream NVIDIA Parakeet model license
and the Omi Med STT model card. At the time of this release, the base Parakeet
v2 model weights are listed as CC-BY-4.0.

NVIDIA, NeMo, and Parakeet are trademarks or names of their respective owners.
This is not an NVIDIA model.

## parakeet.cpp

The GGUF / C++ runtime path is powered by `parakeet.cpp`:

https://github.com/mudler/parakeet.cpp

`parakeet.cpp` is released under the MIT License.

Copyright (c) 2026 the parakeet.cpp authors

The `parakeet-cpp-omi-adapter.patch` file in this repository adapts
`parakeet.cpp` to execute Omi Med STT v1's post-Conformer medical adapter. The
intent is to upstream this support where practical.

## parakeet-mlx

The Apple Silicon MLX runtime interoperates with `parakeet-mlx` / MLX Parakeet
model exports.

Users should consult the upstream project licenses for those dependencies.

## FFmpeg / imageio-ffmpeg

`omi-med-stt` uses `imageio-ffmpeg` to locate a bundled FFmpeg executable when
no system `ffmpeg` is available. This is used as a subprocess for decoding
common clinician audio formats such as `.m4a`, `.mp3`, `.aac`, `.mp4`, and
`.mov` into 16 kHz mono PCM before transcription.

`imageio-ffmpeg` is distributed under the BSD-2-Clause License. Its
platform-specific wheels may include an FFmpeg executable; FFmpeg is governed
by its own upstream license terms.
