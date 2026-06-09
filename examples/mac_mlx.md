# Mac / MLX Example

```bash
pip install -U "omi-med-stt[mlx]"
omi-med-stt check
omi-med-stt consult.wav
```

Equivalent explicit command:

```bash
omi-med-stt consult.wav --runtime mlx --model omi-health/omi-med-stt-v1-mlx
```

Omi Med STT v1 contains a rank-128 medical adapter. The `omi-med-stt` MLX runtime installs that adapter before loading the MLX weights. Do not call stock `parakeet-mlx` directly for this model.
