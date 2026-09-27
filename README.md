# ZTE-STAR: trajectory and reconstruction

This repository provides the ZTE-STAR trajectory and reconstruction code together with representative simulated k-space data.

- `1_trajectory.ipynb`: paired-spoke spiral phyllotaxis trajectory generation and visualization
- `2_reconstruction.ipynb`: algebraic dead-time infilling and reconstruction of the included simulated data

## Data

`data/simulated/` contains static-phantom k-space sampled with the ZTE-STAR and conventional radial trajectories, together with the reference phantom and label image.

## Requirements

- Notebook 1: CPU only.
- Notebook 2: NVIDIA GPU with CUDA 12.
- The reference GPU environment is an NVIDIA RTX 4090.

## Running

With Docker:

```bash
docker build -t zte-star .
docker run --rm --gpus all -p 8888:8888 -v "$PWD:/work" zte-star
```

Open the Jupyter URL printed by Docker and run:

1. `1_trajectory.ipynb`
2. `2_reconstruction.ipynb`

Without Docker:

```bash
pip install -r requirements.txt
jupyter lab
```

## Main settings

| Setting | Value |
| --- | --- |
| Spiral phyllotaxis | `M = 144`, `P = 144`, `k = 55` |
| Total ZTE-STAR spokes | 41,472 |
| Readout samples | 128 per spoke |
| Conventional radial | 2,948 spokes per repetition |
| Dead-time infilling | `n_gap = 5`, readout oversampling = 4 |
| TTV reconstruction | 3 interleaves / frame = 864 spokes, `lambda = 1e-6`, 50 iterations |

## How to cite

If you use this code, please cite the associated article:

> *ZTE-STAR: 4D Zero-echo-time (ZTE) Imaging with a Spiral Phyllotaxis Trajectory using Temporal Total Variation Reconstruction and Algebraic Dead-time Infilling.* Journal of Magnetic Resonance Imaging.

Please use the final journal citation when available.

## License

MIT License. See `LICENSE`.
