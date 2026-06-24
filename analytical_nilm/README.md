# Analytical NILM Method Implementation

This folder contains the released Python implementation of the core method in
the paper **"An Analytical Federated Learning Method for Scalable
Non-Intrusive Load Monitoring"**.

The code is method-focused. It provides the shared feature extractor
pretraining flow and the analytical federated learning flow with synthetic toy
data, so the full method pipeline can be executed without private NILM data.

## Scope

Included in this folder:

- Patch-based shared feature extractor for one-dimensional NILM waveforms.
- Patch embedding and sinusoidal positional encoding.
- 1D convolutional residual feature extraction backbone.
- Pretraining with cross-entropy loss and supervised contrastive loss.
- Analytical local update for client-side closed-form learning.
- Cloud-side analytical aggregation.
- Regularization cleanup for the aggregated analytical classifier.
- Projection/update branch that concatenates raw waveform input and learned features.
- Synthetic toy demos for pretraining and analytical federated learning.

Not included in this folder:

- Raw NILM data or private household data.
- Private data split files.
- Full paper-table reproduction scripts.
- Baseline method implementations.
- Training logs or private checkpoints.
- STM32 embedded method code. See `../stm32_deployment/embedded_method/`
  for the released method-level STM32 C implementation.

## Relation to the Paper

The files in this folder correspond to the method components of the paper:

| Paper method component | Code location |
| --- | --- |
| Shared feature extractor architecture | `pretrain/models.py`, `pretrain/layers.py` |
| Patch embedding and positional encoding | `pretrain/layers.py` |
| 1D convolutional residual feature extraction | `pretrain/models.py` |
| Feature extractor pretraining with CE + SCL | `pretrain/losses.py`, `pretrain/trainer.py`, `examples/toy_pretrain.py` |
| Analytical local update | `federated/analytical_federated.py` |
| Cloud-side analytical aggregation | `federated/analytical_federated.py` |
| Regularization cleanup | `federated/analytical_federated.py` |
| Projection/update branch analytical classifier | `federated/analytical_federated.py`, `examples/toy_federated_learning.py` |
| Synthetic runnable method demo | `examples/toy_pretrain.py`, `examples/toy_federated_learning.py` |

The synthetic demos are intended to verify the implementation flow. They do not
reproduce the experimental tables in the paper.

## Folder Structure

```text
analytical_nilm/
├── configs/
│   ├── pretrain_config.yaml          # Toy pretraining parameters
│   └── federated_config.yaml         # Toy analytical FL parameters
├── examples/
│   ├── toy_pretrain.py               # Synthetic pretraining demo
│   └── toy_federated_learning.py     # Synthetic analytical FL demo
├── federated/
│   ├── __init__.py
│   └── analytical_federated.py       # Local update, aggregation, cleanup, evaluation
├── pretrain/
│   ├── __init__.py
│   ├── layers.py                     # Patch embedding and positional encoding
│   ├── losses.py                     # CE and supervised contrastive loss
│   ├── models.py                     # Patch residual feature extractor
│   └── trainer.py                    # Pretraining loop and checkpoint saving
├── outputs/
│   └── .gitkeep                      # Runtime outputs are ignored by git
├── utils.py                          # Config, synthetic data, splitting, normalization
└── README.md
```

## Quick Start

Run the commands from the repository root.

Install the minimal Python dependencies:

```bash
pip install torch numpy pyyaml
```

Step 1: run toy pretraining.

```bash
python analytical_nilm/examples/toy_pretrain.py --config analytical_nilm/configs/pretrain_config.yaml
```

This creates a toy checkpoint at:

```text
analytical_nilm/outputs/pretrain/checkpoints/patch_residual_toy.pth
```

Step 2: run toy analytical federated learning.

```bash
python analytical_nilm/examples/toy_federated_learning.py --config analytical_nilm/configs/federated_config.yaml
```

The federated demo loads the checkpoint from Step 1, prepares synthetic client
data, computes local analytical updates, aggregates them on the cloud side, and
evaluates the aggregated analytical classifier.

Typical output files:

```text
analytical_nilm/outputs/pretrain/toy_pretrain_metrics.json
analytical_nilm/outputs/pretrain/toy_pretrain_summary.txt
analytical_nilm/outputs/federated/toy_federated_metrics.json
analytical_nilm/outputs/federated/toy_federated_summary.txt
analytical_nilm/outputs/federated/toy_local_update_shapes.json
```

Runtime outputs are ignored by git except for `outputs/.gitkeep`.

## Configuration

The demo uses two separate config files:

- `configs/pretrain_config.yaml`: controls synthetic pretraining, model size,
  optimizer settings, CE + SCL temperature, checkpoint path, and output path.
- `configs/federated_config.yaml`: controls synthetic client splitting,
  analytical regularization, number of clients, non-IID Dirichlet split
  parameters, checkpoint loading, and output path.

Important federated parameters:

- `num_clients`: number of synthetic clients.
- `niid`: whether to use non-IID Dirichlet client splitting.
- `alpha`: Dirichlet concentration parameter. Smaller values create more
  heterogeneous client label distributions.
- `rg`: regularization coefficient used in local closed-form updates.
- `clean_reg`: whether to remove the local regularization contribution after
  aggregation.
- `use_projection`: kept true for the released toy flow.

## Synthetic Data

Both demos use generated one-dimensional waveforms from `utils.py`.

The generator creates class-dependent templates using sine waves, harmonics,
square-like patterns, and pulse-like components. Each sample is then perturbed
with random amplitude, offset, and Gaussian noise. This keeps the tensor shapes
and method flow close to the real NILM pipeline while avoiding any dependency on
private data.

## Notes

- The released Python code is intended for method inspection and runnable toy
  validation.
- The synthetic demos are not intended to reproduce paper metrics.
