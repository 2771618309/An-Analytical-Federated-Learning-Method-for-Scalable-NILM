# An Analytical Federated Learning Method for Scalable NILM

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-green.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Method_Implementation-green.svg)](#)

---

> ### ✍️ Authors
> **Wenlong Guo**, **Tao Yu** *(Senior Member, IEEE)*, **Qingquan Luo**, **Xiaolei Hu**, **Yufeng Wu** *(Graduate Student Member, IEEE)*, and **Zhenning Pan** *(Member, IEEE)*

---

This repository provides the released method implementation and selected supplementary materials for:

**"An Analytical Federated Learning Method for Scalable Non-Intrusive Load Monitoring"**

The proposed method addresses the global-local bias issue in gradient-based federated NILM and provides an analytical federated learning pipeline with performance aligned with centralized learning.

## ✅ Code Availability Statement

This repository currently releases the **core method implementation** and selected supplementary materials.

The released method code includes:

- Python implementation of the shared patch-residual feature extractor pretraining pipeline.
- Patch embedding and positional encoding modules.
- 1D convolutional residual feature extraction backbone.
- Cross-entropy and supervised contrastive pretraining loss.
- Analytical local update for client-side closed-form learning.
- Cloud-side closed-form aggregation.
- Regularization cleanup for the aggregated analytical solution.
- Update-branch / projection-branch analytical classifier reparameterization.
- Synthetic toy demos that run without private NILM data.
- Method-level STM32 C implementation for PatchResNet forward inference, local analytical update, and matrix upload helpers.

This release is intended to provide the **core method implementation**, not a full reproduction package for all paper tables.

The repository does **not** include raw data, private household data, data split files, complete experiment reproduction scripts, baseline implementations, paper-table reproduction commands, training logs, private checkpoints, real deployment weights, company product code, or a complete Keil/STM32CubeIDE firmware project.

## 📦 Currently Available Materials

### ✅ Analytical NILM Method Implementation

- **Location**: [`analytical_nilm/`](analytical_nilm/)
- **Description**: Python implementation of the shared feature extractor pretraining and analytical federated learning pipeline with synthetic toy demos.
- **Documentation**: See [`analytical_nilm/README.md`](analytical_nilm/README.md).

### ✅ STM32 Embedded Method Implementation

- **Location**: [`stm32_deployment/embedded_method/`](stm32_deployment/embedded_method/)
- **Description**: Method-level C implementation for STM32-side deployment, including PatchResNet forward inference, edge-side analytical update, and block-wise matrix upload helpers.
- **Documentation**: See [`stm32_deployment/embedded_method/README.md`](stm32_deployment/embedded_method/README.md).
- **Scope**: This is not a complete firmware project and does not include real model weights.

### ✅ STM32 Monitoring Platform

- **Location**: [`stm32_deployment/stm32_monitoring_platform/`](stm32_deployment/stm32_monitoring_platform/)
- **Description**: Web-based coordination interface for hardware federated learning experiments.
- **Documentation**: See [`stm32_deployment/stm32_monitoring_platform/readme.md`](stm32_deployment/stm32_monitoring_platform/readme.md).

### 🎬 Demo Video and Screenshots

- **Location**: [`stm32_deployment/assets/`](stm32_deployment/assets/)
- **Content**: Hardware deployment demonstration with a dual-device setup.

| Platform | Link |
| --- | --- |
| 🐙 GitHub | [stm32_deployment_demo.mp4](https://github.com/2771618309/An-Analytical-Federated-Learning-Method-for-Scalable-NILM/assets/stm32_deployment_demo.mp4) |
| 📺 Bilibili | [BV1mRAhzoEwt](https://www.bilibili.com/video/BV1mRAhzoEwt/) |
| ▶️ YouTube | [youtu.be/1ytGaKk8w70](https://youtu.be/1ytGaKk8w70) |

### ✅ Data Simulation Parameters

- **Location**: [`data_simulation/`](data_simulation/)
- **Description**: Circuit topologies, electrical parameters, and configurations for synthetic load generation.
- **Documentation**: See [`data_simulation/README.md`](data_simulation/README.md).

## 📋 Repository Structure

```text
An-Analytical-Federated-Learning-Method-for-Scalable-NILM/
|-- analytical_nilm/                # Core Python method implementation and toy demos
|   |-- pretrain/                   # Shared feature extractor and CE + SCL pretraining
|   |-- federated/                  # Analytical local update and cloud aggregation
|   |-- configs/                    # Toy pretraining and federated configs
|   |-- examples/                   # Synthetic-data demos
|   `-- outputs/                    # Runtime outputs ignored except .gitkeep
|-- stm32_deployment/
|   |-- embedded_method/            # Method-level STM32 C implementation
|   |-- stm32_monitoring_platform/  # Web-based FL coordination platform
|   `-- assets/                     # Demo videos and screenshots
|-- data_simulation/                # Simulation parameters and load topologies
|   |-- simulation_load_parameters.pdf
|   |-- simulation_load_parameters.xlsx
|   `-- assets/                     # Circuit topology diagrams
|-- LICENSE
`-- README.md
```

## 🚀 Quick Start

### 🧪 Python Method Demos

See [`analytical_nilm/README.md`](analytical_nilm/README.md) for the pretraining and analytical federated learning toy demos.

### 🔧 STM32 Embedded Method

See [`stm32_deployment/embedded_method/README.md`](stm32_deployment/embedded_method/README.md) for the method-level C implementation, public API, parameter template, and integration notes.

### 🖥️ Monitoring Platform

```bash
cd stm32_deployment/stm32_monitoring_platform
pip install -r requirements.txt
```

On Windows, launch the platform by double-clicking `open_platform .bat`.

## 🛠️ Remaining Release Plan

The repository already includes the core Python method implementation and the method-level STM32 C implementation. Future updates may focus on documentation and integration notes, such as:

- Additional usage notes for the released Python implementation.
- More detailed explanations of the pretraining and analytical federated learning demos.
- Additional notes for adapting the method-level C implementation to specific boards.
- More detailed examples for exporting parameters from PyTorch checkpoints.
- Optional guidance for integrating CMSIS-DSP and board-specific memory allocators.

The release will remain method-focused. Raw data, private data splits, baseline implementations, full experiment reproduction scripts, paper-table reproduction commands, real deployment weights, and complete product firmware are not included.

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{guo2026analytical,
  title={An Analytical Federated Learning Method for Scalable Non-Intrusive Load Monitoring},
  author={Wenlong Guo and Tao Yu and Qingquan Luo and Xiaolei Hu and Yufeng Wu and Zhenning Pan},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2026},
  pages={1--1},
  doi={10.1109/TIM.2026.3704092},
  note={Early Access, published June 16, 2026}
}
```

The paper is available as an IEEE Early Access article in IEEE Transactions on Instrumentation and Measurement.

## 🙏 Acknowledgments

This repository implements the method proposed in our NILM paper. Our work was informed by prior research on analytical federated learning, including the public AFL implementation by He et al. We thank the authors of [`ZHUANGHP/Analytic-federated-learning`](https://github.com/ZHUANGHP/Analytic-federated-learning) for releasing their code. In this repository, we re-derive a more compact analytical federated learning formulation for scalable NILM, substantially improving cloud-side aggregation efficiency and making the method better suited to large-scale deployment. The release also includes a patch-residual waveform feature extractor, NILM-specific pretraining, projection-branch analytical update, synthetic demos, and STM32-side method implementation.

## 📧 Contact

- **Repository**: [GitHub - Analytical Federated Learning for Scalable NILM](https://github.com/2771618309/An-Analytical-Federated-Learning-Method-for-Scalable-NILM)
- **Issues**: For technical questions about the available materials, please open an [issue](https://github.com/2771618309/An-Analytical-Federated-Learning-Method-for-Scalable-NILM/issues).
- **Email**: 2771618309@qq.com

## 📜 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

**Repository Status**: Method implementation and selected supplementary materials  
**Last Updated**: June 2026  
**Current Release Scope**: Core Python method implementation and method-level STM32 C implementation

⭐ **Star this repository to receive notifications when additional materials are released.**
