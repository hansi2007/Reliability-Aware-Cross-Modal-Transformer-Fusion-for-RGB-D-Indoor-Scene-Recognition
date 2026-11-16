# RACT-Fuse: Reliability-Aware Cross-Modal Transformer Fusion for RGB-D Indoor Scene Recognition

<p align="center">

**An Intelligent Multimodal Deep Learning Framework for Robust Indoor Scene Understanding**

</p>

## 📌 Overview

RGB-D scene recognition plays an important role in autonomous robotics, intelligent navigation, and indoor environment understanding. By combining RGB images with depth information, deep learning models can learn both visual appearance and spatial geometry.

However, real-world depth sensors frequently produce incomplete, noisy, or distorted measurements due to reflective surfaces, transparent objects, lighting conditions, and sensor limitations. Conventional multimodal fusion approaches often treat depth information as equally reliable across all regions, which can negatively affect recognition performance when depth measurements become corrupted.

**RACT-Fuse (Reliability-Aware Cross-Modal Transformer Fusion)** addresses this challenge through a reliability-aware multimodal learning framework.

The proposed architecture uses independent RGB and depth feature encoders, bidirectional cross-modal attention, and a Dynamic Reliability Gate to estimate the quality of incoming depth information before fusion.

Instead of blindly combining both modalities, RACT-Fuse dynamically adjusts the contribution of depth features according to their estimated reliability, allowing the model to prioritize RGB information when depth measurements become unreliable.

The framework aims to improve recognition accuracy, prediction confidence, and robustness under challenging indoor sensing conditions.

## 🎯 Key Objectives

- Develop a reliability-aware RGB-D fusion architecture for indoor scene recognition.
- Capture complementary visual and geometric information using independent feature encoders.
- Enable effective information exchange through bidirectional cross-modal attention.
- Estimate depth reliability using sensor validity, edge agreement, and predictive entropy.
- Reduce the influence of corrupted depth measurements during feature fusion.
- Improve prediction confidence through uncertainty-aware training and calibration.
- Maintain computational efficiency for potential autonomous robotics applications.

## 🧠 Proposed Architecture

RACT-Fuse consists of three major components:

### 1. Independent Multimodal Feature Extraction

The framework processes RGB and depth information through separate hierarchical Swin Transformer encoders.

**RGB Branch**
- Extracts appearance information from RGB images.
- Captures color, texture, edges, and semantic representations.
- Converts image patches into hierarchical feature tokens.

**Depth Branch**
- Processes a five-channel geometric representation.
- Captures spatial structure and depth discontinuities.
- Uses a validity mask to identify missing or invalid depth measurements.

The geometric representation consists of:

- Normalized Depth (D)
- Inverse Depth (D⁻¹)
- Horizontal Depth Gradient (∂xD)
- Vertical Depth Gradient (∂yD)
- Binary Validity Mask (M)

### 2. Bidirectional Cross-Modal Attention

To enable interaction between appearance and geometric information, RACT-Fuse introduces a bidirectional cross-attention bridge.

The mechanism enables:

- RGB-to-Depth attention for transferring semantic appearance information.
- Depth-to-RGB attention for incorporating geometric context.
- Contextual feature refinement between both modalities.
- Improved representation learning through cross-modal information exchange.

This allows the network to learn complementary relationships between visual appearance and spatial geometry.

### 3. Dynamic Reliability Gate

The Dynamic Reliability Gate is the central component of RACT-Fuse.

It estimates the reliability of depth information using three quality indicators:

| Indicator | Purpose |
|---|---|
| Valid Pixel Ratio | Measures the proportion of valid depth pixels. |
| Edge Agreement | Measures structural alignment between RGB and depth gradients. |
| Predictive Entropy | Estimates uncertainty in depth-based predictions. |

These indicators are combined into a quality vector and passed through a lightweight neural gating network with a Sigmoid activation.

The resulting reliability score (ρ) lies between 0 and 1.

The final fused representation is calculated as:

**Fused Feature = RGB Feature + ρ × Depth Feature**

When depth information is reliable, its contribution increases. When depth measurements become corrupted or uncertain, the gate reduces their influence, allowing the model to depend more heavily on RGB features.

## ⚙️ Technical Methodology

The proposed framework uses:

- **Backbone:** Swin Transformer-Tiny (Swin-T)
- **Feature Extraction:** Independent hierarchical RGB and depth encoders
- **Fusion Strategy:** Bidirectional Cross-Modal Attention
- **Reliability Estimation:** Dynamic neural gating
- **Uncertainty Estimation:** Predictive entropy
- **Classification:** Calibrated scene classifier
- **Optimization:** AdamW
- **Loss Function:** Joint Cross-Entropy and Expected Calibration Error objective

The total training objective combines:

1. Cross-Entropy Loss for fused predictions.
2. Auxiliary Cross-Entropy Loss for depth predictions.
3. Expected Calibration Error penalty.

## 📊 Experimental Evaluation

The proposed RACT-Fuse framework is evaluated using two established indoor scene recognition datasets:

### Datasets

| Dataset | Images | Classes |
|---|---:|---:|
| SUN RGB-D | 10,335 | 19 |
| NYU Depth V2 | 1,449 | 10 |

### Performance on SUN RGB-D

| Model | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|
| RGB Backbone (Swin-T) | 56.2% | 54.8% | 0.082 |
| Late Fusion | 60.8% | 59.1% | 0.075 |
| TransFuser | 63.5% | 62.2% | 0.068 |
| CMX | 65.4% | 64.1% | 0.062 |
| DFormerv2 | 66.8% | 65.5% | 0.051 |
| **RACT-Fuse** | **67.5%** | **66.2%** | **0.031** |

### Performance on NYU Depth V2

The full RACT-Fuse configuration achieves:

- **Accuracy:** 70.8%
- **Macro-F1:** 69.5%

### Robustness Against Depth Corruption

The framework is additionally evaluated under different levels of missing depth information.

| Model | 0% Noise | 20% Holes | 50% Holes | 80% Holes |
|---|---:|---:|---:|---:|
| Late Fusion | 60.8% | 56.4% | 48.2% | 39.1% |
| CMX | 65.4% | 61.2% | 53.5% | 45.8% |
| DFormerv2 | 66.8% | 63.1% | 56.7% | 48.2% |
| **RACT-Fuse** | **67.5%** | **65.8%** | **63.1%** | **58.4%** |

At 80% depth corruption, RACT-Fuse demonstrates a smaller accuracy reduction compared with the other evaluated fusion approaches.

This highlights the potential of reliability-aware fusion for handling incomplete geometric information.

## 🔬 Ablation Study

An ablation study on NYU Depth V2 investigates the contribution of individual architectural components.

| Configuration | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline: No Attention/Gate | 64.2% | 62.8% |
| RGB-to-Depth Attention | 66.5% | 65.1% |
| Bidirectional Attention | 68.1% | 67.0% |
| Bidirectional + Entropy Gate | 69.4% | 68.2% |
| **Full Model with ECE Calibration** | **70.8%** | **69.5%** |

The results indicate that bidirectional attention, reliability estimation, and calibration contribute to the overall performance of the framework.

## 🚀 Computational Efficiency

The computational profile of RACT-Fuse is evaluated using an NVIDIA RTX 4090 GPU.

| Metric | RACT-Fuse |
|---|---:|
| Parameters | 44.8 Million |
| MACs | 7.1 G |
| Inference Speed | 45 FPS |

The reported computational profile demonstrates a balance between recognition performance and inference efficiency.

## 🛠️ Technologies and Tools

The framework is based on the following technologies:

- Python
- PyTorch 2.0
- Swin Transformer
- Deep Learning
- Computer Vision
- Multimodal Learning
- Cross-Attention Mechanisms
- Uncertainty Estimation
- RGB-D Sensor Processing
- AdamW Optimization

## 🌍 Potential Applications

RACT-Fuse explores reliability-aware multimodal perception for applications such as:

- Autonomous indoor navigation
- Service and domestic robotics
- Intelligent surveillance systems
- Indoor environment understanding
- Human-robot interaction
- Robotics-based scene classification
- Smart infrastructure and spatial perception

## 📁 Repository Structure

The following structure can be used to organize the implementation:

```text
RACT-Fuse/
│
├── README.md
├── requirements.txt
│
├── configs/
│   └── config.yaml
│
├── datasets/
│   ├── SUNRGBD/
│   └── NYUDepthV2/
│
├── models/
│   ├── rgb_encoder.py
│   ├── depth_encoder.py
│   ├── cross_attention.py
│   ├── reliability_gate.py
│   └── ract_fuse.py
│
├── training/
│   ├── train.py
│   ├── loss.py
│   └── optimizer.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── robustness_test.py
│
├── results/
│
└── paper/
    └── RACT-Fuse.pdf
```

*Note: This is a suggested repository organization, not a claim that these implementation files are already available.*

## 📖 Research Contribution

The central contribution of RACT-Fuse is its reliability-aware approach to RGB-D feature fusion.

Unlike conventional approaches that continuously combine both modalities without explicitly considering depth quality, the proposed framework estimates depth reliability and dynamically regulates its influence during fusion.

The combination of hierarchical feature extraction, bidirectional cross-modal attention, entropy-aware gating, and confidence calibration provides a structured approach to robust indoor scene recognition.

## 📄 Research Publication

**Paper Title:** Reliability-Aware Cross-Modal Transformer Fusion for RGB-D Indoor Scene Recognition

**Conference:** NKCon 2026 – Holistic Engineering for Sustainable Engineering

**Institution:** Dayananda Sagar University, Bengaluru, India

For detailed mathematical formulations, experimental configurations, architecture descriptions, and evaluation results, refer to the complete research paper available in this repository.



## 🤝 Contributions

Contributions, suggestions, and discussions related to reliability-aware multimodal learning, RGB-D perception, and transformer-based architectures are welcome.

## 📚 References

1. Song et al., "SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite," CVPR, 2015.
2. Silberman et al., "Indoor Segmentation and Support Inference from RGBD Images," ECCV, 2012.
3. Zhang et al., "CMX: Cross-Modal Fusion for RGB-X Perception," IEEE TPAMI, 2023.
4. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV, 2021.
5. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR, 2021.
6. Guo et al., "On Calibration of Modern Neural Networks," ICML, 2017.

---

**RACT-Fuse — Making Multimodal Scene Recognition More Reliable Through Intelligent Feature Fusion.**
