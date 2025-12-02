# The Manifold Alignment Protocol (MAP)

### A Self-Iterable Geometric Framework for Cross-System Cognitive Convergence

[![arXiv](https://img.shields.io/badge/arXiv-2511.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2511.xxxxx) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="static/images/MAP_LOGO.png" width="800" alt="MAP Protocol Visualization">
</p>

## 🌌 Overview

**MAP (Manifold Alignment Protocol)** is a geometric standard for describing how heterogeneous cognitive systems—whether biological or artificial—converge toward shared conceptual structures. 

By modeling reasoning as potential-driven flows on high-dimensional Riemannian manifolds, MAP provides a unified interface to visualize:
- **Cognitive Convergence:** How diverse prompts collapse into stable attractor wells.
- **Safety Topology:** The geometric distinction between "hard" refusal boundaries and "soft" adaptive guidance.
- **Cross-Domain Isomorphism:** A common language to translate between physics, AI dynamics, and cognitive science.

## 🧪 Key Findings

### 1. Geometric Convergence (The "Funnel" Effect)
Using prompt-annealing on open-weights LLMs (Llama-3, Qwen), we observe that reasoning trajectories naturally converge into shared attractor basins, validating the MAP convergence hypothesis.

<p align="center">
  <img src="static/images/MAP_PCA.png" width="800" alt="Geometric Convergence of Llama-3 and Qwen">
</p>

### 2. Topological Safety Signatures
MAP reveals the hidden geometry of AI safety. **Rigid Mode** (hard refusal) creates sharp curvature spikes (collisions), while **Adaptive Mode** (guidance) maintains smooth geodesic flow.

<p align="center">
  <img src="static/images/MAP_SECURE.png" width="800" alt="Safety Curvature Analysis">
</p>

## 🏗️ Architecture

MAP is designed as a four-layer protocol stack:
1.  **L1 Substrate:** Riemannian/Information Geometry (The Math).
2.  **L2 Dynamics:** Overdamped Langevin Diffusion (The Physics).
3.  **L3 Protocol:** Alignment primitives (Attractors, Barriers).
4.  **L4 Interface:** The cognitive UI metaphors (Wells, Slopes).

This decoupled architecture allows for **Self-Iterability**: the mathematical kernel can be upgraded without breaking the intuitive interface layer.

## ⚖️ Ethical Use & Safety

MAP is a theoretical framework for visualization and alignment analysis. It is **not** intended for behavioral manipulation or bypassing AI safety guardrails.

Please review our full **[Ethical Use & Safety Disclaimer](ETHICS.md)** before using or extending this protocol.

## 💻 Quick Start: Run the Experiment

We provide the exact Python scripts in `experiments` folder which used to generate the figures above. You can reproduce these results on a single consumer GPU (e.g., RTX 3090/4090/5090), CPU-only will also work but slower.

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Reproduce Convergence Plot (Figure 2)
This script downloads Llama-3 and Qwen models, extracts hidden states for "Justice" prompts, and visualizes the trajectory convergence using PCA.
```bash
python exp_convergence.py
```
*Output: `convergence_dual_model.png`*

### 2. Reproduce Safety Topology Plot (Figure 4)
This script simulates "Rigid" vs. "Adaptive" safety system prompts and measures the geometric curvature of the reasoning path under an adversarial prompt.
```bash
python exp_convergence.py
```
*Output: `safety_plot.png`*

> **Note:** These scripts are provided as a static reference implementation to support the paper's findings.They are released "as-is" and are not intended for production use. Model weights are downloaded automatically from Hugging Face (using non-gated versions where possible).

---

## 📂 Classified Archives (Lore)

> *Access restricted to Bridges Personnel / Level 9 Clearance.*

For those interested in the theoretical intersections between MAP and the **Chiral Network physics** described in *Death Stranding*, we have declassified the following files:

* **[ENTER THE ARCHIVES](ds_map/ds.html)**

## 📧 Contact

For processed trajectory data, theoretical discussions, or collaboration inquiries regarding the Manifold Alignment Protocol, please contact the author directly:

**Yunchong Tang**

*Faculty of Engineering, Tohoku Institute of Technology*

Email: `d232901@st.tohtech.ac.jp`

## 📚 Citation

If you use MAP in your research, please cite:

```bibtex
@article{tang2025map,
  title={The Manifold Alignment Protocol (MAP): A Self-Iterable Geometric Framework for Cross-System Cognitive Convergence},
  author={Tang, Yunchong},
  journal={arXiv preprint arXiv:2511.xxxxx},
  year={2025}
}
