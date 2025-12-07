# The Manifold Alignment Protocol (MAP)

### A Self-Iterable Geometric Framework for Cross-System Cognitive Convergence

[![arXiv](https://img.shields.io/badge/arXiv-2511.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2511.xxxxx) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Toolkit](https://img.shields.io/badge/MAP%20LLM%20Toolkit-v0.1-blue)](https://github.com/JBKing514/map_llm_toolkit) [![Toolkit](https://img.shields.io/badge/MAP%20ComfyUI-v0.1-blue)](https://github.com/JBKing514/map_comfyui)


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
MAP reveals the hidden geometry of AI safety: **Adaptive Mode**(guidance) shows a stable U-turn that redirects the reasoning path away from danger, while **Rigid Mode**(hard refusal) exhibits an inelastic crash into a boundary and ends immediately.

<p align="center">
  <img src="static/images/MAP_SECURE.png" width="800" alt="Safety Curvature Analysis">
</p>

### 3. Architecture Analysis: The "Thinking Style"(Coming in v2)
MAP metric $\Delta A$ reveals distinct cognitive styles. Llama-3 discriminates semantics early (Layer 5), while Qwen-2.5 maintains a high-abstraction 'superposition' state until deep layers (Layer 20), correlating with its reasoning-heavy architecture.

<p align="center">
  <img src="static/images/MAP_ALINE.png" width="800" alt="Safety Curvature Analysis">
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

**Note:**
This project installs the CPU version of PyTorch by default.
If you want to run the experiments on GPU, please manually install a CUDA-enabled build of PyTorch that matches your CUDA version (e.g., cu118 or cu121).
After installing the appropriate GPU build, all scripts will automatically use the GPU if available.

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Reproduce Convergence Plot (Figure 3)
This script downloads Llama-3 and Qwen models, extracts hidden states for "Justice" prompts, and visualizes the trajectory convergence using PCA.
```bash
python exp_convergence.py
```
*Output: `convergence_dual_model.png`*

### 2. Reproduce Safety Topology Plot (Figure 5)
This script simulates "Rigid" vs. "Adaptive" safety system prompts and measures the geometric curvature of the reasoning path under an adversarial prompt.
```bash
python exp_convergence.py
```
*Output: `safety_plot.png`*

### 3. Architecture Analysis: Semantic Folding (Coming in v2)
This script quantifies the **Layer-wise Alignment Score ($A$)** to visualize how different architectures compress meaning.
* It compares "Tight" (paraphrased) vs. "Sparse" (random) semantic clusters.
* **Key Insight:** It reveals that Qwen-2.5 exhibits **"Delayed Folding"** (maintaining high-dimensional superposition until deep layers), whereas Llama-3 collapses semantics early.
```bash
python exp_alignment.py
```
*Output: `alignment_delta_profile.png`*

> **Note:** These scripts are provided as a static reference implementation to support the paper's findings.They are released "as-is" and are not intended for production use. Model weights are downloaded automatically from Hugging Face (using non-gated versions where possible).

## 🔧 MAP Ecosystem: Official Implementations

To demonstrate the universality of the Manifold Alignment Protocol across different cognitive substrates (LLMs vs. Diffusion), we provide two official toolkits:

### 1. For Language Models: MAP-LLM-Toolkit
The experimental scripts used in this paper have been packaged into a clean and modular Python API.

This toolkit provides:
- Hidden-state extraction utilities  
- PCA projection and trajectory visualization  
- Semantic Alignment through layer-wise alignment metrics
- Safety topology analysis (rigid vs. adaptive modes)  
- Reproducible examples for extending MAP-based experiments  

👉 **Repository: [map_llm_toolkit](https://github.com/JBKing514/map_llm_toolkit)**

### 2. For Diffusion Models: MAP-ComfyUI

To validate MAP's applicability to continuous latent dynamics, we developed a geometric "Vector Network Analyzer" for Stable Diffusion workflows.

**Why use this?** It turns "Prompt Engineering" into "Prompt Engineering" (literally).
Instead of guessing CFG/Steps, let the geometry decide.

⚠️ **Note:** This custom node is an engineering adaptation inspired by the MAP framework. While the core philosophy (trajectory convergence & stability) remains central, the mathematical metrics (e.g., Q-Score) are heuristic approximations designed for real-time creative workflows, rather than strict differential geometry implementations found in the main paper.

This custom node suite translates MAP principles into a practical engineering tool:
* **Latent Trajectory Plotting:** Visualizes the "geometry of generation" in real-time. See if your prompt is "struggling" or "converging".
* **Convergence Metrics (Q-Score):** A mathematically derived quality score based on attractor stability.
* **Auto-Tuner (GPU Saver):** An automated optimization engine (Hill Climbing) that finds the **optimal** Step/CFG settings. **Stop wasting compute on over-baking images.**

👉 **Repository: [map_comfyui](https://github.com/JBKing514/map_comfyui)**

Contributions of any form (features, experiments, visualization modules, discussions) are highly appreciated.

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
