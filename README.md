# The Manifold Alignment Protocol (MAP)

### A Geometric Framework for Analyzing Convergence and Alignment in Complex Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17900444.svg)](https://doi.org/10.5281/zenodo.17900444)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MAP LLM Toolkit](https://img.shields.io/badge/MAP%20LLM%20Toolkit-v0.1-blue)](https://github.com/JBKing514/map_llm_toolkit)
[![MAP ComfyUI](https://img.shields.io/badge/MAP%20ComfyUI-v0.2-blue)](https://github.com/JBKing514/map_comfyui)

---

## 🌌 Overview

**The Manifold Alignment Protocol (MAP)** is a geometric analysis framework for studying how complex systems—such as large language models, diffusion models, or physical measurement pipelines—**converge, stabilize, and align** under iterative processes.

Rather than focusing on model internals or task-specific metrics, MAP treats system behavior as **trajectories on a latent manifold**, enabling:

- Visualization of convergence and divergence
- Quantification of stability and alignment
- Comparison across heterogeneous systems
- Geometry-based parameter optimization

MAP is designed as a **protocol**, not a fixed model: its mathematical substrate can be replaced or approximated depending on performance and application constraints.

---

## 🔑 Core Ideas

- **Trajectory-Based Analysis**  
  System behavior is modeled as a sequence of latent states evolving over time.

- **Geometric Convergence**  
  Stable systems exhibit contraction toward attractor regions; unstable systems display oscillation or fragmentation.

- **Protocol Modularity**  
  MAP separates geometry (L1/L2) from interpretation and metrics (L3/L4), enabling fast approximations without breaking conceptual consistency.

- **Cross-Domain Applicability**  
  The same protocol applies to LLM reasoning, diffusion sampling, signal processing pipelines, and robotic sensing workflows.

---

## 🏗️ MAP Protocol Stack

MAP is structured as a four-layer architecture:

1. **L1 — Substrate**  
   Latent space geometry (Riemannian, information-geometric, or Euclidean approximation)

2. **L2 — Dynamics**  
   Iterative update rules (e.g., diffusion, optimization, sampling)

3. **L3 — Protocol Primitives**  
   Attractors, barriers, alignment scores, convergence metrics

4. **L4 — Interface**  
   Human-interpretable representations (plots, scores, diagnostics)

This separation enables **self-iterability**: the underlying math can evolve without changing how users interpret results.

---

## 🧪 Reference Experiments

We provide the Python scripts in `experiments` folder to validate the MAP's concept. You can reproduce these results on a single consumer GPU (e.g., RTX 3090/4090/5090), CPU-only will also work but slower.

**Note:**
This project installs the CPU version of PyTorch by default.
If you want to run the experiments on GPU, please manually install a CUDA-enabled build of PyTorch that matches your CUDA version (e.g., cu118 or cu121).
After installing the appropriate GPU build, all scripts will automatically use the GPU if available.

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Latent Trajectory Convergence (LLMs)

- Extracts hidden states from open-weight LLMs
- Projects trajectories using PCA
- Visualizes contraction into attractor regions

```bash
python exp_convergence.py
```

### 2. Semantic Alignment Across Layers

- Computes layer-wise alignment metrics
- Compares tight vs. sparse semantic clusters
- Reveals architecture-dependent folding depth

```bash
python exp_alignment.py
```

### 3. Safety Topology Diagnostics

- Compares rigid vs. adaptive safety behaviors
- Analyzes curvature and termination geometry

```bash
python exp_safety.py
```

> These scripts are provided as **reproducible references**, not production benchmarks.

---

## 🔧 MAP Ecosystem

MAP is accompanied by two official toolkits demonstrating its practical use.

### 🔹 MAP-LLM-Toolkit

A Python API for applying MAP to language models:

- Hidden-state extraction
- Trajectory projection
- Alignment metrics
- Safety topology analysis

👉 https://github.com/JBKing514/map_llm_toolkit

---

### 🔹 MAP-ComfyUI (Diffusion Models)

A geometry-based analysis and auto-tuning system for Stable Diffusion workflows.

Features include:
- Real-time latent trajectory visualization
- Convergence-based quality scoring (Q-score)
- Automatic optimization of Steps / CFG / Scheduler
- Early stopping to reduce wasted compute

⚠️ **Performance Note**  
For interactive use, this implementation employs Euclidean approximations and random projections instead of full Riemannian geometry. This trade-off prioritizes speed while preserving protocol semantics.

👉 https://github.com/JBKing514/map_comfyui

---

## 📧 Contact

**Yunchong Tang**  
Faculty of Engineering, Tohoku Institute of Technology  

Email: `d232901@st.tohtech.ac.jp`

---

## 📖 Citation

If you use MAP or its associated tools, please cite:

```bibtex
@article{tang2025map,
  title   = {The Manifold Alignment Protocol (MAP): A Self-Iterable Geometric Framework for Cross-System Cognitive Convergence},
  author  = {Tang, Yunchong},
  journal = {Zenodo},
  year    = {2025},
  doi     = {10.5281/zenodo.17900444},
  url     = {https://doi.org/10.5281/zenodo.17900444}
}
```

---

## 🤝 Contributions

Contributions are welcome in the form of:
- New case studies
- Visualization modules
- Metric refinements
- Cross-domain applications

---

*MAP is an evolving research protocol.  
Its goal is not to replace domain expertise, but to provide a shared geometric language for reasoning about convergence and alignment.*
