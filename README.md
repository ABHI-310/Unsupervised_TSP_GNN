# Unsupervised Traveling Salesman Problem Solver via Graph Neural Networks (UTSP-GNN)

An unsupervised learning framework that utilizes a Deep Message-Passing Graph Neural Network (GNN) to predict combinatorial edge-selection heatmaps for NP-hard Traveling Salesman Problems (TSP) without relying on ground-truth optimal tours.

The architecture features a fully vectorized matrix-based triplet surrogate loss routine that scales efficiently to large graph dimensions, yielding strong zero-shot generalization performance on unseen node topologies.

---

# Key Features

## Unsupervised Triplet Ranking

Optimizes edge affinity scores using structural geometric constraints without requiring labeled optimal tours.

## Vectorized Matrix Acceleration

Eliminates nested Python loops by broadcasting a custom rank-based surrogate hinge loss inside high-dimensional PyTorch tensors.

## Hybrid Inference Decoder

Combines continuous GNN edge probability distributions with a greedy decoding module and a 2-opt local search heuristic.

## Size-Invariant Generalization

Demonstrates robust zero-shot scaling, yielding a **78.81% path-efficiency improvement** on unseen `(N=100)` graph topologies while being trained exclusively on `(N=20)` graphs.

---

# Project Structure

```text
MY_UTSP/
│
├── .vscode/
│   └── settings.json
│
├── src/
│   ├── config.py          # Global hyperparameter configuration
│   ├── eval.py            # Zero-shot validation and benchmarking
│   ├── fake_utsp.py       # Greedy path reconstruction decoder
│   ├── loss.py            # Vectorized surrogate triplet loss
│   ├── model.py           # Message-passing GNN architecture
│   ├── plot_log.py        # Optimization trajectory visualization
│   ├── train.py           # Training pipeline with early stopping
│   └── tsp_core.py        # 2-opt refinement and geometry utilities
│
├── .gitignore
└── README.md
```

---

# Mathematical Core: Unsupervised Surrogate Loss

The framework optimizes **relative structural rankings** rather than predicting explicit coordinate connections or supervised optimal tours.

Given an anchor node (i), the objective enforces that a structurally closer neighbor (j) receives a higher affinity score than a more distant node (k):

$$\mathcal{L} = \max(0,\ 1.0 - (H_{i,j} - H_{i,k}))$$

Where:

* (H_{i,j}) represents the predicted edge affinity between nodes (i) and (j)
* (H_{i,k}) represents the predicted edge affinity between nodes (i) and (k)

The loss is computed through fully vectorized tensor broadcasting operations using unsqueezed coordinate dimensions, transforming combinatorial optimization into massively parallel matrix computation.

This enables efficient scaling across larger graph topologies while maintaining differentiable structural learning behavior.

---

# Quick Start

## 1. Install Dependencies

Ensure that Python and the required libraries are installed:

```bash
pip install torch numpy scipy matplotlib networkx
```

---

## 2. Train the GNN Framework

Run the primary optimization pipeline:

```bash
python3 src/train.py
```

This executes:

* Random Euclidean graph generation
* Message-passing propagation
* Edge affinity prediction
* Vectorized surrogate optimization
* Greedy decoding
* 2-opt refinement

---

## 3. Evaluate Zero-Shot Generalization

Benchmark the framework against classical random baselines:

```bash
python3 src/eval.py
```

The evaluation pipeline measures:

* Tour efficiency
* Generalization quality
* Structural routing consistency
* Scaling robustness on unseen graph sizes

---

## 4. Visualize Optimization Curves

Plot optimization trajectories and convergence behavior:

```bash
python3 src/plot_log.py
```

---

# Training Pipeline Overview

## Graph Generation

Random Euclidean graph instances are dynamically sampled during training.

## Message Passing

Node embeddings propagate through deep graph neural message-passing layers.

## Edge Heatmap Prediction

Pairwise embedding interactions generate continuous edge-affinity matrices.

## Surrogate Optimization

The vectorized triplet-ranking objective optimizes geometric structural consistency.

## Tour Reconstruction

Greedy decoding converts affinity heatmaps into valid routing trajectories.

## Local Search Refinement

A combinatorial 2-opt heuristic further improves generated tours.

---

# Core Technologies

* PyTorch
* NumPy
* SciPy
* NetworkX
* Matplotlib

---

# Research Motivation

Traditional supervised Traveling Salesman Problem solvers depend heavily on computationally expensive optimal-label generation using exact combinatorial solvers.

This project investigates whether geometric structural priors alone are sufficient for neural networks to learn meaningful routing behaviors without explicit optimal supervision.

The framework therefore emphasizes:

* Relative geometric consistency
* Structural ranking behavior
* Emergent combinatorial optimization
* Zero-shot graph generalization
* Scalable unsupervised routing intelligence

---

# Experimental Highlights

* Fully unsupervised training regime
* No optimal tour labels required
* Strong zero-shot scaling behavior
* Efficient vectorized tensor operations
* Hybrid neural-combinatorial inference pipeline
* Generalizes beyond training graph sizes

---

# Future Improvements

## Architectural Extensions

* Attention-based graph transformers
* Sparse message-passing acceleration
* Dynamic graph batching

## Optimization Improvements

* Reinforcement-learning hybrid decoding
* Curriculum graph scaling
* Beam-search inference strategies

## Systems Optimization

* GPU-optimized sparse tensor kernels
* Distributed training pipelines
* Mixed-precision acceleration

---

# License

This project is intended for research, experimentation, and educational purposes.
