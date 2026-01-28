# Learning to Solve the Traveling Salesman Problem with a Graph Neural Network

This project implements a learning-based approach to the Traveling Salesman Problem (TSP) using a message-passing Graph Neural Network (GNN) trained with a surrogate ranking loss and refined with classical local search (2-opt).

The goal is not to replace exact solvers, but to learn heuristics that produce high-quality tours efficiently and generalize across problem instances.

---

## Problem Overview

The Traveling Salesman Problem (TSP) is a classic NP-hard combinatorial optimization problem:

> Given a set of cities and pairwise distances, find the shortest possible tour that visits each city exactly once and returns to the starting city.

Exact solvers scale poorly with problem size. Practical solutions often rely on heuristics such as 2-opt, 3-opt, or Lin–Kernighan. Recent research explores learning-based methods to guide or replace these heuristics.

---

## Method

### Model Architecture

- Cities are represented as 2D coordinates.
- A **message-passing Graph Neural Network** computes node embeddings.
- Pairwise edge scores (a “heatmap”) are produced from node embeddings and distances.
- A greedy decoder constructs a tour from the heatmap.
- Classical **2-opt local search** refines the tour.

### Learning Objective

The model is trained using a **surrogate ranking loss**:

- For each city \(i\), closer neighbors should receive higher edge scores than farther ones.
- This avoids differentiating through the discrete decoding step.
- The loss encourages the heatmap to align with short edges without requiring ground-truth tours.

### Training Details

- Training graphs: 20 nodes
- Validation set with early stopping
- Adam optimizer
- Stabilization via mean-centering, temperature scaling, and `tanh`

---

## Project Structure
