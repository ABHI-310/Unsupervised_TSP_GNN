import torch
import numpy as np
from tsp_core import generate_cities, compute_distance_matrix, tour_length, two_opt
from fake_utsp import decode_greedy
from model import GNNScorer, build_heatmap_from_gnn
import config

# def main():
#     model = GNNScorer(config.HIDDEN_DIM)
#     try:
#         model.load_state_dict(torch.load("best_utsp_model.pt"))
#     except FileNotFoundError:
#         return

#     model.eval()
#     cities = generate_cities(config.EVAL_N)
#     dist = compute_distance_matrix(cities)

#     with torch.no_grad():
#         heatmap = build_heatmap_from_gnn(model, cities)
#         heatmap = torch.tanh((heatmap - heatmap.mean()) / config.TEMPERATURE)

#     tour = decode_greedy(heatmap.numpy(), start=0)
#     tour = two_opt(tour, dist)
#     learned_len = tour_length(tour, dist)

#     random_heatmap = torch.rand((config.EVAL_N, config.EVAL_N))
#     random_heatmap.fill_diagonal_(0.0)
    
#     base = decode_greedy(random_heatmap.numpy(), start=0)
#     base = two_opt(base, dist)
#     base_len = tour_length(base, dist)

#     print(f"GNN Learned Tour Length: {learned_len:.4f}")
#     print(f"Classical Baseline Length: {base_len:.4f}")
#     print(f"Absolute Path Reduction: {base_len - learned_len:.4f}")
#     print(f"Efficiency Gain: {((base_len - learned_len) / base_len) * 100:.2f}%")

# if __name__ == "__main__":
#     main()

def main():
    model = GNNScorer(config.HIDDEN_DIM)
    try:
        model.load_state_dict(torch.load("best_utsp_model.pt"))
    except FileNotFoundError:
        print("Model file not found.")
        return

    model.eval()
    cities = generate_cities(config.EVAL_N)
    dist = compute_distance_matrix(cities)

    with torch.no_grad():
        heatmap = build_heatmap_from_gnn(model, cities)
        # REMOVE Tanh squashing to preserve the model's raw confidence signals
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 1. GNN + 2-Opt Tour
    tour = decode_greedy(heatmap.numpy(), start=0)
    tour_refined = two_opt(tour, dist)
    learned_len = tour_length(tour_refined, dist)

    # 2. Pure Random Greedy Baseline (Fair comparison)
    random_heatmap = torch.rand((config.EVAL_N, config.EVAL_N))
    random_heatmap.fill_diagonal_(0.0)
    
    base = decode_greedy(random_heatmap.numpy(), start=0)
    base_len = tour_length(base, dist)

    print(f"GNN + 2-Opt Tour Length: {learned_len:.4f}")
    print(f"Classical Greedy Baseline Length: {base_len:.4f}")
    print(f"Absolute Path Reduction: {base_len - learned_len:.4f}")
    print(f"Efficiency Gain: {((base_len - learned_len) / base_len) * 100:.2f}%")

if __name__ == "__main__":
    main()
