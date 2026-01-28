import torch
from tsp_core import generate_cities, compute_distance_matrix, tour_length, two_opt
from fake_utsp import decode_greedy

from model import GNNScorer, build_heatmap_from_gnn
import config

def main():
    model = GNNScorer(config.HIDDEN_DIM)
    model.load_state_dict(torch.load("checkpoints/best_utsp_model.pt"))
    model.eval()

    cities = generate_cities(config.N)
    dist = compute_distance_matrix(cities)

    heatmap = build_heatmap_from_gnn(model, cities)
    heatmap = torch.tanh((heatmap - heatmap.mean()) / config.TEMPERATURE)

    tour = decode_greedy(heatmap.detach().numpy(), start=0)
    tour = two_opt(tour, dist)
    learned_len = tour_length(tour, dist)

    random_heatmap = torch.rand((config.N, config.N))
    random_heatmap.fill_diagonal_(0.0)
    base = decode_greedy(random_heatmap.numpy(), start=0)
    base = two_opt(base, dist)
    base_len = tour_length(base, dist)

    print("Eval tour length:", learned_len)
    print("Baseline tour length:", base_len)
    print("Improvement:", base_len - learned_len)

if __name__ == "__main__":
    main()
