import random
import math
import torch
import torch.nn as nn
import torch.optim as optim

from tsp_core import generate_cities, compute_distance_matrix, tour_length, two_opt
from src.fake_utsp import decode_greedy

class GNNScorer(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, cities):
        return self.node_embed(cities.float())

def build_heatmap_from_gnn(model, cities):
    coords = torch.tensor(cities, dtype=torch.float32)
    h = model(coords)
    n, _ = h.shape
    hi = h.unsqueeze(1).repeat(1, n, 1)
    hj = h.unsqueeze(0).repeat(n, 1, 1)
    ci = coords.unsqueeze(1).repeat(1, n, 1)
    cj = coords.unsqueeze(0).repeat(n, 1, 1)
    dij = torch.norm(ci - cj, dim=-1, keepdim=True)
    msg_input = torch.cat([hi, hj, dij], dim=-1)
    messages = model.msg_mlp(msg_input)
    agg = messages.mean(dim=1)
    h_updated = model.update_mlp(torch.cat([h, agg], dim=-1))
    hi = h_updated.unsqueeze(1).repeat(1, n, 1)
    hj = h_updated.unsqueeze(0).repeat(n, 1, 1)
    edge_input = torch.cat([hi, hj, dij], dim=-1)
    heatmap = model.edge_mlp(edge_input).squeeze(-1)
    heatmap.fill_diagonal_(0.0)
    return heatmap

def surrogate_loss(heatmap, dist):
    n = heatmap.shape[0]
    loss = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                if dist[i][j] < dist[i][k]:
                    loss += torch.relu(1.0 - (heatmap[i, j] - heatmap[i, k]))
                    count += 1
    return loss / max(count, 1)

def eval_on_batch(model, cities_list, dists_list):
    total = 0.0
    for cities, dist in zip(cities_list, dists_list):
        heatmap = build_heatmap_from_gnn(model, cities)
        heatmap = torch.tanh((heatmap - heatmap.mean()) / 2.0)
        tour = decode_greedy(heatmap.detach().numpy(), start=0)
        tour = two_opt(tour, dist)
        total += tour_length(tour, dist)
    return total / len(cities_list)

batch_size = 2
EVAL_ONLY = False

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    n = 20
    model = GNNScorer(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    val_cities = [generate_cities(n) for _ in range(8)]
    val_dists = [compute_distance_matrix(c) for c in val_cities]

    best_val_tour = float("inf")
    patience = 15
    patience_counter = 0

    if not EVAL_ONLY:
        loss_log = []
        tour_log = []

        for epoch in range(50):
            optimizer.zero_grad()
            total_loss = 0.0
            total_true_length = 0.0

            for _ in range(batch_size):
                cities = generate_cities(n)
                dist = compute_distance_matrix(cities)
                heatmap = build_heatmap_from_gnn(model, cities)
                heatmap = torch.tanh((heatmap - heatmap.mean()) / 2.0)
                loss_b = surrogate_loss(heatmap, dist)
                total_loss += loss_b
                tour = decode_greedy(heatmap.detach().numpy(), start=0)
                tour = two_opt(tour, dist)
                total_true_length += tour_length(tour, dist)

            loss = total_loss / batch_size
            avg_true_length = total_true_length / batch_size
            loss_log.append(loss.item())
            tour_log.append(avg_true_length)
            loss.backward()
            optimizer.step()

            val_tour = eval_on_batch(model, val_cities, val_dists)
            if val_tour < best_val_tour:
                best_val_tour = val_tour
                patience_counter = 0
                torch.save(model.state_dict(), "best_utsp_model.pt")
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Train tour: {avg_true_length:.4f} | Val tour: {val_tour:.4f}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        import json
        with open("training_log.json", "w") as f:
            json.dump({"loss": loss_log, "avg_tour": tour_log}, f, indent=2)

    if EVAL_ONLY:
        model.load_state_dict(torch.load("best_utsp_model.pt"))
        model.eval()
        cities = generate_cities(n)
        dist = compute_distance_matrix(cities)
        heatmap = build_heatmap_from_gnn(model, cities)
        heatmap = torch.tanh((heatmap - heatmap.mean()) / 2.0)
        tour = decode_greedy(heatmap.detach().numpy(), start=0)
        tour = two_opt(tour, dist)
        length = tour_length(tour, dist)
        random_heatmap = torch.rand((n, n))
        random_heatmap.fill_diagonal_(0.0)
        baseline = decode_greedy(random_heatmap.numpy(), start=0)
        baseline = two_opt(baseline, dist)
        baseline_length = tour_length(baseline, dist)
        print("Eval tour length:", length)
        print("Baseline tour length:", baseline_length)
        print("Improvement:", baseline_length - length)
