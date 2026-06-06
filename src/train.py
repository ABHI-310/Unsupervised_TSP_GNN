import random
import torch
import torch.optim as optim
import json
from tsp_core import generate_cities, compute_distance_matrix, tour_length, two_opt
from fake_utsp import decode_greedy
from model import GNNScorer, build_heatmap_from_gnn
from loss import surrogate_loss
import config

def eval_on_batch(model, cities_list, dists_list):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for cities, dist in zip(cities_list, dists_list):
            # heatmap = build_heatmap_from_gnn(model, cities)
            # heatmap = torch.tanh((heatmap - heatmap.mean()) / config.TEMPERATURE)
            # Replace the tanh line inside train.py with standard range scaling:
            heatmap = build_heatmap_from_gnn(model, cities)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            tour = decode_greedy(heatmap.numpy(), start=0)
            tour = two_opt(tour, dist)
            total += tour_length(tour, dist)
    return total / len(cities_list)

def main():
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)

    model = GNNScorer(config.HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    val_cities = [generate_cities(config.TRAIN_N) for _ in range(config.VAL_SIZE)]
    val_dists = [compute_distance_matrix(c) for c in val_cities]

    best_val_tour = float("inf")
    patience_counter = 0
    loss_log, tour_log = [], []

    for epoch in range(config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        total_loss, total_len = 0.0, 0.0

        for _ in range(config.BATCH_SIZE):
            cities = generate_cities(config.TRAIN_N)
            dist = compute_distance_matrix(cities)
            dist_tensor = torch.tensor(dist, dtype=torch.float32)

            # heatmap = build_heatmap_from_gnn(model, cities)
            # heatmap = torch.tanh((heatmap - heatmap.mean()) / config.TEMPERATURE)
            # Replace the tanh line inside train.py with standard range scaling:
            heatmap = build_heatmap_from_gnn(model, cities)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            loss_b = surrogate_loss(heatmap, dist_tensor)
            total_loss += loss_b

            tour = decode_greedy(heatmap.detach().numpy(), start=0)
            tour = two_opt(tour, dist)
            total_len += tour_length(tour, dist)

        loss = total_loss / config.BATCH_SIZE
        train_len = total_len / config.BATCH_SIZE

        loss.backward()
        optimizer.step()

        val_tour = eval_on_batch(model, val_cities, val_dists)
        loss_log.append(loss.item())
        tour_log.append(train_len)

        if val_tour < best_val_tour:
            best_val_tour = val_tour
            patience_counter = 0
            torch.save(model.state_dict(), "best_utsp_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            break

    with open("training_log.json", "w") as f:
        json.dump({"loss": loss_log, "avg_tour": tour_log}, f, indent=2)

if __name__ == "__main__":
    main()