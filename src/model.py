import torch
import torch.nn as nn

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
