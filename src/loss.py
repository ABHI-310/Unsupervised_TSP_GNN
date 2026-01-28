import torch

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
