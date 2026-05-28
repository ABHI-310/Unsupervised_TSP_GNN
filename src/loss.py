import torch

def surrogate_loss(heatmap, dist):
    n = heatmap.shape[0]
    device = heatmap.device

    diff_dist = dist.unsqueeze(2) - dist.unsqueeze(1)
    diff_heat = heatmap.unsqueeze(2) - heatmap.unsqueeze(1)

    mask_j = ~torch.eye(n, dtype=torch.bool, device=device).unsqueeze(2)
    mask_k = ~torch.eye(n, dtype=torch.bool, device=device).unsqueeze(1)
    mask_jk = ~torch.eye(n, dtype=torch.bool, device=device).repeat(n, 1, 1)
    
    valid_triplets_mask = mask_j & mask_k & mask_jk
    target_mask = valid_triplets_mask & (diff_dist < 0)

    if not target_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss_values = torch.relu(1.0 - diff_heat)
    return loss_values[target_mask].mean()