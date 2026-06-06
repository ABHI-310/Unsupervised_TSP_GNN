# import torch

# def surrogate_loss(heatmap, dist):
#     n = heatmap.shape[0]
#     device = heatmap.device

#     diff_dist = dist.unsqueeze(2) - dist.unsqueeze(1)
#     diff_heat = heatmap.unsqueeze(2) - heatmap.unsqueeze(1)

#     mask_j = ~torch.eye(n, dtype=torch.bool, device=device).unsqueeze(2)
#     mask_k = ~torch.eye(n, dtype=torch.bool, device=device).unsqueeze(1)
#     mask_jk = ~torch.eye(n, dtype=torch.bool, device=device).repeat(n, 1, 1)
    
#     valid_triplets_mask = mask_j & mask_k & mask_jk
#     target_mask = valid_triplets_mask & (diff_dist < 0)

#     if not target_mask.any():
#         return torch.tensor(0.0, device=device, requires_grad=True)

#     loss_values = torch.relu(1.0 - diff_heat)
#     return loss_values[target_mask].mean()

import torch

def surrogate_loss(heatmap, dist, margin=1.0):
    device = heatmap.device
    n = heatmap.shape[0]
    
    # 3D Tensor operations: [anchor, j, k]
    diff_dist = dist.unsqueeze(2) - dist.unsqueeze(1)  # dist_ij - dist_ik
    diff_heat = heatmap.unsqueeze(2) - heatmap.unsqueeze(1)  # heat_ij - heat_ik
    
    # Correct index masking for 3D triplet space
    idx = torch.arange(n, device=device)
    mask_j = (idx.unsqueeze(1) != idx.unsqueeze(0)).unsqueeze(2)   # i != j
    mask_k = (idx.unsqueeze(1) != idx.unsqueeze(0)).unsqueeze(1)   # i != k
    mask_jk = (idx.unsqueeze(1) != idx.unsqueeze(0)).unsqueeze(0)  # j != k
    valid_triplets_mask = mask_j & mask_k & mask_jk
    
    # Target: Node j is closer than Node k (dist_ij < dist_ik)
    target_mask = valid_triplets_mask & (diff_dist < 0)

    if not target_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # If j is closer, heat_ij should be significantly larger than heat_ik
    # Penalty occurs when margin - (heat_ij - heat_ik) > 0
    loss_values = torch.relu(margin - diff_heat)
    return loss_values[target_mask].mean()