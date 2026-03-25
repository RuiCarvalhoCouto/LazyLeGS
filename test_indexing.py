import torch

N = 10
D = 5
x = torch.randn(N, D)
mask_1d = torch.zeros(N, dtype=torch.bool)
mask_1d[0:2] = True

mask_2d = mask_1d.unsqueeze(1) # [N, 1]

print(f"x shape: {x.shape}")
print(f"mask_1d shape: {mask_1d.shape}")
print(f"mask_2d shape: {mask_2d.shape}")

try:
    out_1d = x[mask_1d]
    print(f"x[mask_1d] shape: {out_1d.shape}")
except Exception as e:
    print(f"x[mask_1d] error: {e}")

try:
    out_2d = x[mask_2d]
    print(f"x[mask_2d] shape: {out_2d.shape}")
except Exception as e:
    print(f"x[mask_2d] error: {e}")
