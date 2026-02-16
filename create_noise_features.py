#!/usr/bin/env python3
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", required=True, help="e.g. ./features/vit_base_patch14_dinov2.lvd142m_cifar100.pt")
parser.add_argument("--noise_dim", type=int, required=True)
parser.add_argument("--save_path", required=True, help="e.g. ./features/vit_base_patch14_dinov2.lvd142m_noise512_cifar100.pt")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)

data = torch.load(args.base_path, map_location="cpu")
if not isinstance(data, dict):
    raise ValueError(f"Expected dict from save_features.py, got {type(data)}")

print("Keys:", list(data.keys()))

for k, v in list(data.items()):
    if torch.is_tensor(v) and v.ndim == 2:
        n, d = v.shape
        if n == 0:
            print(f"{k}: empty {v.shape}, skip")
            continue
        noise = torch.randn(n, args.noise_dim, dtype=v.dtype)
        data[k] = torch.cat([v, noise], dim=1)
        print(f"{k}: {v.shape} -> {data[k].shape}")

torch.save(data, args.save_path)
print(f"Saved: {args.save_path}")
