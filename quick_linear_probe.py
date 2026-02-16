#!/usr/bin/env python3
"""
🎯 TRUE Linear Probe с REAL CIFAR100 labels + GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import argparse
import numpy as np

def load_cifar100_labels():
    """Load REAL CIFAR100 train/test labels"""
    # Train labels
    with open('./data/cifar-100-python/train', 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
    train_labels = torch.tensor(train_data[b'fine_labels']).long()
    
    # Test labels  
    with open('./data/cifar-100-python/test', 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    test_labels = torch.tensor(test_data[b'fine_labels']).long()
    
    return train_labels, test_labels

def linear_probe(features_path, num_epochs=200, lr=1e-2, batch_size=512):
    device = torch.device('cuda')
    
    # REAL labels
    train_labels, test_labels = load_cifar100_labels()
    train_labels, test_labels = train_labels.to(device), test_labels.to(device)
    
    # Features
    data = torch.load(features_path, map_location=device)
    train_feats = data['train']  # (50000, 768)
    test_feats = data['test']    # (10000, 768)
    
    print(f"REAL Labels: train {train_labels.shape}, test {test_labels.shape}")
    print(f"Features match: train {len(train_feats)} == labels {len(train_labels)} ✓")
    
    # Train/val split
    N = len(train_feats)
    idx = torch.randperm(N, device=device)
    train_size = int(0.9 * N)
    train_idx, val_idx = idx[:train_size], idx[train_size:]
    
    # Model
    model = nn.Linear(train_feats.shape[1], 100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        
        # Training loop
        for start in range(0, len(train_idx), batch_size):
            end = min(start + batch_size, len(train_idx))
            batch_idx = train_idx[start:end]
            
            X = train_feats[batch_idx]
            y = train_labels[batch_idx]
            
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_correct = 0
            for start in range(0, len(val_idx), batch_size):
                end = min(start + batch_size, len(val_idx))
                batch_idx = val_idx[start:end]
                
                X = train_feats[batch_idx]
                y = train_labels[batch_idx]
                logits = model(X)
                pred = logits.argmax(1)
                val_correct += (pred == y).sum().item()
            
            val_acc = val_correct / len(val_idx)
        
        if epoch % 20 == 0 or val_acc > best_val_acc:
            print(f"Ep {epoch:3d}: val_acc={val_acc:.1%} (best: {best_val_acc:.1%})")
            best_val_acc = max(best_val_acc, val_acc)
    
    # Test
    model.eval()
    with torch.no_grad():
        test_logits = model(test_feats)
        test_pred = test_logits.argmax(1)
        test_acc = (test_pred == test_labels).float().mean()
    
    print(f"\n🎯 **REAL RESULTS**:")
    print(f"  Best Val:  {best_val_acc:.1%}")
    print(f"  **TEST ACC: {test_acc:.1%}** ← Table 2!")
    
    return test_acc.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_path")
    args = parser.parse_args()
    
    linear_probe(args.features_path)