"""
Federated learning for BraTS 2020 brain tumour segmentation.

Implements:
    RandomClientSelector : uniform random client sampling
    HybridClientSelector : 2 round-robin + 1 random per round
    FedAvg aggregation : weighted average by dataset size
    Local SGD training : one epoch per round per client

Federated setup:
    5 simulated hospitals: 120, 80, 50, 30, 15 samples
    3 clients per round
    50 communication rounds

Usage:
    python federated_learning.py \
        --data_dir ./preprocessed \
        --strategy hybrid \
        --model    simple_unet \
        --out_dir  ./results/federated
"""

import os
import copy
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from models import SimpleUNet, AttentionUNet
from metrics import DiceLoss, evaluate_model
from preprocessing import BraTSDataset


CLIENT_SIZES = [120, 80, 50, 30, 15]


class RandomClientSelector:
    """Algorithm 1: uniform random sampling each round."""
    def __init__(self, K: int, m: int):
        self.K, self.m = K, m
    def select(self, rnd: int) -> list:
        return random.sample(range(self.K), self.m)


class HybridClientSelector:
    """
    Phase II hybrid strategy:
    2 clients via round-robin + 1 client at random.
    Ensures fairness while introducing stochastic diversity.
    """
    def __init__(self, K: int, rr: int = 2, rand: int = 1):
        self.K, self.rr, self.rand = K, rr, rand
    def select(self, rnd: int) -> list:
        start = (rnd * self.rr) % self.K
        rr = [(start + i) % self.K for i in range(self.rr)]
        rest = [c for c in range(self.K) if c not in rr]
        return rr + random.sample(rest, min(self.rand, len(rest)))

def fedavg_aggregate(global_model, client_states, client_weights):
    total = sum(client_weights)
    new_state = copy.deepcopy(client_states[0])
    for key in new_state:
        t = new_state[key]
        if not t.is_floating_point():
            new_state[key] = client_states[0][key].clone()
            continue
        new_state[key] = torch.zeros_like(t, dtype=torch.float32)
        for s, w in zip(client_states, client_weights):
            new_state[key] += s[key].float() * (w / total)
        new_state[key] = new_state[key].to(t.dtype)
    global_model.load_state_dict(new_state)



def local_train(model, loader, loss_fn, lr, device):
    """One epoch of local SGD training (Section III-C)."""
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr,
                    momentum=0.9, weight_decay=1e-4)
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.long().to(device)
        opt.zero_grad()
        loss_fn(model(imgs), lbls).backward()
        opt.step()
    return copy.deepcopy(model.state_dict())



def partition_dataset(dataset, sizes, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(dataset))
    rng.shuffle(idx)
    shards, ptr = [], 0
    for i, s in enumerate(sizes):
        end = ptr + s if i < len(sizes) - 1 else len(idx)
        shards.append(Subset(dataset, idx[ptr:end].tolist()))
        ptr = end
    return shards



def plot_curve(scores, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(scores)+1), scores)
    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Dice Coefficient")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def federated_train(ModelClass, selector, strategy_name,
                    num_rounds=50, lr=1e-4, seed=42,
                    data_dir="./preprocessed",
                    out_dir="./results/federated"):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = os.path.join(out_dir, strategy_name)
    os.makedirs(save_dir, exist_ok=True)

    # Build datasets
    dataset = BraTSDataset(data_dir)
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(dataset))
    rng.shuffle(all_idx)
    n_train = sum(CLIENT_SIZES)
    val_idx = all_idx[:len(dataset) - n_train].tolist()
    tr_idx = all_idx[len(dataset) - n_train:].tolist()

    val_loader = DataLoader(Subset(dataset, val_idx),
                                 batch_size=1, shuffle=False,
                                 num_workers=2, pin_memory=True)
    client_datasets = partition_dataset(
        Subset(dataset, tr_idx), CLIENT_SIZES, seed)
    client_sizes = [len(d) for d in client_datasets]

    model = ModelClass(in_ch=3, num_classes=4).to(device)
    loss_fn = DiceLoss(num_classes=4)

    # Checkpoint resume
    ckpt_path = os.path.join(save_dir, "checkpoint.pth")
    best_path = os.path.join(save_dir, "best.pth")
    dices_path = os.path.join(save_dir, "val_dices.npy")
    start_rnd = 0
    best_dice = 0.0
    val_dices = []

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_rnd = ckpt["round"]
        best_dice = ckpt["best_dice"]
        val_dices = np.load(dices_path).tolist() \
                    if os.path.exists(dices_path) else []
        print(f"Resuming from round {start_rnd + 1}")

    print(f"\nStrategy: {strategy_name} | "
          f"Clients: {client_sizes} | Rounds: {num_rounds}\n")

    for rnd in range(start_rnd, num_rounds):
        selected = selector.select(rnd)
        states, sizes = [], []
        for cid in selected:
            lm = copy.deepcopy(model).to(device)
            ld = DataLoader(client_datasets[cid],
                            batch_size=1, shuffle=True, num_workers=0)
            states.append(local_train(lm, ld, loss_fn, lr, device))
            sizes.append(client_sizes[cid])

        fedavg_aggregate(model, states, sizes)

        metrics = evaluate_model(model, val_loader, device)
        mean_dice = metrics["mean_dice"]
        val_dices.append(mean_dice)

        print(f"Round {rnd+1:03d}/{num_rounds}  "
              f"Clients={selected}  "
              f"Dice={mean_dice:.4f}  "
              f"WT={metrics['dice_WT']:.4f}  "
              f"TC={metrics['dice_TC']:.4f}  "
              f"ET={metrics['dice_ET']:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best: {best_dice:.4f}")

        torch.save({
            "round": rnd + 1,
            "model_state": model.state_dict(),
            "best_dice": best_dice,
        }, ckpt_path)
        np.save(dices_path, np.array(val_dices))

    # Final results
    model.load_state_dict(torch.load(best_path, map_location=device))
    final = evaluate_model(model, val_loader, device)

    if "hybrid" in strategy_name.lower():
        title = "Hybrid Client Selection \u2013 Simple U-Net"
    else:
        mname = ("Simple" if "Simple" in ModelClass.__name__
                 else "Attention")
        title = f"Federated \u2013 {mname} U-Net"

    plot_curve(val_dices, title,
               os.path.join(save_dir,
                            f"{title.replace(' ','_').replace(chr(8211),'-')}.pdf"))

    print(f"\n{'='*45}")
    print(f"FINAL RESULTS: {strategy_name}")
    print(f"{'='*45}")
    print(f"Mean Dice : {final['mean_dice']:.4f}")
    print(f"Mean Jaccard : {final['mean_jaccard']:.4f}")
    print(f"WT Dice : {final['dice_WT']:.4f}")
    print(f"TC Dice : {final['dice_TC']:.4f}")
    print(f"ET Dice : {final['dice_ET']:.4f}")
    return final


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--strategy",
                   choices=["random", "hybrid"],
                   default="hybrid")
    p.add_argument("--model",
                   choices=["simple_unet", "attention_unet"],
                   default="simple_unet")
    p.add_argument("--rounds",type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dir", default="./results/federated")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ModelClass = (AttentionUNet if args.model == "attention_unet"
                  else SimpleUNet)
    selector = (HybridClientSelector(len(CLIENT_SIZES))
                  if args.strategy == "hybrid"
                  else RandomClientSelector(len(CLIENT_SIZES), m=3))
    federated_train(
        ModelClass = ModelClass,
        selector = selector,
        strategy_name = f"{ModelClass.__name__}_{args.strategy}",
        num_rounds = args.rounds,
        lr = args.lr,
        data_dir = args.data_dir,
        out_dir = args.out_dir,
    )
