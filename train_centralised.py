"""

Centralised training for BraTS 2020 brain tumour segmentation.

Settings:
    Optimiser : Adam (lr=1e-4)
    Epochs : 100
    Batch size : 1
    Loss : Dice loss
    Split : 80:20 stratified on ET presence

Usage:
    python train_centralised.py \
        --data_dir ./preprocessed \
        --model    attention_unet \
        --out_dir  ./results/centralised
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from models import SimpleUNet, AttentionUNet
from metrics import DiceLoss, evaluate_model
from preprocessing import BraTSDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model",
                   choices=["simple_unet", "attention_unet"],
                   default="attention_unet")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dir", default="./results/centralised")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def stratified_split(dataset, train_ratio=0.8, seed=42):
    """80:20 stratified split preserving ET presence ratio."""
    rng = np.random.default_rng(seed)
    labels = np.array([
        int((np.load(f)["label"] == 3).any())
        for f in dataset.files
    ])
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = int(len(idx) * train_ratio)
        train_idx.extend(idx[:n].tolist())
        val_idx.extend(idx[n:].tolist())
    return train_idx, val_idx


def plot_curve(scores, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(scores)+1), scores)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Dice Coefficient")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Dataset
    dataset = BraTSDataset(args.data_dir)
    train_indices, val_indices = stratified_split(dataset, seed=args.seed)
    train_loader = DataLoader(Subset(dataset, train_indices),
                              batch_size=1, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(dataset, val_indices),
                              batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)
    print(f"Train: {len(train_indices)}  Val: {len(val_indices)}")

    # Model
    ModelClass = AttentionUNet if args.model == "attention_unet" \
                 else SimpleUNet
    model = ModelClass(in_ch=3, num_classes=4).to(device)
    model_name = ModelClass.__name__
    print(f"Model: {model_name}")

    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = DiceLoss(num_classes=4)

    # Checkpoint resume
    ckpt_pat = os.path.join(args.out_dir, "checkpoint.pth")
    best_path = os.path.join(args.out_dir, "best.pth")
    dices_path = os.path.join(args.out_dir, "val_dices.npy")
    start_epoch = 1
    best_dice  = 0.0
    val_dices  = []

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_dice"]
        val_dices = np.load(dices_path).tolist() \
                      if os.path.exists(dices_path) else []
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.long().to(device)
            optimiser.zero_grad()
            loss_fn(model(imgs), lbls).backward()
            optimiser.step()

        metrics = evaluate_model(model, val_loader, device)
        mean_dice = metrics["mean_dice"]
        val_dices.append(mean_dice)

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"Dice={mean_dice:.4f}  "
              f"WT={metrics['dice_WT']:.4f}  "
              f"TC={metrics['dice_TC']:.4f}  "
              f"ET={metrics['dice_ET']:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best: {best_dice:.4f}")

        # Save checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimiser.state_dict(),
            "best_dice": best_dice,
        }, ckpt_path)
        np.save(dices_path, np.array(val_dices))

    # Final evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))
    final = evaluate_model(model, val_loader, device)

    title = (f"Centralised \u2013 "
             f"{'Simple' if 'Simple' in model_name else 'Attention'} U-Net")
    plot_curve(val_dices, "Epochs", title,
               os.path.join(args.out_dir,
                            f"{title.replace(' ','_').replace(chr(8211),'-')}.pdf"))

    print(f"\n{'='*45}")
    print(f"FINAL RESULTS: {model_name}")
    print(f"{'='*45}")
    print(f"Mean Dice : {final['mean_dice']:.4f}")
    print(f"Mean Jaccard : {final['mean_jaccard']:.4f}")
    print(f"WT Dice : {final['dice_WT']:.4f}")
    print(f"TC Dice: {final['dice_TC']:.4f}")
    print(f"ET Dice : {final['dice_ET']:.4f}")


if __name__ == "__main__":
    main()
