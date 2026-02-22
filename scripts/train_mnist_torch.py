#!/usr/bin/env python3
"""Train MNIST with alrelu-torch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchvision is required for MNIST dataset loading. "
        "Install it with: pip install torchvision"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from alrelu_torch import ALReLU, TrainableALReLU
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "packages" / "alrelu-torch" / "src"))
    from alrelu_torch import ALReLU, TrainableALReLU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST using alrelu-torch.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument(
        "--variant",
        type=str,
        default="fixed",
        choices=["fixed", "learnable", "trainable"],
        help="ALReLU variant: fixed alpha or learnable alpha.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Initial/fixed alpha value.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory where MNIST data will be stored.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device.",
    )
    return parser.parse_args()


def make_activation(variant: str, alpha: float) -> nn.Module:
    if variant == "fixed":
        return ALReLU(alpha=alpha)
    return TrainableALReLU(alpha_init=alpha, non_negative=True)


class MNISTNet(nn.Module):
    def __init__(self, variant: str, alpha: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.act1 = make_activation(variant, alpha)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act2 = make_activation(variant, alpha)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.act3 = make_activation(variant, alpha)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.act3(self.fc1(x))
        return self.fc2(x)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mnist(data_dir: str, batch_size: int, seed: int):
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    val_size = 5000
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_full, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    variant = "learnable" if args.variant == "trainable" else args.variant

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    train_loader, val_loader = load_mnist(args.data_dir, args.batch_size, args.seed)

    model = MNISTNet(variant=variant, alpha=args.alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    final_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = running_loss / total_samples
        final_val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_accuracy: {final_val_acc:.4f}"
        )

    print(f"Final val accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
