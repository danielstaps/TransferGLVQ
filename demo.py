"""Synthetic multi-source demo for T-GMLVQ.

Two classes live in a 2D signal subspace. Each of the two sources adds a large,
source-specific offset along *other* dimensions (a domain shift). Everything is
then mixed by a random rotation, so the model must *learn* which directions carry
source vs class information. A naive nearest-class-prototype classifier in the raw
space is confused by the domain shift; T-GMLVQ learns Omega to separate the
sources and classifies in Omega's null-space, recovering the class structure.
"""

import argparse

import numpy as np
import torch

from transfer_gmlvq import TransferGMLVQ


def make_multisource(n_per=300, seed=0):
    rng = np.random.default_rng(seed)
    n = 6  # ambient dimension
    Xs, Yc, Ys = [], [], []
    class_centers = np.array([[-2.0, 0.0], [2.0, 0.0]])  # 2 classes in dims 0,1
    source_shift = np.array([6.0, -6.0])  # per-source offset magnitude
    for s in range(2):  # 2 sources
        for c in range(2):  # 2 classes
            pts = rng.normal(size=(n_per, n)) * 0.6
            pts[:, :2] += class_centers[c]  # class signal (dims 0,1)
            pts[:, 2] += source_shift[s]  # source-specific shift (dim 2)
            pts[:, 3] += source_shift[s] * 0.5  # and dim 3
            Xs.append(pts)
            Yc += [c] * n_per
            Ys += [s] * n_per
    X = np.concatenate(Xs, 0)
    # mix dimensions with a fixed random rotation so axes are not privileged
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    X = X @ Q
    X = (X - X.mean(0)) / X.std(0)
    return (torch.tensor(X, dtype=torch.float32), torch.tensor(Yc), torch.tensor(Ys))


def accuracy(pred, y):
    return float((pred == y).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--m", type=int, default=2, help="mapping/source-separation rank")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    X, Yc, Ys = make_multisource(seed=args.seed)
    n = X.shape[1]

    model = TransferGMLVQ(
        n_features=n, mapping_dim=args.m, n_classes=2, n_sources=2, ppc=1, pps=1, seed=args.seed
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        # anneal alpha 1 -> 0: first learn source separation, then class in null-space
        alpha = max(0.0, 1.0 - epoch / (args.epochs * 0.6))
        opt.zero_grad()
        loss = model.loss(X, Yc, Ys, alpha=alpha)
        loss.backward()
        opt.step()
        model.orthogonalize_()  # keep Omega sub-orthogonal

    # evaluation
    class_acc = accuracy(model.predict(X), Yc)
    source_acc = accuracy(model.predict_source(X), Ys)
    # sub-orthogonality check: Omega Omega^T should be ~ I_m
    with torch.no_grad():
        oo = model.omega @ model.omega.t()
        ortho_err = float((oo - torch.eye(args.m)).abs().max())

    # naive baseline: nearest class prototype in the RAW space (no null-space)
    with torch.no_grad():
        raw_d = torch.cdist(X, model.class_protos) ** 2
        naive_acc = accuracy(model.class_labels[raw_d.argmin(1)], Yc)

    print(f"n={n}, m={args.m}, 2 classes x 2 sources")
    print(f"class accuracy (null-space) : {class_acc:.3f}")
    print(f"source accuracy (Omega space): {source_acc:.3f}")
    print(f"naive class acc (raw space)  : {naive_acc:.3f}")
    print(f"||Omega Omega^T - I||_max    : {ortho_err:.2e}")


if __name__ == "__main__":
    main()
