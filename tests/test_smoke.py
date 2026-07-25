"""Smoke test: T-GMLVQ separates sources in Omega and classifies in the null-space."""

import torch

from demo import make_multisource
from transfer_gmlvq import TransferGMLVQ


def test_transfer_gmlvq_nullspace_classification():
    torch.manual_seed(0)
    X, Yc, Ys = make_multisource(n_per=100, seed=0)
    model = TransferGMLVQ(
        n_features=X.shape[1],
        mapping_dim=2,
        n_classes=2,
        n_sources=2,
        seed=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for epoch in range(150):
        alpha = max(0.0, 1.0 - epoch / 90.0)
        opt.zero_grad()
        model.loss(X, Yc, Ys, alpha=alpha).backward()
        opt.step()
        model.orthogonalize_()

    class_acc = (model.predict(X) == Yc).float().mean().item()
    assert class_acc > 0.8  # transfer works via the null-space
