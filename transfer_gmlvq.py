"""Transfer-GMLVQ (T-GMLVQ) — PyTorch port.

Null-space transfer classification learning with a siamese-like GMLVQ, after

    Villmann, Staps, Ravichandran, Saralajew, Biehl, Kaden,
    "A Learning Vector Quantization Architecture for Transfer Learning Based
     Classification in Case of Multiple Sources by Means of Null-Space Evaluation",
    IDA 2022, LNCS 13205, pp. 354-364. doi:10.1007/978-3-031-01333-1_28

Two prototype sets share one sub-orthogonal mapping Omega in R^{m x n} (m < n):

  * source prototypes omega_j (source labels), separated in the projection space:
        d_Omega(x, w)  = || Omega (x - w) ||^2                          (paper eq. 1)
  * class prototypes w_k (class labels), separated in the NULL-space of Omega,
    via the complementary orthogonal projector Q = I_n - Omega^T Omega:
        delta_Omega(x, w) = || Q (x - w) ||^2                           (paper eq. 2)

Both use the GLVQ classifier function  mu = (d+ - d-) / (d+ + d-)  and the loss
combines them (paper eq. 5). Omega is kept sub-orthogonal (Omega Omega^T = I_m)
by re-orthogonalization after each step (the TF reference used a Gram-Schmidt
callback; here we use a QR step, which is equivalent and numerically robust).

This is a clean, dependency-light reimplementation (torch only) — it does not
build on prototorch, because T-GMLVQ needs two prototype sets and the null-space
distance, which do not map onto prototorch's single-set GMLVQ.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _sq_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances, a:[N,d] vs b:[K,d] -> [N,K]."""
    return torch.cdist(a, b) ** 2


def glvq_mu(distances: torch.Tensor, proto_labels: torch.Tensor,
            y: torch.Tensor) -> torch.Tensor:
    """GLVQ classifier function mu = (d+ - d-) / (d+ + d-), per sample.

    distances:[N,K], proto_labels:[K], y:[N] -> [N]. Negative where correct.
    """
    match = proto_labels.unsqueeze(0) == y.unsqueeze(1)          # [N,K]
    big = distances.max().detach() + 1.0
    d_plus = torch.where(match, distances, big).min(dim=1).values
    d_minus = torch.where(~match, distances, big).min(dim=1).values
    return (d_plus - d_minus) / (d_plus + d_minus + 1e-12)


class TransferGMLVQ(nn.Module):
    """T-GMLVQ model. `alpha` weights the SOURCE term (matches the TF reference's
    `alpha_source`; typically annealed 1 -> 0 so Omega first learns source
    separation, then class discrimination sharpens in the null-space)."""

    def __init__(self, n_features: int, mapping_dim: int, n_classes: int,
                 n_sources: int, ppc: int = 1, pps: int = 1, seed: int | None = None):
        super().__init__()
        if not mapping_dim < n_features:
            raise ValueError("mapping_dim (m) must be < n_features (n)")
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        self.n_features = n_features
        self.mapping_dim = mapping_dim
        self.omega = nn.Parameter(torch.randn(mapping_dim, n_features, generator=g) * 0.1)
        self.class_protos = nn.Parameter(torch.randn(n_classes * ppc, n_features, generator=g) * 0.1)
        self.source_protos = nn.Parameter(torch.randn(n_sources * pps, n_features, generator=g) * 0.1)
        self.register_buffer("class_labels", torch.arange(n_classes).repeat_interleave(ppc))
        self.register_buffer("source_labels", torch.arange(n_sources).repeat_interleave(pps))
        self.orthogonalize_()

    def complementary_projector(self) -> torch.Tensor:
        """Q = I_n - Omega^T Omega  (orthogonal projection onto the null-space of Omega)."""
        n = self.n_features
        return torch.eye(n, device=self.omega.device, dtype=self.omega.dtype) - self.omega.t() @ self.omega

    def forward(self, x: torch.Tensor):
        """Returns (source_distances[N,Ks], class_distances[N,Kc])."""
        x_omega = x @ self.omega.t()
        p_omega = self.source_protos @ self.omega.t()
        source_distances = _sq_euclidean(x_omega, p_omega)          # eq. 1

        q = self.complementary_projector()
        x_q = x @ q.t()
        p_q = self.class_protos @ q.t()
        class_distances = _sq_euclidean(x_q, p_q)                   # eq. 2
        return source_distances, class_distances

    def loss(self, x: torch.Tensor, y_class: torch.Tensor, y_source: torch.Tensor,
             alpha: float, f=None) -> torch.Tensor:
        """T-GMLVQ loss = alpha * source_loss + (1 - alpha) * class_loss  (paper eq. 5).

        `f` is the monotonic squashing function applied to mu/nu (default: identity,
        as in the TF reference). Pass torch.sigmoid for a bounded variant."""
        f = (lambda t: t) if f is None else f
        d_source, d_class = self.forward(x)
        nu = glvq_mu(d_source, self.source_labels, y_source)        # source classifier (eq. 3)
        mu = glvq_mu(d_class, self.class_labels, y_class)           # class classifier  (eq. 4)
        return alpha * f(nu).mean() + (1.0 - alpha) * f(mu).mean()

    @torch.no_grad()
    def orthogonalize_(self):
        """Make Omega sub-orthogonal (Omega Omega^T = I_m) via a QR step."""
        q, _ = torch.linalg.qr(self.omega.t())     # q: [n, m] with orthonormal columns
        self.omega.copy_(q.t())

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class label (nearest class prototype in the null-space)."""
        _, d_class = self.forward(x)
        return self.class_labels[d_class.argmin(dim=1)]

    @torch.no_grad()
    def predict_source(self, x: torch.Tensor) -> torch.Tensor:
        d_source, _ = self.forward(x)
        return self.source_labels[d_source.argmin(dim=1)]
