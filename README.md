# T-GMLVQ (PyTorch) — null-space transfer classification learning

A clean **PyTorch** reimplementation of Transfer-GMLVQ (T-GMLVQ) from:

> T. Villmann, D. Staps, J. Ravichandran, S. Saralajew, M. Biehl, M. Kaden,
> **"A Learning Vector Quantization Architecture for Transfer Learning Based Classification in
> Case of Multiple Sources by Means of Null-Space Evaluation"**,
> *Advances in Intelligent Data Analysis XX (IDA 2022)*, Springer LNCS 13205, 2022, pp. 354–364.
> DOI: [10.1007/978-3-031-01333-1_28](https://doi.org/10.1007/978-3-031-01333-1_28)

The `main` branch is the modernized **PyTorch** port. The original **TensorFlow** implementation
lives on the [`tensorflow`](https://github.com/danielstaps/TransferGLVQ/tree/tensorflow) branch, and
the exact **as-published** paper state (TensorFlow) on the `published` branch and tag `v-paper-2022a`.

## Method

Classify data from several **sources** correctly, independent of the source domain. A siamese-like
GMLVQ shares one sub-orthogonal mapping `Ω ∈ ℝ^{m×n}` (m < n, `ΩΩᵀ = Iₘ`) between two prototype sets:

```
source prototypes ω_j :  d_Ω(x, ω) = ‖ Ω (x − ω) ‖²          # separate sources in ℝ^m   (eq. 1)
class  prototypes w_k :  δ_Ω(x, w) = ‖ Q (x − w) ‖²,  Q = Iₙ − ΩᵀΩ   # classify in Ω's null-space (eq. 2)
```

Both use the GLVQ classifier `μ = (d⁺ − d⁻)/(d⁺ + d⁻)`; the loss combines them,
`E = Σ α·f(ν_source) + (1−α)·f(μ_class)` (eq. 5). `α` is annealed 1→0: Ω first learns to separate
sources, then class discrimination sharpens in the null-space where source differences are levelled.

Implemented in **plain torch** (no prototorch dependency): T-GMLVQ needs two prototype sets and the
null-space distance, which don't map onto prototorch's single-set GMLVQ.

## Run

```bash
pip install -r requirements.txt
python demo.py --epochs 400
```

The synthetic multi-source demo prints, e.g.:

```
class accuracy (null-space) : 0.999      # T-GMLVQ classifies across sources
source accuracy (Omega space): 1.000     # Ω separates the sources
naive class acc (raw space)  : 0.502     # naive classifier ≈ chance (domain shift)
||Omega Omega^T - I||_max    : 1.2e-07   # Ω stays sub-orthogonal
```

The gap between the naive baseline (≈ chance) and the null-space accuracy (≈ 1.0) is the transfer
effect the paper describes. Full paper reproduction uses the authors' multi-source datasets.

## Files

| File | Purpose |
|---|---|
| `transfer_gmlvq.py` | `TransferGMLVQ` model, GLVQ loss, orthogonalization, prediction |
| `demo.py` | synthetic multi-source example + evaluation |

## How to cite

See [`CITATION.cff`](CITATION.cff) and cite the paper above.

## Acknowledgment

Port by **Daniel Staps** ([0009-0002-4459-4544](https://orcid.org/0009-0002-4459-4544)); method by
T. Villmann, D. Staps, J. Ravichandran, S. Saralajew, M. Biehl, M. Kaden. PyTorch port carried out
with assistance from Claude (Anthropic).

## License

MIT — see [LICENSE](LICENSE).
