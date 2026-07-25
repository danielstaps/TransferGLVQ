# T-GLVQ (TensorFlow) — Transfer Learning GLVQ via null-space evaluation

`protoflow` implementation of the transfer-learning LVQ architecture from:

> D. Staps, J. Ravichandran, S. Saralajew, M. Biehl, M. Kaden, T. Villmann,
> **"A Learning Vector Quantization Architecture for Transfer Learning Based Classification in
> Case of Multiple Sources by Means of Null-Space Evaluation"**,
> *Advances in Intelligent Data Analysis XX (IDA 2022)*, Springer LNCS 13205, 2022, pp. 354–364.
> DOI: [10.1007/978-3-031-01333-1_28](https://doi.org/10.1007/978-3-031-01333-1_28)

## What this is

A Generalized Matrix LVQ (GMLVQ) architecture for **transfer learning across multiple sources**:
knowledge from several source domains is combined and transferred to a target task, using a
**null-space evaluation** of the learned relevance/mapping to decide which directions carry
transferable class information and which should be suppressed. The result is an interpretable,
prototype-based classifier that reuses multi-source structure. Implemented in TensorFlow/Keras as
the `protoflow` package.

## Install & run

```bash
pip install -r requirements.txt
python3 -m examples.bonbons        # example training script
```

Original dependencies (per the paper-time README): keras ≥ 2.9.0, tensorflow ≥ 2.9.1,
numpy ≥ 1.23.1, matplotlib ≥ 3.5.2.

## Branches (repo-cleanup convention)

| Branch / tag | Meaning |
|---|---|
| `published` / `v-paper-2022a` | frozen as-published snapshot (earliest available git state of the model; see note) |
| `main` | maintained/cleaned state |

**Provenance note:** the model code exists in git only as an August-2022 snapshot (post-conference
upload); no commit at the IDA 2022 submission deadline (2021-11-19) survives for the model itself.
The **experiment** code with full deadline-time history lives in
[`T-GLVQ_experiments_paper`](https://github.com/danielstaps/T-GLVQ_experiments_paper)
(`v-paper-2022a` there points at the last pre-deadline commit).

## How to cite

See [`CITATION.cff`](CITATION.cff) and cite the paper above.

## Acknowledgment

By **Daniel Staps** ([0009-0002-4459-4544](https://orcid.org/0009-0002-4459-4544)),
J. Ravichandran, S. Saralajew, M. Biehl, M. Kaden and **Thomas Villmann**
([0000-0001-6725-0141](https://orcid.org/0000-0001-6725-0141)). Repository cleanup with assistance
from Claude (Anthropic).

## License

MIT — see [LICENSE](LICENSE).
