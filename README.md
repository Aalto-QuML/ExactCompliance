# ExactCompliance

[Mattie Ji](https://github.com/maroon-scorch) | [Indradyumna Roy](https://indradyumna.github.io/) | [Vikas Garg](https://www.aalto.fi/en/people/vikas-kumar-garg)

Official repository for the paper **"Exact Topological Compliance: Generating
Persistence-Equivalent Graphs"** (NeurIPS 2026).

## Citation

```bibtex
@inproceedings{
exactcompliance,
title={Exact Topological Compliance: Generating Persistence-Equivalent Graphs},
author={Mattie Ji and Indradyumna Roy and Vikas K Garg},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=xcQkqD4abA}
}
```

---

## Overview

We study *strict* persistence-diagram (PD) compliant graph generation: producing
graphs whose persistent-homology signature under a filtration Φ matches a target PD
**exactly**, rather than approximately. The repository contains three families of
methods:

1. **Local iterative samplers** (Section 3) — rejection samplers that build a
   PD-equivalent filtration stage by stage, with soundness and completeness
   guarantees and an impossibility result (Theorem 3.10).
2. **A global constraint-programming (CP-SAT) encoding** (Section 4) — recasts
   generation as a constraint-satisfaction problem that is sound, complete, and
   minimal for the degree filtration, and extends to many other vertex filtrations.
3. **Topology-aware neural diffusion baselines** — ConStruct and DiGress, each with
   the TAGG conditioning module (PDM loss + TAM attention), which approximate the
   target PD via a learned reverse-diffusion process.

Benchmarks: **comm20**, **planar**, **sbm**, **enzymes** (SPECTRE 64/16/20% splits).

---

## Repository layout

```
.
├── topo_gen/                       # Core algorithmic library
│   ├── filtrations.py              #   vertex / degree / random filtrations
│   ├── persistence.py              #   H0/H1 persistence (union-find + gudhi)
│   ├── degree_filtration_cp_v4.py  #   degree-filtration CP-SAT solver (Section 4)
│   ├── filtration_cp.py            #   PLUGGABLE-filtration CP-SAT solver (Section 4.2)
│   ├── *_pd_equivalent*.py         #   local samplers: Algorithms 1/2/3 (Section 3)
│   └── pdm_utils.py                #   PDM loss + TAM landscape for the TAGG baselines
├── baselines/ConStruct/            # Vendored ConStruct/DiGress diffusion backbones
├── configs/                        # Hydra configuration tree
├── scripts/                        # CP / sampler evaluation + experiment entry points (see below)
├── cli/                            # Training / data-prep entry points
│   ├── dump_data.py                #   dump benchmark splits into a flat pickle format
│   ├── train_{construct,digress}.py      #   train/eval vanilla diffusion baselines
│   └── train_{construct,digress}_tagg.py #   train/eval + TAGG (PDM loss + TAM bias)
├── utils/                          # Dataset / generation / visualization helpers
└── environment.yml                 # Conda environment specification
```

Pre-computed benchmark splits (`data/`) and the generated graphs that back the
paper's tables (`dumps/`, `nx_dumps/`) are shipped with the repository so the reported
numbers can be verified directly (see **Reproducing the paper's results** below).

---

## The pluggable-filtration CP solver (`topo_gen/filtration_cp.py`)

`filtration_cp.py` implements the global CP encoding for **any** vertex filtration Φ
whose per-vertex value admits a finite constraint certificate (Section 4.2 and
Appendix J). Only constraint C1 (Φ-alignment) is filtration-specific; the rest of the
encoding (C2–C6, connectivity, Euler, birth-time survival) is Φ-agnostic. The
registered `phi_type` values are:

| `phi_type`    | Φ(v) | class | encoding |
|---------------|------|-------|----------|
| `degree`      | deg(v) | linear | `Σ_j M_ij = a_i` |
| `sqdeg`       | deg(v)² | polynomial | integer square |
| `wedge`       | deg(v)(deg(v)−1) | polynomial | integer product |
| `triangle`    | # triangles through v | polynomial | AND-linearized edge products, O(n³) aux |
| `fourclique`  | # 4-cliques through v | polynomial | O(n⁴) aux (capped) |
| `clustering`  | 2·tri / (deg(deg−1)) | rational | cleared denominator, per-graph integer scale |
| `avgneighdeg` | avg. neighbor degree | rational | cleared denominator |
| `closeness`   | closeness centrality | fixed-point | BFS distance via Bellman–Ford certificate |
| `katz`        | Katz centrality | linear fixed-point | pin x = 1 + αMx directly (no matrix inverse) |
| `pagerank`    | PageRank | linear fixed-point | stochastic fixed point, degree-quotient vars |


Example:

```python
import networkx as nx
from topo_gen.filtration_cp import solve_from_graph

G = nx.cycle_graph(6)
H, p0, p1 = solve_from_graph(G, phi_type="clustering", time_limit=60)
# H realizes G's clustering-coefficient PD exactly (up to graph isomorphism)
```

---

## Experiment scripts (`scripts/`)

| script | what it does | paper |
|--------|--------------|-------|
| `cp_v4_dataset_eval.py`          | degree-filtration CP solver on benchmark test splits | Table 2 |
| `generate_1_per_graph.py`        | local samplers (Alg. 1/2/3) on benchmark splits | Table 2, App H.4 |
| `filtration_cp_verify.py`        | soundness + exhaustive completeness for the polynomial/rational filtrations | App J |
| `filtration_cp_fixedpoint_verify.py` | soundness + completeness for closeness / Katz / PageRank | App J |
| `filtration_cp_matrix.py`        | consolidated correctness matrix (all encodable + refused filtrations) | App J |
| `filtration_cp_adversarial.py`   | adversarial verification (negative controls, hard graph families) | App J |
| `filtration_cp_bench.py`         | triangle/clustering CP success rate + runtime on benchmarks | App J |
| `filtration_cp_diversity.py`     | structural diversity within a degree-PD class (A: enumeration, B: sampler, C: CP-random) | App K |
| `precompute_mu.py`               | precompute TAGG persistence-landscape conditioning vectors | App H.2.3 |

The verification scripts (`filtration_cp_verify.py`, `filtration_cp_fixedpoint_verify.py`,
`filtration_cp_matrix.py`, `filtration_cp_adversarial.py`, `filtration_cp_truncation.py`)
generate their own small graphs and need no dataset download. The benchmark-scale
scripts (`filtration_cp_bench.py`, `filtration_cp_diversity.py`) read the test splits
as plain-networkx pickles from `nx_dumps/{ds}_test.pkl` or `dumps/cp_v4/*.pkl` — build
these once from the PyG splits to avoid importing torch at run time.

---

## Installation

```bash
conda env create -f environment.yml
conda activate topogen
```

The environment pulls in `pytorch`, `pytorch-lightning`, `torch-geometric`, `gudhi`,
`ortools` (CP-SAT), `hydra-core`, and the standard scientific stack. The CP-SAT solver
and the local samplers run on CPU only; a CUDA-enabled GPU is required to (re)train the
diffusion baselines.

## Quick start

```bash
# Verify the filtration encodings are sound and complete (no data needed):
python scripts/filtration_cp_verify.py            # polynomial / rational filtrations
python scripts/filtration_cp_fixedpoint_verify.py # closeness / Katz / PageRank

# Prepare benchmark splits, then run the degree-filtration CP solver:
python cli/dump_data.py
python scripts/cp_v4_dataset_eval.py --datasets comm20 planar sbm enzymes

# (Re)train a neural baseline. Training entry points live in cli/ and use Hydra;
# on some systems the OpenMP runtime must be preloaded for torch:
LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/12/libgomp.so \
  python cli/train_digress_tagg.py digress_tagg_experiment=comm20
```

---

## Data layout

```
data/{comm20,planar,sbm,enzymes}/   # processed SPECTRE benchmark splits (train/val/test)
nx_dumps/{ds}_test.pkl              # plain-networkx test splits (fast load for the CP scripts)
dumps/cp_v4/                        # degree-filtration CP results per dataset  (Table 2, Table 7)
dumps/generate_1/                   # local-sampler results (RANDOM/VERTEX/DEGREE)  (Table 2, App H.4)
dumps/diversity_graphs/             # generated PD-equivalent graph corpus         (App K)
```

All benchmark splits and generated graphs are included so the paper's numbers verify
without recompute. To regenerate a dataset from scratch instead: `python cli/dump_data.py dataset=<name>`.

## Reproducing the paper's results

**Soundness & completeness of the encodings (App J).** No data needed; each script
generates its own small graphs and checks the generated graph's PD against the target
by independent recomputation + exhaustive enumeration up to isomorphism:

```bash
python scripts/filtration_cp_verify.py            # degree, triangle, clustering, sqdeg, wedge,
                                                  #   fourclique, avgneighdeg  -> 36/36 sound, 5/5 complete
python scripts/filtration_cp_fixedpoint_verify.py # closeness, Katz, PageRank  -> sound + complete
python scripts/filtration_cp_matrix.py            # consolidated matrix
```

**Table 2 (CP success rate) & Table 7 (SBM coverage)** — read directly from the shipped
CP results (`dumps/cp_v4/`), or re-run with `scripts/cp_v4_dataset_eval.py`. The shipped
dumps give comm20 100%, planar 100%, sbm 82.5% (33/40; 7 largest graphs time out),
enzymes 100%.

**App K (diversity within a degree-PD class)** — computed from the generated-graph
corpus in `dumps/diversity_graphs/` (median #distinct graphs 20/18/14/20 for
comm20/planar/sbm/enzymes; the PD-determined invariants β0, β1, average degree and
density are identical across every class member), or re-run end-to-end with
`scripts/filtration_cp_diversity.py`.
