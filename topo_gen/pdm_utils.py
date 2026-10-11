"""Utilities for the Persistence Diagram Matching (PDM) loss and
persistence-landscape-based graph embeddings (µ_G₀) used in TAGG.

Public API
----------
compute_pd_nx(G)
    Compute H0/H1 persistence diagrams for a NetworkX graph using
    degree filtration.

persistence_landscape(pd, num_landscapes, resolution)
    Vectorize a persistence diagram into a persistence landscape vector.

graph_mu(G, num_landscapes, resolution)
    Full pipeline: G → (PD0, PD1) → concatenated landscape µ_G₀.

mean_mu(graphs, num_landscapes, resolution)
    Compute the mean µ̄ over a collection of graphs (used at inference).

pdm_loss(pd0_true, pd1_true, pd0_pred, pd1_pred)
    1-Wasserstein PDM loss between two pairs of persistence diagrams,
    summed over H0 and H1 dimensions.

pdm_terms_from_adj(adj_true, adj_pred, node_mask)
    Takes dense adjacency tensors and returns the optimal matching as
    substitutable terms plus the shared sentinel cap, so the caller can
    re-evaluate the same cost from soft degrees and obtain gradients.
"""

import math
import numpy as np
import networkx as nx

from gudhi.representations import Landscape

from topo_gen.filtrations import degree_filtration
from topo_gen.persistence import persistence_diagrams


# ---------------------------------------------------------------------------
# PD computation
# ---------------------------------------------------------------------------

def compute_pd_nx(G: nx.Graph) -> tuple[list, list]:
    """Return (pd0, pd1) for G under the degree filtration.

    H0 points: (birth, death) — death=inf for surviving components.
    H1 points: (birth, inf)   — all H1 bars persist forever in 1-complexes.

    Empty graphs return ([], []).
    """
    if G.number_of_nodes() == 0:
        return [], []
    node_time, edge_time, _ = degree_filtration(G)
    return persistence_diagrams(node_time, edge_time)


# ---------------------------------------------------------------------------
# Persistence landscape
# ---------------------------------------------------------------------------

def _pd_to_array(pd: list, end: float) -> np.ndarray:
    """Convert a PD list to a float32 array, closing inf deaths at `end`.

    Eq. (5)'s f_(b,d) is piecewise linear on (b, d); with d = inf it is an
    unbounded ramp, so Eq. (6)'s "divide the domain where λ_l(x) ≥ 0 into
    (s+1) equal sub-intervals" has no finite answer and gudhi's Landscape
    returns NaN. The filtration itself supplies the missing endpoint: it
    completes at the maximum filter value, so an essential class is closed
    there rather than discarded.

    Dropping these instead zeroes H1 outright — every H1 class of a 1-complex
    is (b, inf) — which is total information loss on graph data.
    """
    pts = [(b, end if math.isinf(d) else d) for b, d in pd]
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def persistence_landscape(
    pd: list,
    num_landscapes: int = 5,
    resolution: int = 100,
    end: float = None,
) -> np.ndarray:
    """Vectorize a persistence diagram into a persistence landscape.

    Parameters
    ----------
    pd : list of (birth, death) pairs; inf deaths are closed at `end`
    num_landscapes : L — number of landscape layers
    resolution : S — number of discretization points per layer
    end : filtration endpoint for essential classes. Defaults to the largest
          finite coordinate in `pd`; pass the value shared across all
          dimensions of a graph so H0 and H1 sit on one axis.

    Returns
    -------
    np.ndarray of shape (num_landscapes * resolution,) — µ for one PD dimension
    """
    if end is None:
        end = _filtration_end(pd)
    arr = _pd_to_array(pd, end)
    out_dim = num_landscapes * resolution

    if arr.shape[0] == 0:
        return np.zeros(out_dim, dtype=np.float32)

    lsc = Landscape(num_landscapes=num_landscapes, resolution=resolution)
    vec = lsc.fit_transform([arr])   # shape (1, L*S)
    return np.nan_to_num(vec[0], nan=0.0).astype(np.float32)


def _filtration_end(*pds: list) -> float:
    """One step past the largest finite coordinate across the given diagrams.

    Under Eq. (2)'s degree filtration the last simplex enters at the maximum
    node degree, so the filtration completes just after it; surviving classes
    are closed there. The +1 matters on regular graphs, where every cycle is
    born at the single degree value present — closing exactly at that value
    would give every H1 bar zero length and land the landscape back at zero.
    """
    finite = [
        float(c)
        for pd in pds
        for b, d in pd
        for c in (b, d)
        if not math.isinf(c)
    ]
    return (max(finite) + 1.0) if finite else 1.0


# ---------------------------------------------------------------------------
# Graph-level µ_G₀
# ---------------------------------------------------------------------------

def graph_mu(
    G: nx.Graph,
    num_landscapes: int = 5,
    resolution: int = 100,
) -> np.ndarray:
    """Compute the topology embedding µ_G₀ for a single graph.

    Concatenates persistence landscapes for H0 and H1, giving a vector of
    shape (2 * num_landscapes * resolution,).

    At inference time (G₀ unknown) replace with mean_mu() over training data.
    """
    pd0, pd1 = compute_pd_nx(G)
    # One endpoint across both dimensions: the filtration completes once, so
    # H0 and H1 landscapes must share an axis to be comparable.
    end = _filtration_end(pd0, pd1)
    mu0 = persistence_landscape(pd0, num_landscapes, resolution, end=end)
    mu1 = persistence_landscape(pd1, num_landscapes, resolution, end=end)
    return np.concatenate([mu0, mu1], axis=0)  # (2 * L * S,)


def mean_mu(
    graphs: list,
    num_landscapes: int = 5,
    resolution: int = 100,
) -> np.ndarray:
    """Compute the mean µ̄ over a list of NetworkX graphs.

    Used at inference time when the original G₀ is unknown (TAGG §3.2).
    Returns array of shape (2 * num_landscapes * resolution,).
    """
    mus = [graph_mu(G, num_landscapes, resolution) for G in graphs]
    return np.mean(mus, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# PDM loss (1-Wasserstein with diagonal padding)
# ---------------------------------------------------------------------------

def essential_cap(*pds: list) -> float:
    """Shared sentinel standing in for inf deaths across the given diagrams.

    Set to max(finite coordinate) + 1 over ALL diagrams passed in, so an
    essential class gets the same death coordinate on both sides of a
    comparison. Two essential classes then differ only by their birth times,
    and identical diagrams score exactly zero. This matches the convention the
    reported W̄₁ evaluator uses (notebooks/pd_distance.ipynb).

    Computing the cap per-diagram instead would break that: identical
    topologies would score nonzero.
    """
    finite = []
    for pd in pds:
        for point in pd:
            b, d = point[0], point[1]
            finite.append(float(b))
            if not math.isinf(d):
                finite.append(float(d))
    return (max(finite) + 1.0) if finite else 1.0


def _wasserstein1_pd(pd_a: list, pd_b: list, cap: float = None) -> float:
    """1-Wasserstein distance between two persistence diagrams.

    Unmatched points are paired to their nearest point on the diagonal
    (b, d) → ((b+d)/2, (b+d)/2) with cost = persistence/2.

    Inf deaths are replaced by `cap` (default: essential_cap over both
    diagrams) rather than dropped. Dropping them zeroes the H1 term outright,
    since every H1 class in a 1-complex is (b, inf), and charges nothing for a
    wrong cycle count.

    Uses scipy linear_sum_assignment for the optimal matching.
    Handles empty diagrams gracefully.
    """
    from scipy.optimize import linear_sum_assignment

    if cap is None:
        cap = essential_cap(pd_a, pd_b)

    def capped(pd):
        return [(b, cap if math.isinf(d) else d) for b, d in pd]

    fa = capped(pd_a)
    fb = capped(pd_b)

    # Diagonal projections
    diag_a = [((b + d) / 2, (b + d) / 2) for b, d in fa]
    diag_b = [((b + d) / 2, (b + d) / 2) for b, d in fb]

    pts_a = fa + diag_b   # real points of A  +  diagonals of B
    pts_b = fb + diag_a   # real points of B  +  diagonals of A

    n = len(pts_a)
    if n == 0:
        return 0.0

    # Cost matrix: L∞ distance (standard for persistence Wasserstein)
    cost = np.zeros((n, n), dtype=np.float64)
    for i, (b1, d1) in enumerate(pts_a):
        for j, (b2, d2) in enumerate(pts_b):
            cost[i, j] = max(abs(b1 - b2), abs(d1 - d2))

    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].sum())


def pdm_loss(
    pd0_true: list,
    pd1_true: list,
    pd0_pred: list,
    pd1_pred: list,
) -> float:
    """PDM loss = W₁(PD0_true, PD0_pred) + W₁(PD1_true, PD1_pred).

    Both terms use the L∞-ground-metric 1-Wasserstein with diagonal padding,
    per-dimension matching (TAGG §4.3), and one sentinel cap shared across all
    four diagrams so H0 and H1 are on a common scale.
    """
    cap = essential_cap(pd0_true, pd1_true, pd0_pred, pd1_pred)
    return (
        _wasserstein1_pd(pd0_true, pd0_pred, cap=cap)
        + _wasserstein1_pd(pd1_true, pd1_pred, cap=cap)
    )


# ---------------------------------------------------------------------------
# Batched convenience wrapper for training
# ---------------------------------------------------------------------------

def _adj_to_nx(adj: np.ndarray, node_mask: np.ndarray) -> nx.Graph:
    """Convert a dense binary adjacency (n×n) to NetworkX, respecting mask."""
    n = int(node_mask.sum())
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                G.add_edge(i, j)
    return G


def _matched_terms(pd_a: list, pd_b: list, cap: float) -> list:
    """Optimal pairing between two tracked diagrams, as substitutable terms.

    Each returned term is (a_coords, b_coords) where a coordinate is either
    ('src', simplex) — substitute that simplex's (soft) filtration value — or
    ('const', x) for the sentinel cap and for diagonal projections, which are
    handled as constants at the fixed pairing.

    The pairing itself is combinatorial and stays detached; the caller
    re-evaluates the same L∞ cost with soft values held at this assignment.
    """
    from scipy.optimize import linear_sum_assignment

    def prep(pd):
        out = []
        for b, d, bs, ds in pd:
            bc = ("src", bs)
            dc = ("const", cap) if ds is None else ("src", ds)
            out.append((bc, dc, float(b), cap if math.isinf(d) else float(d)))
        return out

    fa, fb = prep(pd_a), prep(pd_b)

    def diag(pts):
        out = []
        for _, _, b, d in pts:
            m = (b + d) / 2.0
            out.append((("const", m), ("const", m), m, m))
        return out

    pts_a = fa + diag(fb)
    pts_b = fb + diag(fa)

    n = len(pts_a)
    if n == 0:
        return []

    cost = np.zeros((n, n), dtype=np.float64)
    for i, (_, _, b1, d1) in enumerate(pts_a):
        for j, (_, _, b2, d2) in enumerate(pts_b):
            cost[i, j] = max(abs(b1 - b2), abs(d1 - d2))

    row_ind, col_ind = linear_sum_assignment(cost)
    terms = []
    for i, j in zip(row_ind, col_ind):
        ab, ad, _, _ = pts_a[i]
        bb, bd, _, _ = pts_b[j]
        terms.append(((ab, ad), (bb, bd)))
    return terms


def pdm_terms_from_adj(
    adj_true: np.ndarray,
    adj_pred: np.ndarray,
    node_mask: np.ndarray,
) -> tuple[list, float]:
    """Pairing terms for a differentiable PDM re-evaluation.

    Runs the same hard-graph persistence and optimal matching that pdm_loss
    performs, but returns the matched terms (with provenance) and the shared
    sentinel cap instead of a collapsed scalar, so the caller can recompute
    the identical cost from soft degrees and get gradients.

    Returns
    -------
    (terms, cap) where terms covers H0 and H1 concatenated.
    """
    from topo_gen.filtrations import degree_filtration
    from topo_gen.persistence import persistence_diagrams_tracked

    G_true = _adj_to_nx(adj_true, node_mask)
    G_pred = _adj_to_nx(adj_pred, node_mask)

    nt_t, et_t, _ = degree_filtration(G_true)
    nt_p, et_p, _ = degree_filtration(G_pred)
    pd0_t, pd1_t = persistence_diagrams_tracked(nt_t, et_t)
    pd0_p, pd1_p = persistence_diagrams_tracked(nt_p, et_p)

    cap = essential_cap(pd0_t, pd1_t, pd0_p, pd1_p)
    terms = _matched_terms(pd0_t, pd0_p, cap) + _matched_terms(pd1_t, pd1_p, cap)
    return terms, cap
