"""
Pluggable-Filtration CP Solver
==============================
Generalizes ``degree_filtration_cp_v4.solve`` from the degree filtration to any
vertex filtration Phi whose per-vertex value is an *exact* polynomial/rational
function of the adjacency matrix M.

Mathematical setup
------------------
A vertex filtration assigns each vertex v a value Phi(v).  The induced
vertex-edge filtration uses node_time[v] = Phi(v) and
edge_time[(u,v)] = max(Phi(u), Phi(v)).  Its H0/H1 persistence diagrams
(P0, P1) are the topological target we must reproduce.

Two arrays are threaded through the solver (see the CRITICAL SUBTLETY in the
design doc):

  stage[i]      : integer 0..K = rank of vertex i's DISTINCT Phi value.
                  This is what C2..C9 use as the active-set threshold.  It is
                  the exact analogue of the old ``vd`` array, and the target PD
                  (P0, P1) fed to C2..C9 is computed on the STAGE-reduced
                  filtration so it stays integer-valued exactly as the degree
                  solver expects (paper Remark 2.3, the Z>=0 reduction).

  phi_target[i] : the ACTUAL Phi value of vertex i (degree, triangle count, or
                  the integer-scaled clustering numerator).  This drives C1 ONLY.

For the DEGREE filtration stage == phi_target == degree, because degree is both
the birth time (an integer stage) and the literal filtration value; the two
arrays coincide and this solver reproduces ``degree_filtration_cp_v4.solve``
bit-for-bit (regression: run_suite still passes).

For TRIANGLE / CLUSTERING they DIVERGE: phi_target[i] (triangle count) can be
any integer >= 0 while stage[i] is merely the rank of that value among the
distinct values present.

Only constraint C1 is filtration-specific.  It is isolated in
``phi_alignment(...)``.  C2..C9 + connectivity + ALIVE are reused verbatim from
the degree solver and consume ``stage`` unchanged.

Guarantee boundary
------------------
Soundness + completeness transfer to *any* Phi that can be written as a finite
polynomial (triangle count) or cleared-denominator rational (local clustering)
in the Boolean edge variables M_ij, because those admit an exact CP-SAT
encoding.  Spectral filtrations (eigenvector / subgraph centrality, Katz,
PageRank) and shortest-path filtrations (closeness, betweenness) are NOT finite
polynomials/rationals in M -- leading-eigenvector selection, a matrix
exponential, and a shortest-path minimum are all outside the encodable class --
so this module refuses them with NotImplementedError.  Minimality (Thm 4.2.3)
is degree-specific and does not transfer; no minimality test is written here.
"""

import itertools
import math
import time
from collections import defaultdict
from functools import reduce

import networkx as nx
import numpy as np
from ortools.sat.python import cp_model

# Reuse the persistence machinery (union-find + gudhi cross-check).
from topo_gen.persistence import persistence_diagrams, gudhi_persistence
# Reuse boolean helpers and PH-comparison utilities from the degree solver so
# nothing is duplicated / can drift.
from topo_gen.degree_filtration_cp_v4 import (
    bool_and_var,
    bool_or_vars,
    norm_ph,
    ph_equal,
)


# ─────────────────────────────────────────────────────────────────────────────
# Phi computation on a concrete networkx graph (ground truth, no CP)
# ─────────────────────────────────────────────────────────────────────────────

def clustering_scale_for(G: nx.Graph) -> int:
    """Exact common denominator D such that D * C_i is an integer for EVERY
    vertex, where C_i = 2 t_i / (d_i (d_i - 1)).

    Because t_i, d_i are known integers for the concrete source graph, choosing
    D = lcm{ d_i (d_i - 1) : d_i >= 2 } guarantees p_i = 2 D t_i / (d_i(d_i-1))
    is an exact integer.  A fixed scale (e.g. 12) does NOT: for d_i=5 the
    denominator is 20 and 12*C_i can be fractional (e.g. 12*0.3 = 3.6), which
    rounds to an unrealizable target and yields a spurious INFEASIBLE.

    Returns at least 1 (graphs with all deg<2 have no constraint denominator).
    """
    denoms = {G.degree(v) * (G.degree(v) - 1) for v in G.nodes() if G.degree(v) >= 2}
    if not denoms:
        return 1
    return reduce(lambda a, b: a * b // math.gcd(a, b), denoms)


def avgneighdeg_scale_for(G: nx.Graph) -> int:
    """Exact common denominator D such that D * ANeighDeg(v) is an integer for
    EVERY vertex, where ANeighDeg(v) = (Sum_j M_ij d_j) / d_v (0 when d_v = 0).

    The denominator is d_v, so D = lcm{ d_v : d_v >= 1 } makes every
    p_v = D * (num_v / d_v) an exact integer.  Same spurious-INFEASIBLE trap as
    clustering if a fixed scale is used instead.
    """
    degs = {G.degree(v) for v in G.nodes() if G.degree(v) >= 1}
    if not degs:
        return 1
    return reduce(lambda a, b: a * b // math.gcd(a, b), degs)


# ═════════════════════════════════════════════════════════════════════════════
# FIXED-POINT-CERTIFIABLE filtrations  (closeness / katz / pagerank)
# ═════════════════════════════════════════════════════════════════════════════
# These are NOT finite polynomials/rationals in M, so the old code refused them.
# They ARE, however, *fixed-point certifiable*: each is the UNIQUE solution of a
# finite constraint system, and CP can pin that solution exactly by adding
# auxiliary variables whose unique feasible assignment equals Phi.  This is the
# same trick as the paper's spanning-tree connectivity certificate (C4b): the
# witness variables are not a closed form, they are a *certificate* forced to be
# correct by local-consistency + a witness family.
#
#   closeness : BFS shortest-path distance via local-smoothness + a fully-reified
#               predecessor witness (Bellman-Ford local optimality).  Distances
#               are forced to the true SP distance; closeness is then the rational
#               (n-1)/S_i pinned by a cleared-denominator constraint.
#   katz      : x = 1 + alpha*M*x  ((I-alpha M) x = 1).  Cleared to integers, the
#               linear system has a UNIQUE integer-scaled solution X_i = D*x_i,
#               with D = det-denominator of (I - alpha M).  NO matrix inverse.
#   pagerank  : x_i = (1-alpha)/n + alpha*sum_{j->i} x_j/d_j.  Same idea, but the
#               per-vertex degree division needs quotient variables, so D must
#               ALSO clear the degrees: D = lcm(denoms(x)) * lcm(degrees).
#
# WARNING (encodable != tractable): the scale D is a graph-GLOBAL determinant.
# It grows exponentially and overflows CP-SAT's ~2^62 safe integer domain.  When
# the cleared right-hand side would exceed FIXEDPOINT_SAFE_BOUND the builder
# raises ScaleOverflow, which the harness REPORTS as an honest finding, not a
# hidden failure.  See katz_scale_for / pagerank_scale_for for the exact D.

from fractions import Fraction as _Fraction

# CP-SAT integer variables (and every product intermediate) must lie inside
# [-kint64max/2, kint64max/2] ~ +-2^62.  We keep everything strictly below 2^61
# so that the largest cleared right-hand side still validates.
FIXEDPOINT_SAFE_BOUND = 2 ** 61

KATZ_ALPHA_FP_NUM, KATZ_ALPHA_FP_DEN = 1, 10      # Katz alpha = 1/10
PR_ALPHA_FP_NUM,   PR_ALPHA_FP_DEN   = 17, 20     # PageRank damping alpha = 17/20

# The fixed-point-certifiable filtrations.  Their cleared integer coefficients are
# large (a graph-global determinant / distance-sum LCM), and CP-SAT's PRESOLVE has
# a coefficient-overflow path that can wrongly report INFEASIBLE on these large-D
# instances even when the model is genuinely feasible (verified: the same model
# solves OPTIMAL with presolve disabled).  We therefore re-solve fixed-point
# INFEASIBLE results once with presolve OFF to distinguish a genuine "no graph
# realizes this PD" from a presolve artifact.  This changes ONLY solver
# parameters, never the model (C2..C9 untouched).
_FIXEDPOINT = {"closeness", "katz", "pagerank"}


class ScaleOverflow(Exception):
    """Raised when the exact integer scale D (or the cleared RHS it induces) for a
    fixed-point filtration exceeds the CP-SAT safe 64-bit domain.  The filtration
    is still *theoretically* exactly encodable via its fixed point; the rational
    scale simply blows up.  Carries the offending ``n`` and ``bits`` for reporting.
    """

    def __init__(self, phi_type, n, D, bits):
        self.phi_type = phi_type
        self.n = n
        self.D = D
        self.bits = bits
        super().__init__(
            f"{phi_type}: exact scale D has {bits} bits (n={n}); exceeds CP-SAT "
            f"safe domain (~62 bits). Theoretically exact via the fixed point, but "
            f"the rational scale overflows -- encodable != tractable.")


def _solve_linear_exact(A_rows, b):
    """Exact Gaussian elimination over the rationals (Fraction).  ``A_rows`` is a
    list of lists of Fractions (n x n), ``b`` a list of Fractions.  Returns the
    unique solution list of Fractions, or raises ValueError if singular.
    """
    n = len(b)
    M2 = [list(A_rows[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = None
        for r in range(c, n):
            if M2[r][c] != 0:
                piv = r
                break
        if piv is None:
            raise ValueError("singular linear system")
        M2[c], M2[piv] = M2[piv], M2[c]
        pv = M2[c][c]
        M2[c] = [v / pv for v in M2[c]]
        for r in range(n):
            if r != c and M2[r][c] != 0:
                f = M2[r][c]
                M2[r] = [M2[r][k] - f * M2[c][k] for k in range(n + 1)]
    return [M2[i][n] for i in range(n)]


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def closeness_targets(G: nx.Graph):
    """Exact integer-scaled closeness targets on a CONNECTED graph.

    closeness_i = (n-1) / S_i,  S_i = sum_{j!=i} dist(i,j).
    Scale D = lcm{ S_i } so p_i = D*(n-1)/S_i is an EXACT integer.  Returns
    (D, {node: p_i}, {node: S_i}).  Raises ValueError if G is not connected.
    """
    n = G.number_of_nodes()
    if n <= 1 or not nx.is_connected(G):
        raise ValueError("closeness_targets requires a connected graph with n>=2")
    sp = dict(nx.all_pairs_shortest_path_length(G))
    S = {i: sum(sp[i][j] for j in range(n) if j != i) for i in range(n)}
    D = reduce(_lcm, S.values())
    p = {}
    for i in range(n):
        num = D * (n - 1)
        assert num % S[i] == 0
        p[i] = num // S[i]
    return D, p, S


def katz_scale_for(G: nx.Graph):
    """Exact scale D and integer targets X_i = D*x_i for Katz centrality x solving
    (I - alpha M) x = 1, alpha = 1/10, via exact rational linear solve.  D is the
    lcm of the denominators of the exact solution (= det-denominator of (I-alpha M)
    up to reduction).  Returns (D, {node: X_i}).
    """
    n = G.number_of_nodes()
    alpha = _Fraction(KATZ_ALPHA_FP_NUM, KATZ_ALPHA_FP_DEN)
    A = [[_Fraction(1) if i == j else _Fraction(0) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and G.has_edge(i, j):
                A[i][j] -= alpha
    b = [_Fraction(1) for _ in range(n)]
    x = _solve_linear_exact(A, b)
    D = 1
    for xi in x:
        D = _lcm(D, xi.denominator)
    tgt = {}
    for i in range(n):
        val = x[i] * D
        assert val.denominator == 1
        tgt[i] = int(val)
    return D, tgt


def pagerank_scale_for(G: nx.Graph):
    """Exact scale D and integer targets X_i = D*x_i for PageRank x solving
    x_i = (1-alpha)/n + alpha*sum_{j: j~i} x_j/d_j, alpha = 17/20, on a graph with
    no dangling nodes required to be handled specially here (isolated vertices
    carry only the teleport term).  D = lcm(denoms(x)) * lcm(degrees>=1) so that
    BOTH D*x_j and D*x_j/d_j are exact integers (the quotient vars need the second).
    Returns (D, {node: X_i}).
    """
    n = G.number_of_nodes()
    an, ad = PR_ALPHA_FP_NUM, PR_ALPHA_FP_DEN
    alpha = _Fraction(an, ad)
    deg = [G.degree(i) for i in range(n)]
    A = [[_Fraction(1) if i == j else _Fraction(0) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.has_edge(j, i) and deg[j] > 0:
                A[i][j] -= alpha * _Fraction(1, deg[j])
    b = [_Fraction(ad - an, ad) / n for _ in range(n)]
    x = _solve_linear_exact(A, b)
    Ld = 1
    for xi in x:
        Ld = _lcm(Ld, xi.denominator)
    Lg = 1
    for d in set(deg):
        if d >= 1:
            Lg = _lcm(Lg, d)
    D = Ld * Lg
    tgt = {}
    for i in range(n):
        val = x[i] * D
        assert val.denominator == 1, f"pagerank scale not integral at v{i}"
        tgt[i] = int(val)
    return D, tgt


def scale_for(G: nx.Graph, phi_type: str) -> int:
    """Resolve the exact integer scale D for a rational filtration; 1 for the
    integer-valued (polynomial) filtrations."""
    if phi_type == "clustering":
        return clustering_scale_for(G)
    if phi_type == "avgneighdeg":
        return avgneighdeg_scale_for(G)
    if phi_type == "closeness":
        return closeness_targets(G)[0]
    if phi_type == "katz":
        return katz_scale_for(G)[0]
    if phi_type == "pagerank":
        return pagerank_scale_for(G)[0]
    return 1


# ═════════════════════════════════════════════════════════════════════════════
# TRUNCATED / APPROXIMATE filtrations  (finite-K surrogates for REFUSED schemes)
# ═════════════════════════════════════════════════════════════════════════════
# Each Phi^(K)(v) is a finite-K polynomial/rational in M, so CP can pin it
# EXACTLY, but its K -> infinity limit is a non-encodable spectral / random-walk
# filtration.  The experiment (see scripts/filtration_cp_truncation.py) measures:
#   Axis A -- can we ENCODE Phi^(K) (feasibility, solve time, scaling integer D)?
#   Axis B -- does the solved H reproduce the TRUE (converged) filtration of G?
# The whole point: CP pins the *truncated* value (truncated-W1 ~ 0) but the TRUE
# value of H generally differs from G (true-W1 > 0), and the gap need not close
# at any feasibly-encodable K.
#
# Parameters:  KATZ_ALPHA (well-conditioned, denominator 10^K),
#              PR_ALPHA   (PageRank damping 0.85 = 17/20).
KATZ_ALPHA_NUM, KATZ_ALPHA_DEN = 1, 10        # alpha = 0.1
PR_ALPHA_NUM,   PR_ALPHA_DEN   = 17, 20       # alpha = 0.85

_TRUNCATED = {"eigvec_K", "katz_K", "rw_K", "pagerank_K"}


def is_truncated(phi_type: str) -> bool:
    return phi_type in _TRUNCATED


def _adj_int(G, n):
    """Symmetric 0/1 adjacency as a list of lists on vertices 0..n-1."""
    A = [[0] * n for _ in range(n)]
    for u, v in G.edges():
        A[u][v] = A[v][u] = 1
    return A


def _matvec_int(A, x, n):
    """Integer M x for symmetric A (list of lists) and integer vector x."""
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]


def truncated_scale_for(G, phi_type, K):
    """Integer scale D that makes D * Phi^(K)(v) an exact integer for every v.

      eigvec_K   : polynomial, D = 1.
      katz_K     : alpha = 1/10 so terms have denominator 10^k; D = 10^K.
      rw_K       : each of K steps divides by a vertex degree; the walk numerator
                   over a common denominator uses D = L^K with L = lcm(degrees>=1)
                   (an upper bound that clears every per-step degree).  This is the
                   scale that BLOWS UP with K.
      pagerank_K : alpha = 17/20 damping AND K degree-normalisation steps;
                   D = (20)^K * L^K  (both the damping denominator and the walk
                   denominator), the worst blow-up.
    """
    if phi_type == "eigvec_K":
        return 1
    if phi_type == "katz_K":
        return KATZ_ALPHA_DEN ** K
    degs = [G.degree(v) for v in G.nodes() if G.degree(v) >= 1]
    L = 1
    for d in set(degs):
        L = L * d // math.gcd(L, d)
    if phi_type == "rw_K":
        return L ** K
    if phi_type == "pagerank_K":
        # clear: n (from (1-alpha)/n) * 20^(K+1) (damping, incl. the base 1/20)
        # * L^K (walk degrees).  The extra power of 20 clears the (1-alpha)=3/20.
        n = G.number_of_nodes()
        return n * (PR_ALPHA_DEN ** (K + 1)) * (L ** K)
    raise ValueError(phi_type)


def truncated_phi_values(G, phi_type, K, scale=None):
    """Exact integer-scaled Phi^(K)(v) on a concrete graph, via exact fraction
    arithmetic (Fraction), then multiplied by ``scale`` (must be a multiple of
    every denominator; pass None to resolve via truncated_scale_for)."""
    from fractions import Fraction
    n = G.number_of_nodes()
    A = _adj_int(G, n)
    deg = [G.degree(v) for v in range(n)]
    D = truncated_scale_for(G, phi_type, K) if scale is None else scale

    if phi_type == "eigvec_K":
        # x = M^K 1  (unnormalised power iteration)
        x = [1] * n
        for _ in range(K):
            x = _matvec_int(A, x, n)
        return {v: D * x[v] for v in range(n)}, D

    if phi_type == "katz_K":
        # x = sum_{k=1}^K alpha^k M^k 1,  alpha = 1/10
        alpha = Fraction(KATZ_ALPHA_NUM, KATZ_ALPHA_DEN)
        acc = [Fraction(0)] * n
        mk = [1] * n
        for k in range(1, K + 1):
            mk = _matvec_int(A, mk, n)
            coef = alpha ** k
            acc = [acc[i] + coef * mk[i] for i in range(n)]
        out = {}
        for v in range(n):
            val = acc[v] * D
            assert val.denominator == 1, f"katz scale {D} not integral at v{v}"
            out[v] = int(val)
        return out, D

    if phi_type == "rw_K":
        # x = (M^T D^-1)^K 1 = K-step random-walk occupancy (start uniform 1).
        x = [Fraction(1)] * n
        for _ in range(K):
            nx_ = [Fraction(0)] * n
            for i in range(n):
                for j in range(n):
                    if A[j][i] and deg[j] >= 1:
                        nx_[i] += Fraction(x[j], deg[j])
            x = nx_
        out = {}
        for v in range(n):
            val = x[v] * D
            assert val.denominator == 1, f"rw scale {D} not integral at v{v}"
            out[v] = int(val)
        return out, D

    if phi_type == "pagerank_K":
        # pi^(K) = (1-alpha)/n * sum_{k=0}^K alpha^k (M^T D^-1)^k 1
        # We define the integer target as  T_i = D * pi^(K)_i, expressed in the
        # SAME closed form the CP builder pins:  D carries n * 20^K * L^K, and
        # walk_k (exact k-step occupancy) is multiplied by an integer coefficient
        #   coeff_k = from_num * PR_NUM^k * PR_DEN^(K-k)                    (integer)
        # while D contributes n*L^K to the (1/n)*(walk denominators) clearing.
        # Concretely: T_i = sum_k coeff_k * (Lpow * walk_k[i]),  Lpow = L^K,
        # and one checks T_i == D * pi^(K)_i.
        from_num = PR_ALPHA_DEN - PR_ALPHA_NUM          # 1-alpha numerator (=3)
        Lpow = D // (n * (PR_ALPHA_DEN ** (K + 1)))     # = L^K
        walk = [Fraction(1)] * n
        walks_int = [[Lpow * 1 for _ in range(n)]]      # Lpow * walk_0  (integers)
        for k in range(1, K + 1):
            nw = [Fraction(0)] * n
            for i in range(n):
                for j in range(n):
                    if A[j][i] and deg[j] >= 1:
                        nw[i] += Fraction(walk[j], deg[j])
            walk = nw
            row = []
            for i in range(n):
                val = Lpow * walk[i]
                assert val.denominator == 1, f"pagerank Lpow {Lpow} not integral"
                row.append(int(val))
            walks_int.append(row)
        out = {}
        for v in range(n):
            T = sum(from_num * (PR_ALPHA_NUM ** k) * (PR_ALPHA_DEN ** (K - k))
                    * walks_int[k][v] for k in range(0, K + 1))
            out[v] = int(T)
        # sanity: T_i / D == pi^(K)_i
        alpha = Fraction(PR_ALPHA_NUM, PR_ALPHA_DEN)
        base = Fraction(from_num, PR_ALPHA_DEN * n)
        walk2 = [Fraction(1)] * n
        acc = [base * walk2[i] for i in range(n)]
        for k in range(1, K + 1):
            nw = [Fraction(0)] * n
            for i in range(n):
                for j in range(n):
                    if A[j][i] and deg[j] >= 1:
                        nw[i] += Fraction(walk2[j], deg[j])
            walk2 = nw
            acc = [acc[i] + base * (alpha ** k) * walk2[i] for i in range(n)]
        for v in range(n):
            assert Fraction(out[v], D) == acc[v], (
                f"pagerank target/D mismatch at v{v}: {Fraction(out[v], D)} vs {acc[v]}")
        return out, D

    raise ValueError(phi_type)


# ── TRUE (converged) filtration values, for round-trip fidelity (Axis B) ─────

def true_phi_values(G, phi_type):
    """The TRUE (K->infinity) real-valued filtration corresponding to a
    truncated scheme, computed with networkx at full convergence.  Returns a
    dict node -> float.  Used only for measuring the fidelity gap, never encoded.
    """
    n = G.number_of_nodes()
    if phi_type == "eigvec_K":
        # true eigenvector centrality (normalised leading eigenvector)
        try:
            return dict(nx.eigenvector_centrality_numpy(G))
        except Exception:
            return {v: 0.0 for v in G.nodes()}
    if phi_type == "katz_K":
        try:
            return dict(nx.katz_centrality_numpy(
                G, alpha=KATZ_ALPHA_NUM / KATZ_ALPHA_DEN))
        except Exception:
            return {v: 0.0 for v in G.nodes()}
    if phi_type == "pagerank_K":
        try:
            return dict(nx.pagerank(G, alpha=PR_ALPHA_NUM / PR_ALPHA_DEN))
        except Exception:
            return {v: 1.0 / n for v in G.nodes()}
    if phi_type == "rw_K":
        # stationary distribution of the random walk = d_v / (2|E|) on each
        # connected component (converged K-step occupancy).
        deg = dict(G.degree())
        out = {}
        for comp in nx.connected_components(G):
            m2 = sum(deg[v] for v in comp)
            for v in comp:
                out[v] = (deg[v] / m2) if m2 > 0 else 0.0
        for v in G.nodes():
            out.setdefault(v, 0.0)
        return out
    raise ValueError(phi_type)


def phi_values(G: nx.Graph, phi_type: str, clustering_scale=None):
    """Compute the per-vertex Phi value of every node of ``G``, independently
    of the CP encoding (plain networkx), returning a dict node -> value.

    For ``phi_type='clustering'`` the *exact scaled integer* value is returned:
    p_i = clustering_scale * C_i, computed as 2 * D * t_i // (d_i (d_i - 1)) with
    D = ``clustering_scale``.  ``clustering_scale`` MUST be a common multiple of
    every d_i(d_i-1) so that p_i is an exact integer; pass None to have it
    computed exactly via ``clustering_scale_for(G)`` (recommended).  The raw
    float clustering coefficients are available via ``phi_values_raw``.
    """
    if phi_type == "degree":
        return {v: G.degree(v) for v in G.nodes()}

    if phi_type == "triangle":
        # networkx triangles(G)[v] = number of triangles through v.
        return dict(nx.triangles(G))

    if phi_type == "clustering":
        D = clustering_scale_for(G) if clustering_scale is None else clustering_scale
        tri = nx.triangles(G)
        out = {}
        for v in G.nodes():
            d = G.degree(v)
            if d < 2:
                out[v] = 0                      # convention C_i = 0
                continue
            num = 2 * D * tri[v]
            den = d * (d - 1)
            assert num % den == 0, (
                f"clustering_scale={D} not a multiple of denominator {den} for "
                f"vertex {v} (deg {d}); pass clustering_scale=None for exact D")
            out[v] = num // den
        return out

    # ── Tier 1 (polynomial in M) ──────────────────────────────────────────
    if phi_type == "sqdeg":
        # squared degree Phi(v) = d_v^2
        return {v: G.degree(v) ** 2 for v in G.nodes()}

    if phi_type == "wedge":
        # number of 2-paths (wedges) centred at v: Phi(v) = d_v (d_v - 1)
        return {v: G.degree(v) * (G.degree(v) - 1) for v in G.nodes()}

    if phi_type == "fourclique":
        # number of K4's through v = # of 4-subsets {v,j,k,l} that are complete.
        # Count via networkx: for each vertex, count 4-cliques containing it.
        cnt = {v: 0 for v in G.nodes()}
        for clq in nx.enumerate_all_cliques(G):
            if len(clq) == 4:
                for v in clq:
                    cnt[v] += 1
        return cnt

    # ── Tier 2 (rational, cleared-denominator) ────────────────────────────
    if phi_type == "avgneighdeg":
        # average neighbour degree ANeighDeg(v) = (Sum_j M_vj d_j) / d_v; 0 if d_v=0.
        D = avgneighdeg_scale_for(G) if clustering_scale is None else clustering_scale
        deg = dict(G.degree())
        out = {}
        for v in G.nodes():
            dv = deg[v]
            if dv == 0:
                out[v] = 0                      # convention ANeighDeg = 0
                continue
            num = D * sum(deg[u] for u in G.neighbors(v))
            assert num % dv == 0, (
                f"avgneighdeg scale={D} not a multiple of denominator {dv} for "
                f"vertex {v}; pass clustering_scale=None for exact D")
            out[v] = num // dv
        return out

    # ── Tier 3 (fixed-point certifiable): closeness / katz / pagerank ─────
    # These return the EXACT integer-scaled per-vertex target (Phi * D), the same
    # value C1 pins.  ``clustering_scale`` (if given) is IGNORED here because the
    # scale D is intrinsic to the fixed point (a determinant / distance-sum LCM),
    # not a free choice; the caller must thread the SAME D returned by scale_for.
    if phi_type == "closeness":
        _, p, _ = closeness_targets(G)          # raises ValueError if disconnected
        return p

    if phi_type == "katz":
        _, tgt = katz_scale_for(G)
        return tgt

    if phi_type == "pagerank":
        _, tgt = pagerank_scale_for(G)
        return tgt

    _refuse(phi_type)


def phi_values_raw(G: nx.Graph, phi_type: str):
    """Raw (unscaled) Phi values -- floats for clustering."""
    if phi_type == "clustering":
        return dict(nx.clustering(G))
    if phi_type == "closeness":
        return dict(nx.closeness_centrality(G))
    if phi_type == "katz":
        return dict(nx.katz_centrality_numpy(
            G, alpha=KATZ_ALPHA_FP_NUM / KATZ_ALPHA_FP_DEN, normalized=False))
    if phi_type == "pagerank":
        return dict(nx.pagerank(G, alpha=PR_ALPHA_FP_NUM / PR_ALPHA_FP_DEN))
    return phi_values(G, phi_type)


_TIER3 = {
    "eigenvector":  "leading-eigenvector selection is not a finite polynomial in M",
    "subgraph":     "the matrix exponential exp(M) is not a finite polynomial in M",
    "betweenness":  "shortest-path counting/selection is not a polynomial/rational in M",
    # NOTE: closeness / katz / pagerank were formerly refused here.  They are NOT
    # finite polynomials/rationals in M, but they ARE fixed-point CERTIFIABLE:
    # each is the unique solution of a finite constraint system whose auxiliary
    # variables are forced to equal Phi (see the Tier-3 C1 builders above).  They
    # are therefore now supported (subject to the ScaleOverflow caveat for
    # katz/pagerank, whose exact rational scale D is a graph-global determinant).
}


def _refuse(phi_type: str):
    reason = _TIER3.get(
        phi_type,
        "unknown filtration; only {degree, triangle, clustering} are exactly encodable",
    )
    raise NotImplementedError(
        f"phi_type={phi_type!r} is not exactly encodable in CP-SAT: {reason}. "
        f"Exact soundness+completeness transfer only to filtrations that are "
        f"finite polynomials (triangle) or cleared-denominator rationals "
        f"(clustering) in the Boolean edge variables M_ij."
        # Scaffold note (Katz / PageRank): an MILP fixed-point encoding
        # (I - alpha M) x == b with x integer-scaled would be required; that is
        # a different modelling regime and is deliberately not attempted here.
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage reduction (paper Remark 2.3): map distinct Phi values -> integer stages
# ─────────────────────────────────────────────────────────────────────────────

def stage_reduction(phi_by_node: dict):
    """Given node -> Phi value, return
        nodes        : list of nodes in a fixed order
        phi_target   : list of raw Phi values in that order
        stage        : list of integer stages (rank of distinct Phi value)
        stage_to_phi : dict stage -> phi value (both directions kept)
    Consistency assertion: equal Phi <=> equal stage.
    """
    nodes = list(phi_by_node.keys())
    distinct = sorted(set(phi_by_node.values()))
    phi_to_stage = {p: s for s, p in enumerate(distinct)}
    stage_to_phi = {s: p for p, s in phi_to_stage.items()}

    phi_target = [phi_by_node[v] for v in nodes]
    stage = [phi_to_stage[phi_by_node[v]] for v in nodes]

    # equal Phi <=> equal stage (both directions)
    for a in range(len(nodes)):
        for b in range(len(nodes)):
            same_phi = phi_target[a] == phi_target[b]
            same_stage = stage[a] == stage[b]
            assert same_phi == same_stage, "stage reduction violates equal-Phi<=>equal-stage"

    return nodes, phi_target, stage, stage_to_phi


def target_pd_from_graph(G: nx.Graph, phi_type: str, clustering_scale=None,
                         cross_check: bool = True):
    """Build the STAGE-reduced target persistence diagram (P0, P1) that the
    C2..C9 machinery consumes.

    The reduction guarantees integer, contiguous filtration times, which is what
    the degree solver's stage arithmetic assumes.  Returns:
        p0, p1        : lists of (birth, death) / (birth, inf), integer-valued
        stage_to_phi  : dict stage -> phi value
        phi_by_node   : node -> raw phi (scaled for clustering)
        scale         : the resolved integer clustering scale D actually used
                        (1 for degree/triangle; the exact LCM for clustering)
    """
    if clustering_scale is None:
        scale = scale_for(G, phi_type)   # exact D for rational types; 1 otherwise
    else:
        scale = clustering_scale
    phi_by_node = phi_values(G, phi_type, clustering_scale=scale)
    nodes, phi_target, stage, stage_to_phi = stage_reduction(phi_by_node)
    stage_of = {v: s for v, s in zip(nodes, stage)}

    node_time = {v: stage_of[v] for v in G.nodes()}
    edge_time = {
        (min(u, v), max(u, v)): max(stage_of[u], stage_of[v])
        for u, v in G.edges()
    }
    p0, p1 = persistence_diagrams(node_time, edge_time)

    if cross_check:
        g0, g1 = gudhi_persistence(node_time, edge_time)
        # union-find returns death=inf as float('inf'); gudhi returns inf too.
        assert sorted(p0) == sorted(g0), (
            f"union-find/gudhi H0 mismatch: {sorted(p0)} vs {sorted(g0)}")
        assert sorted(p1) == sorted(g1), (
            f"union-find/gudhi H1 mismatch: {sorted(p1)} vs {sorted(g1)}")

    # Convert to the solver's convention: death None means essential (inf).
    p0_conv = [(int(b), None if d == float("inf") else int(d)) for b, d in p0]
    p1_conv = [(int(b), float("inf")) for b, _ in p1]
    return p0_conv, p1_conv, stage_to_phi, phi_by_node, scale


# ─────────────────────────────────────────────────────────────────────────────
# C1 strategy: phi_alignment
# ─────────────────────────────────────────────────────────────────────────────

def phi_alignment(model, M, n, phi_target, phi_type, stage, aux_cfg, triangle_y=None):
    """Add the filtration-specific C1 constraints pinning Phi_M(i) == phi_target[i]
    for all vertices i.  Returns a dict with auxiliary bookkeeping
    (``n_aux`` = number of auxiliary variables created, and any reusable
    structures such as the per-triple triangle Booleans).

    ``M`` is a dict (i,j)->BoolVar for i<j; ``ev`` recovers the undirected edge.
    ``triangle_y`` may be passed in to reuse already-built triple Booleans
    (clustering reuses triangle auxiliaries).
    """
    def ev(u, v):
        return M[min(u, v), max(u, v)]

    info = {"n_aux": 0, "triangle_y": triangle_y}

    if phi_type == "degree":
        # C1_degree: sum_j M_ij == phi_target[i]   (== stage[i] == vd[i])
        for i in range(n):
            model.add(sum(ev(i, j) for j in range(n) if j != i) == phi_target[i])
        return info

    if phi_type == "triangle":
        y = triangle_y if triangle_y is not None else _build_triangle_y(model, ev, n, info)
        # C1_triangle: sum over triples containing i of y_{ijk} == phi_target[i]
        for i in range(n):
            terms = [y[t] for t in y if i in t]
            model.add(sum(terms) == phi_target[i])
        info["triangle_y"] = y
        return info

    if phi_type == "clustering":
        # Local clustering C_i = 2 t_i / (d_i (d_i - 1)); convention C_i = 0 when d_i < 2.
        # Targets p_i = round(D * C_i) are constants; D = clustering_scale.
        # Cleared-denominator constraint (d_i >= 2):  2*D*t_i == p_i * dd_i
        # where dd_i = d_i*(d_i-1).  When d_i <= 1 we force p_i == 0 (C=0 convention).
        D = aux_cfg["clustering_scale"]
        y = triangle_y if triangle_y is not None else _build_triangle_y(model, ev, n, info)
        info["triangle_y"] = y

        for i in range(n):
            # degree of i
            d_i = model.new_int_var(0, n - 1, f"deg_{i}")
            model.add(d_i == sum(ev(i, j) for j in range(n) if j != i))
            info["n_aux"] += 1

            # dd_i = d_i * (d_i - 1)
            dm1 = model.new_int_var(-1, n - 1, f"degm1_{i}")
            model.add(dm1 == d_i - 1)
            dd_i = model.new_int_var(0, (n - 1) * (n - 1), f"dd_{i}")
            model.add_multiplication_equality(dd_i, [d_i, dm1])
            info["n_aux"] += 2

            # t_i = number of triangles through i (reuse triple Booleans)
            t_terms = [y[t] for t in y if i in t]
            t_i = model.new_int_var(0, len(t_terms) if t_terms else 0, f"tri_{i}")
            model.add(t_i == sum(t_terms))
            info["n_aux"] += 1

            # reify d_i >= 2
            has_two = model.new_bool_var(f"deg_ge2_{i}")
            model.add(d_i >= 2).only_enforce_if(has_two)
            model.add(d_i <= 1).only_enforce_if(has_two.Not())
            info["n_aux"] += 1

            p_i = int(phi_target[i])
            # When d_i >= 2: 2*D*t_i == p_i * dd_i
            model.add(2 * D * t_i == p_i * dd_i).only_enforce_if(has_two)
            # When d_i <= 1: clustering convention C_i = 0 => target must be 0.
            # (If p_i != 0 the model is simply infeasible in this branch, which
            #  is correct: no graph realizes a nonzero clustering with deg<2.)
            model.add(p_i == 0).only_enforce_if(has_two.Not())

        return info

    # ── Tier 1 (polynomial): sqdeg, wedge, fourclique ─────────────────────
    if phi_type == "sqdeg":
        # Phi(v) = d_v^2.  d_i integer var; sq_i = d_i*d_i via multiplication.
        for i in range(n):
            d_i = model.new_int_var(0, n - 1, f"deg_{i}")
            model.add(d_i == sum(ev(i, j) for j in range(n) if j != i))
            sq_i = model.new_int_var(0, (n - 1) ** 2, f"sqdeg_{i}")
            model.add_multiplication_equality(sq_i, [d_i, d_i])
            model.add(sq_i == int(phi_target[i]))
            info["n_aux"] += 2
        return info

    if phi_type == "wedge":
        # Phi(v) = d_v (d_v - 1) = 2*C(d_v,2).  d_i integer; dd_i = d_i*(d_i-1).
        for i in range(n):
            d_i = model.new_int_var(0, n - 1, f"deg_{i}")
            model.add(d_i == sum(ev(i, j) for j in range(n) if j != i))
            dm1 = model.new_int_var(-1, n - 1, f"degm1_{i}")
            model.add(dm1 == d_i - 1)
            dd_i = model.new_int_var(0, (n - 1) * (n - 1), f"wedge_{i}")
            model.add_multiplication_equality(dd_i, [d_i, dm1])
            model.add(dd_i == int(phi_target[i]))
            info["n_aux"] += 3
        return info

    if phi_type == "fourclique":
        # Phi(v) = # of 4-cliques {v,j,k,l}.  One Boolean per 4-subset =
        # AND of its 6 edges (linearised).  O(n^4) auxiliaries; capped below.
        n_cap = aux_cfg.get("fourclique_ncap", 9)
        if n > n_cap:
            raise ValueError(
                f"fourclique n={n} exceeds configured cap n_cap={n_cap}; "
                f"C(n,4) auxiliaries blow up (O(n^4)). Raise aux_cfg["
                f"'fourclique_ncap'] to override.")
        z = {}
        for quad in itertools.combinations(range(n), 4):
            zt = model.new_bool_var("z4_" + "_".join(map(str, quad)))
            edges6 = [ev(a, b) for a, b in itertools.combinations(quad, 2)]
            for e in edges6:
                model.add(zt <= e)
            model.add(zt >= sum(edges6) - 5)   # AND of 6 booleans
            z[quad] = zt
            info["n_aux"] += 1
        # C1: sum over 4-subsets containing i of z == phi_target[i]
        for i in range(n):
            terms = [z[q] for q in z if i in q]
            model.add(sum(terms) == int(phi_target[i]))
        info["fourclique_z"] = z
        return info

    # ── Tier 2 (rational): avgneighdeg ────────────────────────────────────
    if phi_type == "avgneighdeg":
        # ANeighDeg(v) = (Sum_j M_vj d_j) / d_v; convention 0 when d_v = 0.
        # Cleared denominator (d_v >= 1): D * num_i == p_i * d_i, where
        # num_i = Sum_j M_ij d_j (Boolean x integer products), D = scale.
        D = aux_cfg["clustering_scale"]
        # per-vertex degree int vars (shared by numerator products)
        d = {}
        for i in range(n):
            di = model.new_int_var(0, n - 1, f"deg_{i}")
            model.add(di == sum(ev(i, j) for j in range(n) if j != i))
            d[i] = di
            info["n_aux"] += 1
        for i in range(n):
            # num_i = Sum_{j != i} M_ij * d_j  (product of a Boolean and an int)
            prod_terms = []
            for j in range(n):
                if j == i:
                    continue
                pij = model.new_int_var(0, n - 1, f"andeg_{i}_{j}")
                # pij = M_ij * d_j   (Boolean * integer, exact)
                model.add_multiplication_equality(pij, [ev(i, j), d[j]])
                prod_terms.append(pij)
                info["n_aux"] += 1
            num_i = model.new_int_var(0, (n - 1) * (n - 1), f"annum_{i}")
            model.add(num_i == sum(prod_terms))
            info["n_aux"] += 1

            # reify d_i >= 1
            has_one = model.new_bool_var(f"deg_ge1_{i}")
            model.add(d[i] >= 1).only_enforce_if(has_one)
            model.add(d[i] == 0).only_enforce_if(has_one.Not())
            info["n_aux"] += 1

            p_i = int(phi_target[i])
            # d_i >= 1:  D * num_i == p_i * d_i
            model.add(D * num_i == p_i * d[i]).only_enforce_if(has_one)
            # d_i == 0:  convention ANeighDeg = 0 => target must be 0.
            model.add(p_i == 0).only_enforce_if(has_one.Not())
        return info

    # ── Tier 3 (fixed-point certifiable): closeness / katz / pagerank ─────
    if phi_type == "closeness":
        _phi_alignment_closeness(model, ev, n, phi_target, aux_cfg, info)
        return info
    if phi_type == "katz":
        _phi_alignment_katz(model, ev, n, phi_target, aux_cfg, info)
        return info
    if phi_type == "pagerank":
        _phi_alignment_pagerank(model, ev, n, phi_target, aux_cfg, info)
        return info

    # ── TRUNCATED / APPROXIMATE schemes (finite-K, encodable) ─────────────
    if phi_type in _TRUNCATED:
        K = aux_cfg["K"]
        D = aux_cfg["clustering_scale"]      # the integer scale (may be huge)
        if phi_type == "pagerank_K":
            _phi_alignment_truncated_pagerank(model, ev, n, phi_target, K, D, info)
        else:
            _phi_alignment_truncated(model, ev, n, phi_target, phi_type, K, D, info)
        return info

    _refuse(phi_type)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 C1 builders: fixed-point CERTIFICATES (closeness / katz / pagerank)
# ─────────────────────────────────────────────────────────────────────────────

def _phi_alignment_closeness(model, ev, n, phi_target, aux_cfg, info):
    """C1 for closeness on a CONNECTED graph.

    Encodes the shortest-path distance delta[i][j] via a fixed-point CERTIFICATE
    (Bellman-Ford local optimality), then pins the rational closeness
    (n-1)/S_i == p_i/D with a cleared-denominator constraint.

    Certificate constraints (per source i):
      (dom)     delta[i][i] = 0 ;  delta[i][j] in [1, n-1] for j != i.
      (smooth)  for every pair j<k with M_jk = 1:  |delta[i][j]-delta[i][k]| <= 1.
                (an edge cannot change the distance-to-i by more than one hop.)
      (witness) for every j != i there EXISTS a neighbour p of j (M_jp=1) with
                delta[i][p] == delta[i][j]-1.  Encoded with a FULLY-REIFIED
                Boolean valid[i][j][p] <=> (M_jp=1 AND delta[i][p]=delta[i][j]-1)
                and sum_p valid[i][j][p] >= 1.
    Because valid[.] is *functionally determined* by (M, delta) -- not a free
    selector -- these three families force delta[i][j] to the TRUE SP distance and
    admit exactly one delta per graph (verified: no wrong-distance solution and no
    auxiliary multiplicity).

    Closeness pin (cleared denominator, S_i = sum_{j!=i} delta[i][j]):
        D*(n-1) == p_i * S_i ,     p_i = phi_target[i] = D * closeness_i.
    D = aux_cfg['clustering_scale'] must be the lcm{S_i} returned by scale_for.
    """
    D = aux_cfg["clustering_scale"]
    # Overflow guard: the cleared closeness pin is  D*(n-1) == p_i * S_i  with
    # p_i = D*closeness_i (up to D) and S_i <= (n-1)^2.  D = lcm{S_i} is a
    # distance-sum LCM that also blows up (like the katz/pagerank determinants),
    # so guard both products against the CP-SAT safe domain and REPORT overflow.
    lhs = D * (n - 1)
    max_pi = D  # closeness_i <= 1 so p_i = D*closeness_i <= D
    rhs = max_pi * (n - 1) * (n - 1)
    if lhs >= FIXEDPOINT_SAFE_BOUND or rhs >= FIXEDPOINT_SAFE_BOUND:
        raise ScaleOverflow("closeness", n, D, max(lhs, rhs).bit_length())

    delta = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                delta[i, j] = model.new_int_var(0, 0, f"clo_d_{i}_{i}")
            else:
                delta[i, j] = model.new_int_var(1, n - 1, f"clo_d_{i}_{j}")
            info["n_aux"] += 1

    for i in range(n):
        # (smooth) adjacency-smoothness on every unordered pair (edge-reified)
        for j in range(n):
            for k in range(j + 1, n):
                e = ev(j, k)
                model.add(delta[i, j] - delta[i, k] <= 1).only_enforce_if(e)
                model.add(delta[i, k] - delta[i, j] <= 1).only_enforce_if(e)
        # (witness) predecessor certificate for every j != i
        for j in range(n):
            if j == i:
                continue
            valids = []
            for p in range(n):
                if p == j:
                    continue
                # eqp <=> delta[i][p] == delta[i][j]-1   (full reification)
                eqp = model.new_bool_var(f"clo_eq_{i}_{j}_{p}")
                model.add(delta[i, p] == delta[i, j] - 1).only_enforce_if(eqp)
                model.add(delta[i, p] != delta[i, j] - 1).only_enforce_if(eqp.Not())
                # vp <=> (M_jp = 1) AND eqp   (full reification -> functional)
                vp = model.new_bool_var(f"clo_v_{i}_{j}_{p}")
                model.add_bool_and([ev(j, p), eqp]).only_enforce_if(vp)
                model.add_bool_or([ev(j, p).Not(), eqp.Not()]).only_enforce_if(vp.Not())
                valids.append(vp)
                info["n_aux"] += 2
            # at least one predecessor witnesses delta[i][j] = delta[i][p]+1
            model.add(sum(valids) >= 1)

    # closeness pin: D*(n-1) == p_i * S_i
    for i in range(n):
        S_i = model.new_int_var(0, (n - 1) * (n - 1), f"clo_S_{i}")
        model.add(S_i == sum(delta[i, j] for j in range(n) if j != i))
        info["n_aux"] += 1
        model.add(D * (n - 1) == int(phi_target[i]) * S_i)


def _phi_alignment_katz(model, ev, n, phi_target, aux_cfg, info):
    """C1 for Katz centrality: pin X_i = D * x_i as the UNIQUE solution of the
    integer-cleared linear fixed point.  No matrix inverse.

    x = 1 + alpha*M*x  with alpha = num/den (=1/10).  Scale X_i = D*x_i, clear:
        den * X_i == den * D + num * sum_{j != i} M_ij * X_j.
    M_ij * X_j is a Boolean*integer product (exact).  The linear system has a
    unique integer solution equal to D*x_i; D = aux_cfg['clustering_scale'] is the
    det-denominator returned by scale_for/katz_scale_for.  Raises ScaleOverflow if
    the cleared RHS would exceed the CP-SAT safe domain (encodable != tractable).
    """
    D = aux_cfg["clustering_scale"]
    an, ad = KATZ_ALPHA_FP_NUM, KATZ_ALPHA_FP_DEN
    # X_i = D*x_i is PINNED to phi_target[i], so the exact domain bound is the max
    # target (NOT D*n: Katz x_i is not bounded by n and can be far larger when
    # alpha*lambda_max is close to 1).  Every product mx = M_ij*X_j <= max target.
    Xb = max((int(p) for p in phi_target), default=0)
    Xb = max(Xb, D)                      # keep >= D so the +D term always fits
    rhs_bound = ad * Xb + an * (n - 1) * Xb
    # CP-SAT additionally requires the SUM of all variable domains to fit in int64
    # (overflow guard in its validator).  This encoding creates ~n^2 product vars
    # each of domain Xb, so the aggregate ~ n^2 * Xb must also stay bounded.
    domain_sum = (n * n + 2 * n) * Xb
    if (Xb >= FIXEDPOINT_SAFE_BOUND or rhs_bound >= FIXEDPOINT_SAFE_BOUND
            or domain_sum >= 2 ** 62):
        raise ScaleOverflow("katz", n, D, max(D, rhs_bound, domain_sum).bit_length())

    X = [model.new_int_var(0, Xb, f"katz_X_{i}") for i in range(n)]
    info["n_aux"] += n
    for i in range(n):
        prod_terms = []
        for j in range(n):
            if j == i:
                continue
            mx = model.new_int_var(0, Xb, f"katz_mx_{i}_{j}")
            model.add_multiplication_equality(mx, [ev(i, j), X[j]])
            prod_terms.append(mx)
            info["n_aux"] += 1
        # den * X_i == den * D + num * sum_j M_ij X_j
        model.add(ad * X[i] == ad * D + an * sum(prod_terms))
        # pin to the exact target (the unique solution); makes the certificate
        # explicit and lets an unrealizable target surface as honest INFEASIBLE.
        model.add(X[i] == int(phi_target[i]))


def _phi_alignment_pagerank(model, ev, n, phi_target, aux_cfg, info):
    """C1 for PageRank: pin X_i = D * x_i as the UNIQUE solution of the stochastic
    linear fixed point with per-vertex degree division.

    x_i = (1-alpha)/n + alpha * sum_{j: j~i} x_j / d_j ,  alpha = an/ad (=17/20).
    Introduce quotient vars Q_j = X_j / d_j (gated by d_j >= 1; isolated vertices
    carry only teleport).  Cleared over the common denominator ad*n:
        ad*n * X_i == (ad-an)*D + an*n * sum_{j != i} M_ij * Q_j .
    D = aux_cfg['clustering_scale'] must be lcm(denoms(x)) * lcm(degrees) (from
    scale_for/pagerank_scale_for) so BOTH X_j and X_j/d_j are integral.  Raises
    ScaleOverflow if the cleared RHS overflows the CP-SAT safe domain.
    """
    D = aux_cfg["clustering_scale"]
    an, ad = PR_ALPHA_FP_NUM, PR_ALPHA_FP_DEN
    Xb = D                                   # x_i in (0,1) so X_i <= D
    prod_b = Xb * max(n - 1, 1)              # Q_j * d_j <= X_j <= D; safe with (n-1)
    rhs_bound = (ad - an) * D + an * n * (n - 1) * Xb
    lhs_bound = ad * n * Xb
    # aggregate domain-sum guard (CP-SAT validator): ~n^2 product vars of size Xb,
    # plus prod vars of size prod_b.
    domain_sum = (n * n + 3 * n) * Xb + n * prod_b
    if (Xb >= FIXEDPOINT_SAFE_BOUND or prod_b >= FIXEDPOINT_SAFE_BOUND
            or rhs_bound >= FIXEDPOINT_SAFE_BOUND or lhs_bound >= FIXEDPOINT_SAFE_BOUND
            or domain_sum >= 2 ** 62):
        raise ScaleOverflow("pagerank", n, D,
                            max(D, rhs_bound, lhs_bound, prod_b, domain_sum).bit_length())

    X = [model.new_int_var(0, Xb, f"pr_X_{i}") for i in range(n)]
    info["n_aux"] += n
    d = [model.new_int_var(0, n - 1, f"pr_deg_{i}") for i in range(n)]
    for i in range(n):
        model.add(d[i] == sum(ev(i, j) for j in range(n) if j != i))
        info["n_aux"] += 1

    Q = [model.new_int_var(0, Xb, f"pr_Q_{j}") for j in range(n)]
    for j in range(n):
        info["n_aux"] += 1
        has1 = model.new_bool_var(f"pr_h_{j}")
        model.add(d[j] >= 1).only_enforce_if(has1)
        model.add(d[j] == 0).only_enforce_if(has1.Not())
        prod = model.new_int_var(0, prod_b, f"pr_qd_{j}")
        model.add_multiplication_equality(prod, [Q[j], d[j]])
        model.add(prod == X[j]).only_enforce_if(has1)     # Q_j * d_j == X_j
        model.add(Q[j] == 0).only_enforce_if(has1.Not())  # isolated: no in-mass
        info["n_aux"] += 2

    for i in range(n):
        terms = []
        for j in range(n):
            if j == i:
                continue
            mq = model.new_int_var(0, Xb, f"pr_mq_{i}_{j}")
            model.add_multiplication_equality(mq, [ev(i, j), Q[j]])
            terms.append(mq)
            info["n_aux"] += 1
        model.add(ad * n * X[i] == (ad - an) * D + an * n * sum(terms))
        model.add(X[i] == int(phi_target[i]))


def _phi_alignment_truncated(model, ev, n, phi_target, phi_type, K, D, info):
    """C1 for the finite-K surrogates.  Builds integer matrix-power vectors
    w^(k) = M w^(k-1) (eigvec/katz) or the degree-cleared random-walk recursion
    (rw/pagerank), then pins the K-truncated combination to phi_target.

    Bound note: entries of M^K 1 are <= n^K; scaled by D they can be very large.
    CP-SAT integer variables are 64-bit, so this is where huge D / large K make
    the model ill-posed or intractable (reported by the harness).
    """
    BOUND = 10 ** 15   # keep var domains inside CP-SAT's safe 64-bit range

    # degree int vars (needed by rw / pagerank denominators)
    deg = {}
    if phi_type in ("rw_K", "pagerank_K"):
        for i in range(n):
            di = model.new_int_var(0, n - 1, f"trdeg_{i}")
            model.add(di == sum(ev(i, j) for j in range(n) if j != i))
            deg[i] = di
            info["n_aux"] += 1

    if phi_type in ("eigvec_K", "katz_K"):
        # w^(0) = 1 ; w^(k)_i = sum_j M_ij w^(k-1)_j   (Boolean * integer)
        w = {i: 1 for i in range(n)}   # w^0 as python ints (constant vector)
        # first power produces variables
        powers = []   # powers[k-1][i] = w^(k)_i  as CP int var
        cur = {i: None for i in range(n)}
        prev_is_const = True
        prev = w
        for k in range(1, K + 1):
            new = {}
            for i in range(n):
                terms = []
                for j in range(n):
                    if j == i:
                        continue
                    if prev_is_const:
                        # M_ij * const  -> linear term  const * M_ij
                        terms.append(prev[j] * ev(i, j))
                    else:
                        pij = model.new_int_var(0, BOUND, f"mp_{phi_type}_{k}_{i}_{j}")
                        model.add_multiplication_equality(pij, [ev(i, j), prev[j]])
                        info["n_aux"] += 1
                        terms.append(pij)
                wi = model.new_int_var(0, BOUND, f"w_{phi_type}_{k}_{i}")
                model.add(wi == sum(terms))
                info["n_aux"] += 1
                new[i] = wi
            powers.append(new)
            prev = new
            prev_is_const = False

        if phi_type == "eigvec_K":
            # pin D * (M^K 1)_i ; D == 1 for eigvec
            for i in range(n):
                model.add(D * powers[K - 1][i] == int(phi_target[i]))
        else:  # katz_K : sum_{k=1}^K alpha^k M^k 1, scaled by D = 10^K
            # coefficient of M^k term after scaling: D * alpha^k = 10^(K-k)
            for i in range(n):
                expr = sum((KATZ_ALPHA_DEN ** (K - k)) * powers[k - 1][i]
                           for k in range(1, K + 1))
                model.add(expr == int(phi_target[i]))
        return

    # rw_K / pagerank_K : degree-cleared random walk.
    # Maintain an integer vector u^(k) that equals (scale_k) * (M^T D^-1)^k 1,
    # where scale_k clears k degree-divisions.  We use the per-vertex identity
    #   u^(k)_i * 1  =  sum_j M_ji * (u^(k-1)_j / d_j)   scaled by an extra deg.
    # Exact clearing with variable degrees requires, at each step, an integer
    # q^(k-1)_j = u^(k-1)_j / d_j enforced by  q * d_j == u  (multiplication).
    # Since d_j may be 0 (isolated), we gate those out (they carry 0 mass).
    # This is EXACT but the values grow ~ (max_deg)^k * n^k, so D and the domains
    # blow up fast -- the intended demonstration.
    # u^(0) = D0 * 1, with D0 chosen so all subsequent divisions stay integral;
    # here we lean on the caller-supplied D (= L^K [* 20^K]) as the global scale
    # and enforce the divided recursion with exact multiplication constraints.
    # Represent the walk with a running integer vector and per-step quotient vars.
    u = {i: model.new_int_var(0, BOUND, f"u0_{i}") for i in range(n)}
    for i in range(n):
        model.add(u[i] == D)          # u^0 = D * 1
        info["n_aux"] += 1
    for k in range(1, K + 1):
        # quotient q_j = u^(k-1)_j / d_j  (integer, gated by d_j>=1)
        q = {}
        for j in range(n):
            qj = model.new_int_var(0, BOUND, f"q_{phi_type}_{k}_{j}")
            has1 = model.new_bool_var(f"trge1_{phi_type}_{k}_{j}")
            model.add(deg[j] >= 1).only_enforce_if(has1)
            model.add(deg[j] == 0).only_enforce_if(has1.Not())
            # q_j * d_j == u_j   when d_j>=1 ; q_j == 0 when d_j==0
            prod = model.new_int_var(0, BOUND, f"qd_{phi_type}_{k}_{j}")
            model.add_multiplication_equality(prod, [qj, deg[j]])
            model.add(prod == u[j]).only_enforce_if(has1)
            model.add(qj == 0).only_enforce_if(has1.Not())
            info["n_aux"] += 3
            q[j] = qj
        # u^(k)_i = sum_j M_ji q_j
        newu = {}
        for i in range(n):
            terms = []
            for j in range(n):
                if j == i:
                    continue
                mij_qj = model.new_int_var(0, BOUND, f"mq_{phi_type}_{k}_{i}_{j}")
                model.add_multiplication_equality(mij_qj, [ev(i, j), q[j]])
                info["n_aux"] += 1
                terms.append(mij_qj)
            ui = model.new_int_var(0, BOUND, f"u_{phi_type}_{k}_{i}")
            model.add(ui == sum(terms))
            info["n_aux"] += 1
            newu[i] = ui
        u = newu

    if phi_type == "rw_K":
        # target = D * (M^T D^-1)^K 1 == u^(K)  (u carries exactly D * walk)
        for i in range(n):
            model.add(u[i] == int(phi_target[i]))
    else:  # pagerank_K: (1-alpha)/n * sum_{k=0}^K alpha^k walk_k, scaled by D.
        # We only pinned the K-th walk vector u^(K); pagerank needs the full sum.
        # Rebuild the accumulated sum from the intermediate u^(k). To keep the
        # encoding faithful we instead pin the K-truncated pagerank via the
        # SAME per-step vectors: accumulate them.  (Handled in the sum below.)
        raise NotImplementedError(
            "pagerank_K accumulation handled in _phi_alignment_truncated_pagerank")


def _phi_alignment_truncated_pagerank(model, ev, n, phi_target, K, D, info):
    """Builder for pagerank_K: accumulate ALL walk vectors u^(0..K) with integer
    damping weights.  ``walks[k][i]`` is enforced to equal Lpow * walk_k[i]
    (Lpow = L^K), matching truncated_phi_values('pagerank_K') exactly, so the
    source graph is always a witness and any per-vertex mismatch surfaces as an
    honest INFEASIBLE rather than a silent wrong answer."""
    BOUND = 10 ** 15
    from_num = PR_ALPHA_DEN - PR_ALPHA_NUM
    Lpow = D // (n * (PR_ALPHA_DEN ** (K + 1)))  # = L^K

    deg = {}
    for i in range(n):
        di = model.new_int_var(0, n - 1, f"prdeg_{i}")
        model.add(di == sum(ev(i, j) for j in range(n) if j != i))
        deg[i] = di
        info["n_aux"] += 1

    # u^(0) = Lpow * 1
    u = {i: model.new_int_var(0, BOUND, f"pru0_{i}") for i in range(n)}
    for i in range(n):
        model.add(u[i] == Lpow)
        info["n_aux"] += 1
    walks = [dict(u)]
    for k in range(1, K + 1):
        q = {}
        for j in range(n):
            qj = model.new_int_var(0, BOUND, f"prq_{k}_{j}")
            has1 = model.new_bool_var(f"prge1_{k}_{j}")
            model.add(deg[j] >= 1).only_enforce_if(has1)
            model.add(deg[j] == 0).only_enforce_if(has1.Not())
            prod = model.new_int_var(0, BOUND, f"prqd_{k}_{j}")
            model.add_multiplication_equality(prod, [qj, deg[j]])
            model.add(prod == u[j]).only_enforce_if(has1)
            model.add(qj == 0).only_enforce_if(has1.Not())
            info["n_aux"] += 3
            q[j] = qj
        newu = {}
        for i in range(n):
            terms = []
            for j in range(n):
                if j == i:
                    continue
                mij_qj = model.new_int_var(0, BOUND, f"prmq_{k}_{i}_{j}")
                model.add_multiplication_equality(mij_qj, [ev(i, j), q[j]])
                info["n_aux"] += 1
                terms.append(mij_qj)
            ui = model.new_int_var(0, BOUND, f"pru_{k}_{i}")
            model.add(ui == sum(terms))
            info["n_aux"] += 1
            newu[i] = ui
        u = newu
        walks.append(dict(u))

    for i in range(n):
        expr = sum(from_num * (PR_ALPHA_NUM ** k) * (PR_ALPHA_DEN ** (K - k))
                   * walks[k][i]
                   for k in range(0, K + 1))
        model.add(expr == int(phi_target[i]))


def _build_triangle_y(model, ev, n, info):
    """Create ONE Boolean y_{ijk} per unordered triple with AND-linearization
    y = M_ij & M_ik & M_jk.  Returns dict triple(tuple sorted) -> BoolVar.
    O(n^3) auxiliaries, one per triple, reused across all 3 incident vertices.
    """
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                t = (i, j, k)
                yt = model.new_bool_var(f"y_{i}_{j}_{k}")
                a, b, c = ev(i, j), ev(i, k), ev(j, k)
                model.add(yt <= a)
                model.add(yt <= b)
                model.add(yt <= c)
                model.add(yt >= a + b + c - 2)
                y[t] = yt
                info["n_aux"] += 1
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Core builder (C2..C9 + connectivity + ALIVE, filtration-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(p0_target, p1_target, stage, phi_target, phi_type, aux_cfg,
                 verbose=False):
    """Build the CP-SAT model.  ``stage`` plays the role of the old ``vd`` array
    for C2..C9 (active-set thresholds and stage-histogram aggregates); C1 is
    delegated to ``phi_alignment`` using ``phi_target``.

    Returns (model, M, n, aux_info).
    """
    # ── 1. Stage statistics (identical arithmetic to the degree solver, but the
    #       per-vertex "birth time" is now ``stage`` rather than degree) ───────
    p0b = defaultdict(int)
    p0d = defaultdict(int)
    p0bd = defaultdict(int)
    p1b = defaultdict(int)

    for b, d in p0_target:
        p0b[b] += 1
        if d is not None:
            p0d[d] += 1
            p0bd[(b, d)] += 1
    for b, _ in p1_target:
        p1b[b] += 1

    ms = max(max(p0b, default=0), max(p0d, default=0), max(p1b, default=0))

    # Pair (stage, phi) and sort by stage so vertex indices line up with the
    # histogram-derived active sets exactly as the degree solver builds vd by
    # iterating birth times in increasing order.  ``vd`` is the stage array.
    paired = sorted(zip(stage, phi_target), key=lambda x: x[0])
    vd = [s for s, _ in paired]
    phi_ordered = [p for _, p in paired]
    n = len(vd)

    # Sanity: histogram of ``vd`` (stage) must equal the birth histogram of P0.
    vd_hist = defaultdict(int)
    for s in vd:
        vd_hist[s] += 1
    for k in range(ms + 1):
        assert vd_hist.get(k, 0) == p0b.get(k, 0), (
            f"stage histogram {dict(vd_hist)} inconsistent with P0 births {dict(p0b)}")

    na = {k: sum(1 for d in vd if d <= k) for k in range(ms + 1)}
    cc = {}; br = dr = 0
    for k in range(ms + 1):
        br += p0b[k]; dr += p0d[k]; cc[k] = br - dr
    cy = {}; cr = 0
    for k in range(ms + 1):
        cr += p1b[k]; cy[k] = cr
    ns = max(cc.values()) if cc else 1

    bt = sorted(p0b.keys())
    alive = {}
    for k in range(ms + 1):
        alive[k] = {}
        for b in bt:
            alive[k][b] = p0b[b] - sum(p0bd.get((b, d_), 0) for d_ in range(k + 1))

    unchanged = {k: (p0b[k] == 0 and p0d[k] == 0 and p1b[k] == 0)
                 for k in range(ms + 1)}
    unchanged[0] = False
    need_conn = {k: (p0d[k] > 0 or p1b[k] > 0) for k in range(ms + 1)}

    if verbose:
        print(f"n={n}, max_stage={ms}, comp_slots={ns}, phi_type={phi_type}")
        print(f"vertex_stage : {vd}")
        print(f"phi_target   : {phi_ordered}")

    # ── 2. Model + variables ──────────────────────────────────────────────
    model = cp_model.CpModel()
    M = {(i, j): model.new_bool_var(f"M_{i}_{j}")
         for i in range(n) for j in range(i + 1, n)}

    def ev(u, v):
        return M[min(u, v), max(u, v)]

    comp = {}
    for k in range(ms + 1):
        if unchanged[k]:
            comp[k] = comp[k - 1]; continue
        comp[k] = {c: {v: model.new_bool_var(f"cp_{k}_{c}_{v}")
                       for v in range(n) if vd[v] <= k}
                   for c in range(ns)}

    cu = {}
    for k in range(ms + 1):
        if unchanged[k]:
            cu[k] = cu[k - 1]; continue
        av = [v for v in range(n) if vd[v] <= k]; cu[k] = {}
        for c in range(ns):
            u = model.new_bool_var(f"used_{k}_{c}"); cu[k][c] = u
            if av:
                model.add_max_equality(u, [comp[k][c][v] for v in av])
            else:
                model.add(u == 0)

    A = {}
    for k in range(1, ms + 1):
        if unchanged[k] or cc[k] == 0 or cc[k - 1] == 0:
            continue
        A[k] = {(c, d): model.new_bool_var(f"A_{k}_{c}_{d}")
                for c in range(cc[k]) for d in range(cc[k - 1])}

    # ── C1: filtration-specific alignment (the ONLY plug point) ───────────
    aux_info = phi_alignment(model, M, n, phi_ordered, phi_type, vd, aux_cfg)

    # ── C2: Partition ────────────────────────────────────────────────────
    for k in range(ms + 1):
        if unchanged[k]:
            continue
        for v in range(n):
            if vd[v] <= k:
                model.add_exactly_one(comp[k][c][v] for c in range(ns))

    # ── C3: Symmetry breaking ────────────────────────────────────────────
    for k in range(ms + 1):
        if unchanged[k]:
            continue
        for c in range(1, ns):
            model.add(cu[k][c] <= cu[k][c - 1])

    # ── C8: Component count ───────────────────────────────────────────────
    for k in range(ms + 1):
        if unchanged[k]:
            continue
        model.add(sum(cu[k][c] for c in range(ns)) == cc[k])

    # ── C4: Edge => same slot ─────────────────────────────────────────────
    for k in range(ms + 1):
        if unchanged[k]:
            continue
        for u in range(n):
            for v in range(u + 1, n):
                if vd[u] <= k and vd[v] <= k:
                    e = ev(u, v)
                    for c in range(cc[k]):
                        model.add(e + comp[k][c][u] - comp[k][c][v] <= 1)
                        model.add(e + comp[k][c][v] - comp[k][c][u] <= 1)

    # ── C4b: Spanning-tree connectivity ───────────────────────────────────
    BIG = n
    for k in range(ms + 1):
        if unchanged[k] or not need_conn[k]:
            continue
        av = [v for v in range(n) if vd[v] <= k]
        if len(av) <= 1:
            continue
        n_used = cc[k]
        for c in range(n_used):
            is_root = {}
            for v in av:
                if v == 0:
                    is_root[v] = comp[k][c][v]
                else:
                    ir = model.new_bool_var(f"ir_{k}_{c}_{v}")
                    preds = [comp[k][c][u].Not() for u in av if u < v]
                    model.add_bool_and([comp[k][c][v]] + preds).only_enforce_if(ir)
                    model.add_bool_or([comp[k][c][v].Not()] +
                                      [comp[k][c][u] for u in av if u < v]).only_enforce_if(ir.Not())
                    is_root[v] = ir

            L = {v: model.new_int_var(0, n, f"L_{k}_{c}_{v}") for v in av}
            par = {(i, j): model.new_bool_var(f"par_{k}_{c}_{i}_{j}")
                   for i in av for j in av if i != j}

            for i in av:
                for j in av:
                    if i == j:
                        continue
                    model.add(par[i, j] <= ev(i, j))
                    model.add(par[i, j] <= comp[k][c][i])
                    model.add(par[i, j] <= comp[k][c][j])
                    model.add(L[j] >= L[i] + 1 - BIG * (1 - par[i, j]))
                    model.add(L[j] <= L[i] + 1 + BIG * (1 - par[i, j]))

            for v in av:
                in_par = sum(par[u, v] for u in av if u != v)
                model.add(in_par == comp[k][c][v] - is_root[v])
                model.add(L[v] <= BIG * (1 - is_root[v]))
                model.add(L[v] <= BIG * comp[k][c][v])

    # ── C6: Ancestry forward ─────────────────────────────────────────────
    for k in range(1, ms + 1):
        if unchanged[k] or k not in A:
            continue
        for c in range(cc[k]):
            for d in range(cc[k - 1]):
                a = A[k][c, d]; ud = cu[k - 1][d]
                for v in range(n):
                    if vd[v] <= k - 1:
                        model.add(comp[k - 1][d][v] <= comp[k][c][v] + (1 - a) + (1 - ud))

    # ── C6c: A reflects transitions ───────────────────────────────────────
    for k in range(1, ms + 1):
        if unchanged[k] or k not in A:
            continue
        for v in range(n):
            if vd[v] <= k - 1:
                for c in range(cc[k]):
                    for d in range(cc[k - 1]):
                        model.add(comp[k - 1][d][v] + comp[k][c][v] - 1 <= A[k][c, d])

    # ── C7: A column-stochastic ───────────────────────────────────────────
    for k in range(1, ms + 1):
        if unchanged[k] or k not in A:
            continue
        for d in range(cc[k - 1]):
            model.add(sum(A[k][c, d] for c in range(cc[k])) == cu[k - 1][d])

    # ── C9: Cycle count ───────────────────────────────────────────────────
    for k in range(ms + 1):
        if unchanged[k]:
            continue
        ae = [ev(u, v) for u in range(n) for v in range(u + 1, n)
              if vd[u] <= k and vd[v] <= k]
        model.add(sum(ae) - na[k] + sum(cu[k][c] for c in range(ns)) == cy[k])

    # ── C_ALIVE: per-birth-time component survival ────────────────────────
    for k in range(ms + 1):
        if unchanged[k]:
            continue
        av = [v for v in range(n) if vd[v] <= k]
        if not av:
            continue
        for b in bt:
            vob = [v for v in av if vd[v] == b]
            vlt = [v for v in av if vd[v] < b]
            if not vob:
                continue
            elder_vars = []
            for c in range(cc[k]):
                has_b = bool_or_vars(model, [comp[k][c][v] for v in vob],
                                     f"hasb_{k}_{c}_{b}")
                if vlt:
                    no_lt = model.new_bool_var(f"nolt_{k}_{c}_{b}")
                    model.add_bool_and([comp[k][c][v].Not() for v in vlt]).only_enforce_if(no_lt)
                    model.add_bool_or([comp[k][c][v] for v in vlt]).only_enforce_if(no_lt.Not())
                    tmp = bool_and_var(model, has_b, no_lt, f"tmp_{k}_{c}_{b}")
                    elder = bool_and_var(model, cu[k][c], tmp, f"elder_{k}_{c}_{b}")
                else:
                    elder = bool_and_var(model, cu[k][c], has_b, f"elder_{k}_{c}_{b}")
                elder_vars.append(elder)
            model.add(sum(elder_vars) == alive[k][b])

    return model, M, n, aux_info


# ─────────────────────────────────────────────────────────────────────────────
# Public solve entry points
# ─────────────────────────────────────────────────────────────────────────────

def solve_phi(p0_target, p1_target, stage, phi_target, phi_type="degree",
              time_limit=60.0, verbose=True, cp_sat_workers=8,
              clustering_scale=1, fourclique_ncap=9, K=None,
              return_status=False):
    """Find a graph H whose Phi-filtration PH matches (p0_target, p1_target),
    where ``stage``/``phi_target`` are the per-vertex stage / raw-Phi arrays.

    For rational filtrations (phi_type in {'clustering','avgneighdeg'}),
    ``clustering_scale`` MUST be the SAME exact integer D used to scale
    ``phi_target`` (see ``scale_for`` / ``target_pd_from_graph``); otherwise C1
    clears the wrong denominator.  Ignored for the integer-valued filtrations.
    ``fourclique_ncap`` caps n for the O(n^4) fourclique encoding.  ``K`` is the
    truncation order for the truncated schemes.  If ``return_status`` also
    returns (H, status_name, solve_time, n_vars, n_aux).

    Returns an nx.Graph (n vertices, integer labels 0..n-1) or None.
    """
    aux_cfg = {"clustering_scale": clustering_scale,
               "fourclique_ncap": fourclique_ncap, "K": K}
    model, M, n, aux_info = _build_model(
        p0_target, p1_target, stage, phi_target, phi_type, aux_cfg, verbose=verbose)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False
    solver.parameters.num_search_workers = cp_sat_workers

    n_vars = len(model.proto.variables)
    if verbose:
        print(f"Model: {n_vars} variables, "
              f"{len(model.proto.constraints)} constraints, "
              f"aux(C1)={aux_info['n_aux']}")

    t0 = time.time()
    status = solver.solve(model)
    elapsed = time.time() - t0
    sname = solver.status_name(status)

    # Guard against the CP-SAT presolve coefficient-overflow that wrongly reports
    # INFEASIBLE on large-D fixed-point encodings: re-solve once with presolve OFF.
    if status == cp_model.INFEASIBLE and phi_type in _FIXEDPOINT:
        solver2 = cp_model.CpSolver()
        solver2.parameters.max_time_in_seconds = max(time_limit - elapsed, 5.0)
        solver2.parameters.log_search_progress = False
        solver2.parameters.num_search_workers = cp_sat_workers
        solver2.parameters.cp_model_presolve = False
        t1 = time.time()
        status = solver2.solve(model)
        elapsed += time.time() - t1
        sname = solver2.status_name(status)
        solver = solver2

    if verbose:
        print(f"Status: {sname}  ({elapsed:.2f}s)")

    H = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                adj[i, j] = adj[j, i] = solver.value(M[i, j])
        H = nx.from_numpy_array(adj)

    if return_status:
        return H, sname, elapsed, n_vars, aux_info["n_aux"]
    return H


def solve_from_graph(G: nx.Graph, phi_type="degree", time_limit=60.0,
                     verbose=True, cp_sat_workers=8, clustering_scale=None,
                     cross_check=True, fourclique_ncap=9):
    """Convenience: compute the target PD of ``G`` under ``phi_type`` and solve.

    ``clustering_scale=None`` resolves the exact per-graph scale (recommended);
    the SAME resolved value is threaded into C1 so target and constraint agree.

    Returns (H, p0_target, p1_target).
    """
    G = nx.convert_node_labels_to_integers(G)
    p0, p1, stage_to_phi, phi_by_node, scale = target_pd_from_graph(
        G, phi_type, clustering_scale=clustering_scale, cross_check=cross_check)
    nodes, phi_target, stage, _ = stage_reduction(phi_by_node)
    H = solve_phi(p0, p1, stage, phi_target, phi_type=phi_type,
                  time_limit=time_limit, verbose=verbose,
                  cp_sat_workers=cp_sat_workers, clustering_scale=scale,
                  fourclique_ncap=fourclique_ncap)
    return H, p0, p1


# ═════════════════════════════════════════════════════════════════════════════
# Truncated-scheme public API + fidelity helpers (Axis A / Axis B)
# ═════════════════════════════════════════════════════════════════════════════

def _stage_pd_from_phi_by_node(G, phi_by_node, cross_check=False):
    """Build the stage-reduced (P0,P1) from an arbitrary node->value dict.
    Values may be integers (truncated) or floats (TRUE). Stage reduction ranks
    the DISTINCT values, so the resulting filtration is always integer-timed."""
    nodes, phi_target, stage, stage_to_phi = stage_reduction(phi_by_node)
    stage_of = {v: s for v, s in zip(nodes, stage)}
    node_time = {v: stage_of[v] for v in G.nodes()}
    edge_time = {(min(u, v), max(u, v)): max(stage_of[u], stage_of[v])
                 for u, v in G.edges()}
    p0, p1 = persistence_diagrams(node_time, edge_time)
    if cross_check:
        g0, g1 = gudhi_persistence(node_time, edge_time)
        assert sorted(p0) == sorted(g0)
        assert sorted(p1) == sorted(g1)
    return p0, p1, phi_target, stage


def target_pd_truncated(G, phi_type, K, cross_check=True):
    """Stage-reduced target PD of the K-truncated filtration Phi^(K) on G.
    Returns (p0, p1, phi_target, stage, scale)."""
    phi_by_node, scale = truncated_phi_values(G, phi_type, K)
    p0, p1, phi_target, stage = _stage_pd_from_phi_by_node(
        G, phi_by_node, cross_check=cross_check)
    p0c = [(int(b), None if d == float("inf") else int(d)) for b, d in p0]
    p1c = [(int(b), float("inf")) for b, _ in p1]
    return p0c, p1c, phi_target, stage, scale


def solve_truncated_from_graph(G, phi_type, K, time_limit=60.0, verbose=False,
                               cp_sat_workers=8, cross_check=True):
    """Axis A: compute Phi^(K) target on G, encode, solve.  Returns
    (H, status, solve_time, n_vars, n_aux, scale, p0, p1)."""
    G = nx.convert_node_labels_to_integers(G)
    p0, p1, phi_target, stage, scale = target_pd_truncated(
        G, phi_type, K, cross_check=cross_check)
    H, sname, el, nv, na = solve_phi(
        p0, p1, stage, phi_target, phi_type=phi_type, time_limit=time_limit,
        verbose=verbose, cp_sat_workers=cp_sat_workers, clustering_scale=scale,
        K=K, return_status=True)
    return H, sname, el, nv, na, scale, p0, p1


def true_pd(G, phi_type):
    """Stage-reduced PD of the TRUE (converged) filtration of G (float values)."""
    G2 = nx.convert_node_labels_to_integers(G)
    phi_by = true_phi_values(G2, phi_type)
    p0, p1, _, _ = _stage_pd_from_phi_by_node(G2, phi_by, cross_check=False)
    return p0, p1


def wasserstein1_pd(pd_a, pd_b):
    """Diagonal-padded L-infinity 1-Wasserstein between two PDs.

    Same algorithm as topo_gen/pdm_utils._wasserstein1_pd (vendored here because
    pdm_utils is not present in this checkout).  inf-death points are dropped.
    """
    from scipy.optimize import linear_sum_assignment

    def finite(pd):
        return [(b, d) for b, d in pd if d != float("inf") and not (isinstance(d, float) and math.isinf(d))]

    fa = finite(pd_a)
    fb = finite(pd_b)
    diag_a = [((b + d) / 2, (b + d) / 2) for b, d in fa]
    diag_b = [((b + d) / 2, (b + d) / 2) for b, d in fb]
    pts_a = fa + diag_b
    pts_b = fb + diag_a
    m = len(pts_a)
    if m == 0:
        return 0.0
    cost = np.zeros((m, m), dtype=np.float64)
    for i, (b1, d1) in enumerate(pts_a):
        for j, (b2, d2) in enumerate(pts_b):
            cost[i, j] = max(abs(b1 - b2), abs(d1 - d2))
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].sum())


# ─────────────────────────────────────────────────────────────────────────────
# Solution enumeration (for the completeness test)
# ─────────────────────────────────────────────────────────────────────────────

class _EdgeCollector(cp_model.CpSolverSolutionCallback):
    """Collect the edge set of every feasible M assignment."""
    def __init__(self, M, n, limit=None):
        super().__init__()
        self._M = M
        self._n = n
        self._limit = limit
        self.solutions = []

    def on_solution_callback(self):
        n = self._n
        edges = frozenset(
            (i, j) for i in range(n) for j in range(i + 1, n)
            if self.value(self._M[i, j]) == 1
        )
        self.solutions.append(edges)
        if self._limit is not None and len(self.solutions) >= self._limit:
            self.stop_search()


def enumerate_cp_solutions(p0_target, p1_target, stage, phi_target,
                           phi_type="degree", clustering_scale=1,
                           time_limit=120.0, limit=None, fourclique_ncap=9):
    """Enumerate ALL feasible graphs H (as edge sets on n labeled vertices) that
    the CP model accepts.  Used by the completeness test.  For rational
    filtrations, ``clustering_scale`` must be the SAME exact D used to build
    ``phi_target``.
    """
    aux_cfg = {"clustering_scale": clustering_scale,
               "fourclique_ncap": fourclique_ncap}
    model, M, n, _ = _build_model(
        p0_target, p1_target, stage, phi_target, phi_type, aux_cfg, verbose=False)

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1  # enumeration requires single worker
    if phi_type in _FIXEDPOINT:
        # Disable presolve for the large-coefficient fixed-point encodings: its
        # coefficient-overflow path can both wrongly prune feasible solutions and
        # mislabel the model INFEASIBLE (see _FIXEDPOINT note).  Enumeration must
        # see every solution, so we run without it.
        solver.parameters.cp_model_presolve = False
    collector = _EdgeCollector(M, n, limit=limit)
    status = solver.solve(model, collector)
    return collector.solutions, n, solver.status_name(status)
