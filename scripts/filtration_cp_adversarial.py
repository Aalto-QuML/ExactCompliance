"""Adversarial verification of the pluggable-C1 filtration CP solver.

This does NOT trust the existing harness. It attacks the three claims:

  1. SOUNDNESS is real (round-trip via independent recomputation).
  2. COMPLETENESS is real — and STRICTER than the shipped test: it compares the
     actual SET OF GRAPHS (edge sets), not just the collapsed PD-class, so a solver
     that MISSES valid PD-equivalent graphs is caught. The shipped test cannot catch
     that (both sides collapse to {target_key}).
  3. The truncation refutation is real (true_w1>0 on genuinely non-iso H).

Plus NEGATIVE CONTROLS — the piece every existing test lacks: proof the checker
can FAIL when it should. If the PD-equality check passes a graph with a different
PD, all "PASS" results are meaningless.
"""
import sys, os, itertools, math
from collections import Counter

WT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WT)

import networkx as nx
import numpy as np

from topo_gen.filtration_cp import (
    phi_values, true_phi_values, target_pd_from_graph, stage_reduction,
    enumerate_cp_solutions, solve_from_graph, is_truncated, scale_for,
    solve_truncated_from_graph, true_pd, wasserstein1_pd, _refuse,
)

EXACT = ["degree", "triangle", "clustering", "sqdeg", "wedge", "fourclique", "avgneighdeg"]


# ---------------------------------------------------------------------------
# Independent PD computation (do NOT import the solver's PD path — recompute
# from scratch so a bug there can't hide a bug here).
# ---------------------------------------------------------------------------

def indep_phi(G, phi_type):
    """Independent per-vertex Phi, computed with plain networkx, no solver code."""
    if phi_type == "degree":
        return {v: G.degree(v) for v in G}
    if phi_type == "sqdeg":
        return {v: G.degree(v) ** 2 for v in G}
    if phi_type == "wedge":
        return {v: G.degree(v) * (G.degree(v) - 1) for v in G}
    if phi_type == "triangle":
        return nx.triangles(G)
    if phi_type == "fourclique":
        # count K4's through each vertex, independently
        out = {v: 0 for v in G}
        nodes = list(G)
        adj = {v: set(G[v]) for v in G}
        for quad in itertools.combinations(nodes, 4):
            if all(b in adj[a] for a, b in itertools.combinations(quad, 2)):
                for v in quad:
                    out[v] += 1
        return out
    if phi_type == "clustering":
        return nx.clustering(G)
    if phi_type == "avgneighdeg":
        out = {}
        for v in G:
            d = G.degree(v)
            out[v] = (sum(G.degree(u) for u in G[v]) / d) if d > 0 else 0.0
        return out
    raise ValueError(phi_type)


def indep_pd_key(G, phi_type):
    """PD of G under Phi, computed independently: rank Phi-values into stages,
    build vertex/edge filtration, run union-find PH from scratch."""
    phi = indep_phi(G, phi_type)
    # stage-reduce: rank distinct values
    vals = sorted(set(phi.values()))
    rank = {val: i for i, val in enumerate(vals)}
    node_t = {v: rank[phi[v]] for v in G}
    edge_t = {(u, w): max(node_t[u], node_t[w]) for u, w in G.edges()}
    return _pd_key_uf(node_t, edge_t, G.number_of_nodes())


def _pd_key_uf(node_t, edge_t, n):
    """Union-find PH, fully independent implementation."""
    parent = {}; born = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    events = [(t, 0, v) for v, t in node_t.items()] + [(t, 1, e) for e, t in edge_t.items()]
    events.sort(key=lambda x: (x[0], x[1]))
    pd0 = []; pd1 = []
    for t, dim, s in events:
        if dim == 0:
            parent[s] = s; born[s] = t
        else:
            u, w = s
            ru, rw = find(u), find(w)
            if ru == rw:
                pd1.append((t, math.inf))
            else:
                # elder rule: smaller born survives
                if born[ru] < born[rw]: ru, rw = rw, ru
                pd0.append((born[ru], t))
                parent[ru] = rw
                if born[rw] > born[ru]: born[rw] = born[rw]  # survivor keeps its birth
    for r, b in born.items():
        if find(r) == r:
            pd0.append((b, math.inf))
    return (tuple(sorted(pd0)), tuple(sorted(pd1)))


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS — prove the checker can fail
# ---------------------------------------------------------------------------

def negative_controls():
    print("=" * 70)
    print("NEGATIVE CONTROLS (checker must DISTINGUISH different PDs)")
    print("=" * 70)
    ok = True

    # C1: two graphs with genuinely different degree-PDs must get different keys
    P4 = nx.path_graph(4)
    C4 = nx.cycle_graph(4)
    for phi in ["degree", "triangle", "clustering"]:
        kP = indep_pd_key(P4, phi); kC = indep_pd_key(C4, phi)
        distinct = (kP != kC)
        # P4 vs C4: degrees differ (1,2,2,1 vs 2,2,2,2) -> different PD expected
        print(f"  [{phi}] P4 key != C4 key : {distinct}  {'OK' if distinct else 'FAIL!!'}")
        ok &= distinct

    # C2: my independent PD must AGREE with the solver's PD path on the SAME graph
    #     (cross-implementation agreement — if they disagree, one is buggy)
    from topo_gen.filtration_cp import target_pd_from_graph
    agree = True
    for phi in EXACT:
        for G in [nx.path_graph(5), nx.cycle_graph(5), nx.complete_graph(5),
                  nx.star_graph(4), nx.wheel_graph(5)]:
            G = nx.convert_node_labels_to_integers(G)
            mine = indep_pd_key(G, phi)
            p0, p1, _, _, _ = target_pd_from_graph(G, phi, cross_check=False)
            theirs = (tuple(sorted((b, (d if d is not None else math.inf)) for b, d in p0)),
                      tuple(sorted((b, math.inf) for b, _ in p1)))
            if mine != theirs:
                agree = False
                print(f"  [{phi}] DISAGREE on {G}: mine={mine} theirs={theirs}")
    print(f"  independent-PD == solver-PD on all probes : {agree}  {'OK' if agree else 'FAIL!!'}")
    ok &= agree

    # C3: a WRONG graph must NOT match the target key (sanity that == isn't trivial)
    tgt = indep_pd_key(nx.cycle_graph(6), "degree")
    wrong = indep_pd_key(nx.path_graph(6), "degree")
    print(f"  [degree] wrong graph (P6) != target (C6) : {tgt != wrong}  {'OK' if tgt!=wrong else 'FAIL!!'}")
    ok &= (tgt != wrong)

    print(f"\n  NEGATIVE CONTROLS: {'ALL OK' if ok else 'FAILURE — results below are UNTRUSTWORTHY'}")
    return ok


# ---------------------------------------------------------------------------
# STRICT completeness: compare the SET OF GRAPHS, not the collapsed PD-class
# ---------------------------------------------------------------------------

def _all_graphs(n):
    pairs = list(itertools.combinations(range(n), 2))
    for bits in itertools.product([0, 1], repeat=len(pairs)):
        G = nx.Graph(); G.add_nodes_from(range(n))
        G.add_edges_from(p for p, b in zip(pairs, bits) if b)
        yield G, frozenset(p for p, b in zip(pairs, bits) if b)


def strict_completeness(phi_type, n, targets, time_limit=30.0):
    """CP must reproduce EXACTLY the set of labeled graphs whose Phi-PD == target.
    This catches a solver that misses valid graphs (the shipped test cannot)."""
    print(f"\n[strict-completeness/{phi_type}] n={n}")
    allpass = True
    for name, Gt in targets:
        Gt = nx.convert_node_labels_to_integers(Gt)
        tgt = indep_pd_key(Gt, phi_type)

        # brute force: ALL labeled edge-sets whose independent PD == target
        brute = set()
        for G, es in _all_graphs(n):
            if indep_pd_key(G, phi_type) == tgt:
                brute.add(es)

        # CP enumeration
        p0, p1, _, phi_by, scale = target_pd_from_graph(Gt, phi_type, cross_check=False)
        _, phi_target, stage, _ = stage_reduction(phi_by)
        sols, n_cp, status = enumerate_cp_solutions(
            p0, p1, stage, phi_target, phi_type=phi_type,
            clustering_scale=scale, time_limit=time_limit)
        cp = set(frozenset(tuple(sorted(e)) for e in edges) for edges in sols)

        exhaustive = (status != "UNKNOWN")
        # every CP graph must actually realize the target (independent recheck)
        cp_all_valid = all(
            indep_pd_key(_es_to_G(es, n), phi_type) == tgt for es in cp)
        exact = (cp == brute) and cp_all_valid and exhaustive
        allpass &= exact
        missing = len(brute - cp); extra = len(cp - brute)
        print(f"  {name:12s}: brute={len(brute):5d} cp={len(cp):5d} "
              f"missing={missing} extra={extra} valid={cp_all_valid} "
              f"status={status}  {'PASS' if exact else 'FAIL'}")
    return allpass


def _es_to_G(es, n):
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(es); return G


# ---------------------------------------------------------------------------
# Adversarial SOUNDNESS: hard graph families + bigger n, independent round-trip
# ---------------------------------------------------------------------------

def adversarial_soundness(time_limit=60.0):
    print("\n" + "=" * 70)
    print("ADVERSARIAL SOUNDNESS (hard families, independent round-trip)")
    print("=" * 70)
    fams = [
        ("P8", nx.path_graph(8)), ("C8", nx.cycle_graph(8)),
        ("K5", nx.complete_graph(5)),
        ("Petersen", nx.petersen_graph()),                 # 3-regular: degree uninformative
        ("K3,3", nx.complete_bipartite_graph(3, 3)),        # regular, no triangles
        ("cube Q3", nx.hypercube_graph(3)),                 # 3-regular bipartite
        ("Prism", nx.circular_ladder_graph(4)),             # 3-regular
        ("ER8", nx.gnp_random_graph(8, 0.4, seed=7)),
        ("ER9", nx.gnp_random_graph(9, 0.35, seed=11)),
        ("2 triangles", nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))),
    ]
    tot = {p: [0, 0, 0] for p in EXACT}  # pass, fail, nosol
    for phi in EXACT:
        for name, G in fams:
            G = nx.convert_node_labels_to_integers(G)
            n = G.number_of_nodes()
            if phi == "fourclique" and n > 9:
                continue
            tgt = indep_pd_key(G, phi)
            try:
                H = solve_from_graph(G, phi_type=phi, time_limit=time_limit, verbose=False)
            except NotImplementedError:
                H = None
            if H is None:
                tot[phi][2] += 1
                print(f"  [{phi:11s}] {name:14s}: NO-SOLUTION")
                continue
            got = indep_pd_key(H, phi)
            iso = nx.is_isomorphic(G, H)
            if got == tgt:
                tot[phi][0] += 1
            else:
                tot[phi][1] += 1
                print(f"  [{phi:11s}] {name:14s}: PD MISMATCH  iso={iso}")
        p, f, ns = tot[phi]
        print(f"  => {phi:11s}: {p} PASS  {f} FAIL  {ns} no-sol")
    return all(v[1] == 0 for v in tot.values())


# ---------------------------------------------------------------------------
# Adversarial truncation: independent true-value recompute + hand-checked case
# ---------------------------------------------------------------------------

def adversarial_truncation():
    print("\n" + "=" * 70)
    print("ADVERSARIAL TRUNCATION (independent true-value; contrast trunc vs true)")
    print("=" * 70)
    # A graph where degree does NOT determine the graph: two non-iso graphs with
    # same degree sequence. Use for eigvec/katz where the surrogate can pick the
    # 'wrong' one.
    fams = [("P6", nx.path_graph(6)), ("C6", nx.cycle_graph(6)),
            ("star5", nx.star_graph(5)), ("paw", nx.Graph([(0,1),(1,2),(2,0),(2,3)])),
            ("bull", nx.Graph([(0,1),(1,2),(2,0),(1,3),(2,4)]))]
    for scheme in ["eigvec_K", "katz_K", "pagerank_K", "rw_K"]:
        print(f"\n  scheme={scheme}")
        for name, G in fams:
            G = nx.convert_node_labels_to_integers(G)
            for K in [1, 2, 4]:
                try:
                    ret = solve_truncated_from_graph(G, scheme, K, time_limit=30.0, verbose=False)
                    H, status = ret[0], ret[1]
                except Exception as e:
                    print(f"    {name:6s} K={K}: ERROR {type(e).__name__}: {str(e)[:50]}")
                    continue
                if H is None:
                    print(f"    {name:6s} K={K}: {status}")
                    continue
                # independent true W1: recompute converged value with networkx myself
                tw = _indep_true_w1(G, H, scheme)
                iso = nx.is_isomorphic(G, H)
                print(f"    {name:6s} K={K}: status={str(status):8s} "
                      f"iso={'Y' if iso else 'N'} true_w1(indep)={tw:.3f}")


def _indep_true_w1(G, H, scheme):
    """Recompute the TRUE converged filtration PD for both G and H with networkx,
    independent of the solver's true_pd path, and W1 them."""
    def true_vals(X):
        if scheme == "eigvec_K":
            try: return nx.eigenvector_centrality_numpy(X)
            except Exception: return {v: 0.0 for v in X}
        if scheme == "katz_K":
            try: return nx.katz_centrality_numpy(X, alpha=0.1)
            except Exception: return {v: 0.0 for v in X}
        if scheme == "pagerank_K":
            return nx.pagerank(X, alpha=0.85)
        if scheme == "rw_K":
            m = X.number_of_edges()
            return {v: (X.degree(v) / (2 * m) if m else 0.0) for v in X}
    def key(X):
        phi = true_vals(X)
        vals = sorted(set(round(v, 9) for v in phi.values()))
        rank = {v: i for i, v in enumerate(vals)}
        nt = {v: rank[round(phi[v], 9)] for v in X}
        et = {(u, w): max(nt[u], nt[w]) for u, w in X.edges()}
        return _pd_key_uf(nt, et, X.number_of_nodes())
    # W1 on the raw (birth,death) multisets of the stage-reduced true PDs
    (p0g, _), (p0h, _) = key(G), key(H)
    return wasserstein1_pd(
        [(b, d) for b, d in p0g], [(b, d) for b, d in p0h])


if __name__ == "__main__":
    controls_ok = negative_controls()
    if not controls_ok:
        print("\nABORTING: negative controls failed; nothing below can be trusted.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("STRICT COMPLETENESS (set-of-graphs, catches missed solutions)")
    print("=" * 70)
    targets6 = [("path_6", nx.path_graph(6)), ("cycle_6", nx.cycle_graph(6)),
                ("star_5", nx.star_graph(5)), ("tri+edge", nx.Graph([(0,1),(1,2),(2,0),(3,4)])),
                ("K4", nx.complete_graph(4))]
    sc_ok = True
    for phi in EXACT:
        t = [(nm, G) for nm, G in targets6 if not (phi == "fourclique" and G.number_of_nodes() > 9)]
        sc_ok &= strict_completeness(phi, 6, t)

    snd_ok = adversarial_soundness()
    adversarial_truncation()

    print("\n" + "=" * 70)
    print(f"VERDICT: controls={'OK' if controls_ok else 'FAIL'}  "
          f"strict-completeness={'OK' if sc_ok else 'FAIL'}  "
          f"adversarial-soundness={'OK' if snd_ok else 'FAIL'}")
    print("=" * 70)
