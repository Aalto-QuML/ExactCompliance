"""
Verification harness for the FIXED-POINT-CERTIFIABLE filtrations
================================================================
closeness / katz / pagerank -- three vertex filtrations that are NOT finite
polynomials/rationals in the adjacency M, and were previously refused by the
pluggable CP solver.  They are, however, *fixed-point certifiable*: each is the
UNIQUE solution of a finite constraint system whose auxiliary variables are
forced to equal Phi (a certificate, exactly like the paper's spanning-tree
connectivity constraint).  See topo_gen/filtration_cp.py, the Tier-3 C1 builders
(_phi_alignment_closeness / _katz / _pagerank).

This script is the DELIVERABLE.  It does three things per filtration:

1. SOUNDNESS on the 95-case structural suite (reused verbatim from
   degree_filtration_cp_v4.run_suite's graph list).  CLOSENESS is restricted to
   the CONNECTED subset (disconnected graphs are annotated SKIP).  For each graph:
     - compute target Phi with networkx (nx.closeness_centrality /
       katz_centrality_numpy / pagerank),
     - build the stage-reduced target PD (union-find, gudhi cross-checked),
     - solve for H,
     - INDEPENDENTLY recompute Phi on H with networkx, build H's PD,
     - assert PD == target PD as a multiset up to relabeling (NEVER adjacency).
   A ScaleOverflow (exact rational scale D exceeds the CP-SAT safe domain) is a
   REPORTED result, not a hidden failure.

2. COMPLETENESS exhaustive up to ISOMORPHISM at n <= 6.  Enumerate all labeled
   graphs on n vertices (connected-only for closeness), collect the iso-classes
   whose Phi-PD matches the target; separately enumerate ALL CP solutions; assert
   the two iso-class sets coincide AND every CP solution is independently valid.

3. Every independent PD recomputation is from scratch via networkx + union-find
   (never the solver's own PD path), so the solver cannot check itself.
"""

import argparse
import itertools
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import networkx as nx

TOPOGEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TOPOGEN_ROOT not in sys.path:
    sys.path.insert(0, TOPOGEN_ROOT)

from topo_gen import filtration_cp as fc
from topo_gen.filtration_cp import (
    stage_reduction,
    target_pd_from_graph,
    solve_phi,
    enumerate_cp_solutions,
    ScaleOverflow,
)

FIXEDPOINT_PHI = ["closeness", "katz", "pagerank"]


# ─────────────────────────────────────────────────────────────────────────────
# INDEPENDENT Phi + PD recomputation (from scratch; never the solver's PD path)
# ─────────────────────────────────────────────────────────────────────────────

def independent_phi(G, phi_type):
    """Per-vertex Phi via plain networkx, computed with NO reference to the CP
    encoding or its scaled integer targets.  Returns node -> float.

      closeness : nx.closeness_centrality
      katz      : nx.katz_centrality_numpy (alpha = 1/10, unnormalised)
      pagerank  : nx.pagerank (alpha = 0.85)
    Raises ValueError for a disconnected graph under closeness.
    """
    G = nx.convert_node_labels_to_integers(G)
    if phi_type == "closeness":
        if not nx.is_connected(G) or G.number_of_nodes() < 2:
            raise ValueError("closeness requires connected n>=2")
        raw = dict(nx.closeness_centrality(G))
    elif phi_type == "katz":
        raw = dict(nx.katz_centrality_numpy(G, alpha=0.1, normalized=False))
    elif phi_type == "pagerank":
        raw = dict(nx.pagerank(G, alpha=0.85))
    else:
        raise ValueError(phi_type)
    # Round to a tolerance FAR below the smallest genuine gap between distinct
    # Phi values (empirically ~1e-2 for these filtrations on small graphs) but FAR
    # above float noise (~1e-15).  Without this, numpy/pagerank last-ULP jitter
    # (e.g. 1.25 vs 1.2500000000000002 on relabeled cycles) would split a single
    # Phi value into two spurious stages, wrongly rejecting a valid solution.  The
    # stage reduction only cares about the RANK of distinct values, so rounding is
    # exact for the topology while killing the noise.
    return {v: round(val, 9) for v, val in raw.items()}


def independent_pd_key(G, phi_type):
    """From-scratch stage-reduced Phi-filtration PD of G, as a canonical multiset
    key.  Uses independent_phi (float networkx values) + a from-scratch union-find
    style persistence via fc._stage_pd_from_phi_by_node (which stage-ranks the
    DISTINCT float values, so it never touches the solver's integer-scaled path).
    """
    G = nx.convert_node_labels_to_integers(G)
    phi_by = independent_phi(G, phi_type)
    p0, p1, _, _ = fc._stage_pd_from_phi_by_node(G, phi_by, cross_check=False)
    s0 = tuple(sorted((b, (d if d != float("inf") else float("inf"))) for b, d in p0))
    s1 = tuple(sorted((b, float("inf")) for b, _ in p1))
    return (s0, s1)


# ─────────────────────────────────────────────────────────────────────────────
# The 95-case structural suite (graph list reused from run_suite in
# degree_filtration_cp_v4.py, verbatim)
# ─────────────────────────────────────────────────────────────────────────────

def suite_graphs(seed=123):
    G = []
    for n in [3, 4, 5, 6, 8, 10, 12, 15, 20, 25]:
        G.append((f"Path P{n}", nx.path_graph(n), "Path"))
    for n in [3, 4, 5, 6, 8, 10, 12, 15, 20, 25]:
        G.append((f"Cycle C{n}", nx.cycle_graph(n), "Cycle"))
    for k in [2, 3, 4, 5, 6, 8, 10, 12]:
        G.append((f"Star K_1_{k}", nx.star_graph(k), "Star"))
    for n in [3, 4, 5, 6, 7]:
        G.append((f"Complete K{n}", nx.complete_graph(n), "Complete"))
    for n, s in [(6, 857), (8, 4385), (10, 1428), (10, 6672), (12, 4367),
                 (12, 1764), (15, 625), (15, 6211), (18, 8785), (20, 9213),
                 (20, 5442), (25, 5583)]:
        G.append((f"Tree n={n} s={s}", nx.random_labeled_tree(n, seed=s), "Tree"))
    for n, p, s in [(6, .2, 850), (8, .2, 2615), (10, .2, 2212), (10, .2, 5524),
                    (12, .2, 9190), (12, .2, 5468), (15, .2, 4016), (15, .2, 2683),
                    (18, .2, 27), (20, .2, 7147)]:
        g = nx.erdos_renyi_graph(n, p, seed=s)
        if not g.edges():
            g.add_edge(0, 1)
        G.append((f"ER p=0.2 n={n} s={s}", g, "ER-p0.2"))
    for n, p, s in [(6, .3, 1435), (8, .3, 9791), (10, .3, 6186), (10, .3, 1144),
                    (12, .3, 108), (12, .3, 5168), (15, .3, 7345), (15, .3, 1671),
                    (18, .3, 719), (20, .3, 1519)]:
        g = nx.erdos_renyi_graph(n, p, seed=s)
        if not g.edges():
            g.add_edge(0, 1)
        G.append((f"ER p=0.3 n={n} s={s}", g, "ER-p0.3"))
    for n, p, s in [(6, .5, 2329), (8, .5, 2068), (10, .5, 347), (10, .5, 4780),
                    (12, .5, 7055), (15, .5, 9394), (18, .5, 7813)]:
        G.append((f"ER p=0.5 n={n} s={s}", nx.erdos_renyi_graph(n, p, seed=s), "ER-p0.5"))
    G.append(("2 disjoint edges", nx.Graph([(0, 1), (2, 3)]), "Disconnected"))
    G.append(("3 disjoint edges", nx.Graph([(0, 1), (2, 3), (4, 5)]), "Disconnected"))
    G.append(("Path3+edge", nx.Graph([(0, 1), (1, 2), (3, 4)]), "Disconnected"))
    G.append(("Triangle+edge", nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4)]), "Disconnected"))
    G.append(("2 triangles disconn.",
              nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]), "Disconnected"))
    for r, c in [(2, 3), (2, 4), (3, 3), (2, 5), (3, 4)]:
        G.append((f"Grid {r}x{c}", nx.grid_2d_graph(r, c), "Grid"))
    for m1, m2 in [(3, 3), (4, 4), (3, 4), (4, 5)]:
        G.append((f"Barbell ({m1},{m2})", nx.barbell_graph(m1, m2), "Barbell"))
    G.append(("Lollipop (4,2)", nx.lollipop_graph(4, 2), "Lollipop"))
    G.append(("Lollipop (5,3)", nx.lollipop_graph(5, 3), "Lollipop"))
    for n in [5, 6, 7, 8]:
        G.append((f"Wheel W{n}", nx.wheel_graph(n), "Wheel"))
    G.append(("Petersen", nx.petersen_graph(), "Special"))
    G.append(("Ladder n=4", nx.ladder_graph(4), "Special"))
    G.append(("Ladder n=6", nx.ladder_graph(6), "Special"))
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 1. SOUNDNESS
# ─────────────────────────────────────────────────────────────────────────────

def run_soundness(phi_type, time_limit, verbose=True):
    graphs = suite_graphs()
    n_pass = n_fail = n_nosol = n_timeout = n_skip = n_overflow = 0
    fails = []
    overflow_ns = []
    considered = 0

    for name, G0, cat in graphs:
        G = nx.convert_node_labels_to_integers(G0)
        connected = nx.is_connected(G) and G.number_of_nodes() >= 2

        if phi_type == "closeness" and not connected:
            n_skip += 1
            continue

        considered += 1

        # independent target PD (with gudhi cross-check inside target_pd_from_graph)
        try:
            p0, p1, _, phi_by, scale = target_pd_from_graph(
                G, phi_type, clustering_scale=None, cross_check=True)
        except ValueError:
            # disconnected under closeness already skipped; any other ValueError
            # here means the ground-truth Phi is undefined -> skip honestly.
            n_skip += 1
            considered -= 1
            continue

        _, phi_target, stage, _ = stage_reduction(phi_by)
        target_key = independent_pd_key(G, phi_type)   # from-scratch, independent

        try:
            H, sname, el, nv, na = solve_phi(
                p0, p1, stage, phi_target, phi_type=phi_type,
                time_limit=time_limit, verbose=False, clustering_scale=scale,
                return_status=True)
        except ScaleOverflow as e:
            n_overflow += 1
            overflow_ns.append((name, e.n, e.bits))
            continue

        if H is None:
            if sname == "INFEASIBLE":
                n_fail += 1
                fails.append((name, "INFEASIBLE on a realizable target (source is a witness)"))
            else:
                n_timeout += 1
            continue

        got_key = independent_pd_key(H, phi_type)      # from-scratch, independent
        if got_key == target_key:
            n_pass += 1
        else:
            n_fail += 1
            fails.append((name, f"PD mismatch\n      target={target_key}\n      got   ={got_key}"))

    if verbose:
        parts = [f"{n_pass} PASS", f"{n_fail} FAIL", f"{n_nosol + n_timeout} timeout",
                 f"{n_overflow} overflow"]
        if phi_type == "closeness":
            parts.append(f"{n_skip} skip-disconnected")
        print(f"[soundness/{phi_type}] " + "  ".join(parts)
              + f"   (considered {considered} of {len(graphs)})")
        for nm, m, b in overflow_ns[:20]:
            print(f"      OVERFLOW {nm}: n={m}, D has {b} bits (>62)")
        for nm, msg in fails[:20]:
            print(f"      FAIL {nm}: {msg}")

    return dict(passed=n_pass, failed=n_fail, timeout=n_timeout, overflow=n_overflow,
                skipped=n_skip, considered=considered,
                overflow_ns=overflow_ns, fails=fails)


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPLETENESS (exhaustive up to isomorphism at n <= 6)
# ─────────────────────────────────────────────────────────────────────────────

def _all_labeled_graphs(n, connected_only=False):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for bits in itertools.product([0, 1], repeat=len(pairs)):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(p for p, b in zip(pairs, bits) if b)
        if connected_only and (n < 2 or not nx.is_connected(G)):
            continue
        yield G


def _edges_to_graph(edges, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def _iso_dedup(graphs):
    """Deduplicate a list of graphs up to isomorphism; return the list of reps."""
    reps = []
    for G in graphs:
        if not any(nx.is_isomorphic(G, R) for R in reps):
            reps.append(G)
    return reps


def run_completeness(phi_type, n, target_graphs, time_limit, verbose=True):
    """For each target graph on n vertices:
        BRUTE : all labeled graphs (connected-only for closeness) whose
                independent Phi-PD == target, deduped to iso-classes.
        CP    : all CP solutions (enumerate_cp_solutions), deduped to iso-classes,
                each independently validated (its Phi-PD == target).
      Assert the two iso-class sets coincide AND every CP solution is valid.
    """
    connected_only = (phi_type == "closeness")
    results = []

    for name, Gt in target_graphs:
        Gt = nx.convert_node_labels_to_integers(Gt)
        if connected_only and not nx.is_connected(Gt):
            results.append(dict(name=name, n=n, status="SKIP-disconnected-target",
                                match=None))
            if verbose:
                print(f"[completeness/{phi_type}] {name} n={n}: SKIP (disconnected target)")
            continue

        target_key = independent_pd_key(Gt, phi_type)

        # ---- BRUTE ground truth: iso-classes of matching labeled graphs --------
        brute_matches = []
        for G in _all_labeled_graphs(n, connected_only=connected_only):
            if independent_pd_key(G, phi_type) == target_key:
                brute_matches.append(G)
        brute_reps = _iso_dedup(brute_matches)

        # ---- CP enumeration ----------------------------------------------------
        p0, p1, _, phi_by, scale = target_pd_from_graph(
            Gt, phi_type, clustering_scale=None, cross_check=False)
        _, phi_target, stage, _ = stage_reduction(phi_by)
        try:
            sols, n_cp, status = enumerate_cp_solutions(
                p0, p1, stage, phi_target, phi_type=phi_type,
                clustering_scale=scale, time_limit=time_limit)
        except ScaleOverflow as e:
            results.append(dict(name=name, n=n, status=f"OVERFLOW(bits={e.bits})",
                                match=None))
            if verbose:
                print(f"[completeness/{phi_type}] {name} n={n}: OVERFLOW "
                      f"(D {e.bits} bits)")
            continue

        cp_graphs = [_edges_to_graph(edges, n_cp) for edges in sols]
        cp_all_valid = all(independent_pd_key(H, phi_type) == target_key
                           for H in cp_graphs)
        # connected-only sanity for closeness (a disconnected CP solution would be
        # a leak, since closeness is only defined on connected graphs)
        if connected_only:
            cp_all_valid = cp_all_valid and all(nx.is_connected(H) for H in cp_graphs)
        cp_reps = _iso_dedup(cp_graphs)

        # iso-class set equality: every brute rep matches some cp rep and vice versa
        def _class_sets_equal(A, B):
            if len(A) != len(B):
                return False
            for a in A:
                if not any(nx.is_isomorphic(a, b) for b in B):
                    return False
            return True

        classes_match = _class_sets_equal(brute_reps, cp_reps) and cp_all_valid
        exhaustive = (status != "UNKNOWN")

        results.append(dict(
            name=name, n=n,
            brute_labeled=len(brute_matches), brute_iso=len(brute_reps),
            cp_sols=len(sols), cp_iso=len(cp_reps),
            cp_all_valid=cp_all_valid, classes_match=classes_match,
            status=status, exhaustive=exhaustive,
            match=(classes_match and exhaustive),
        ))
        if verbose:
            flag = "PASS" if (classes_match and exhaustive) else "FAIL"
            print(f"[completeness/{phi_type}] {name} n={n}: {flag}  "
                  f"brute {len(brute_matches)} labeled -> {len(brute_reps)} iso-classes, "
                  f"cp {len(sols)} sols -> {len(cp_reps)} iso-classes, "
                  f"cp_all_valid={cp_all_valid}, status={status}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120.0,
                    help="per-solve time limit (heavy encodings; timeout is a reported result)")
    ap.add_argument("--completeness-n", type=int, default=6,
                    help="exhaustive completeness vertex count (<=6)")
    ap.add_argument("--completeness-time-limit", type=float, default=120.0)
    ap.add_argument("--phi", nargs="+", default=FIXEDPOINT_PHI)
    ap.add_argument("--skip-soundness", action="store_true")
    ap.add_argument("--skip-completeness", action="store_true")
    args = ap.parse_args()

    summary = {}

    if not args.skip_soundness:
        print("=" * 74)
        print("SOUNDNESS  (95-case structural suite; closeness on connected subset)")
        print("=" * 74)
        for phi in args.phi:
            t0 = time.time()
            summary.setdefault(phi, {})["soundness"] = run_soundness(
                phi, time_limit=args.time_limit)
            print(f"    ({phi} soundness took {time.time()-t0:.1f}s)")

    if not args.skip_completeness:
        n = args.completeness_n
        print("\n" + "=" * 74)
        print(f"COMPLETENESS  (exhaustive up to isomorphism, n={n})")
        print("=" * 74)
        # every target MUST have exactly n nodes: star_graph(k) has k+1 nodes,
        # wheel_graph(m) has m nodes in this networkx version (hub + (m-1)-cycle).
        targets = [
            (f"path_{n}", nx.path_graph(n)),                       # n nodes
            (f"cycle_{n}", nx.cycle_graph(n)),                     # n nodes
            (f"complete_{n}", nx.complete_graph(n)),               # n nodes
            (f"star_{n}", nx.star_graph(n - 1)),                   # n nodes
            (f"wheel_{n}", nx.wheel_graph(n)),                     # n nodes
        ]
        targets = [(nm, g) for nm, g in targets if g.number_of_nodes() == n]
        for phi in args.phi:
            t0 = time.time()
            summary.setdefault(phi, {})["completeness"] = run_completeness(
                phi, n, targets, time_limit=args.completeness_time_limit)
            print(f"    ({phi} completeness took {time.time()-t0:.1f}s)")

    print("\n" + "=" * 74)
    print("HEADLINE")
    print("=" * 74)
    for phi in args.phi:
        s = summary.get(phi, {}).get("soundness")
        if s:
            print(f"  {phi:<10} soundness: {s['passed']}/{s['considered']} PASS  "
                  f"({s['failed']} fail, {s['timeout']} timeout, "
                  f"{s['overflow']} overflow"
                  + (f", {s['skipped']} skip-disconnected" if phi == 'closeness' else "")
                  + ")")
            if s["overflow_ns"]:
                first = min(b for _, _, b in s["overflow_ns"])
                minn = min(nn for _, nn, _ in s["overflow_ns"])
                print(f"             overflow first bites at n={minn} "
                      f"(D ~{first} bits); see per-case list above")
        c = summary.get(phi, {}).get("completeness")
        if c:
            ok = sum(1 for r in c if r.get("match"))
            total = sum(1 for r in c if r.get("match") is not None)
            print(f"  {phi:<10} completeness: {ok}/{total} targets: CP iso-classes "
                  f"== brute iso-classes (exhaustive)")
    return summary


if __name__ == "__main__":
    main()
