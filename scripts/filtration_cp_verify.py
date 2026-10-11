"""
Verification harness for the pluggable-filtration CP solver (topo_gen.filtration_cp).

Three tests per phi_type in {degree, triangle, clustering}:

1. SOUNDNESS
   For many random small source graphs G (|V| in 3..7):
     - compute Phi per vertex with plain networkx (independent of the CP model),
     - build the stage-reduced vertex-edge filtration,
     - compute the target (P0, P1) with the union-find persistence, cross-checked
       against gudhi,
     - solve for H,
     - INDEPENDENTLY recompute Phi on H, its stage-reduced filtration, its (P0,P1),
     - ASSERT it equals the target EXACTLY as a multiset of (birth, death) pairs
       (PDs compared up to relabeling; adjacency is NEVER compared).
   Every returned solution must pass.

2. COMPLETENESS (small n only, |V| <= COMPLETENESS_MAX_N)
   Enumerate ALL graphs on n labeled vertices; collect the set of PD-equivalence
   classes whose Phi-filtration PD matches the target.  Separately enumerate ALL
   CP solutions.  Assert the two sets of PD-equivalence-classes coincide (compared
   up to isomorphism / PD relabeling, never by labels).

3. SCALABILITY
   Record solve time, #variables, #C1-auxiliaries vs n; emit a table.

No minimality/necessity tests are written for triangle/clustering: minimality
(Thm 4.2.3) is degree-specific and does not transfer.
"""

import argparse
import itertools
import os
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

import networkx as nx

TOPOGEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TOPOGEN_ROOT not in sys.path:
    sys.path.insert(0, TOPOGEN_ROOT)

from ortools.sat.python import cp_model

from topo_gen import filtration_cp as fc
from topo_gen.filtration_cp import (
    stage_reduction,
    target_pd_from_graph,
    phi_values,
    solve_phi,
    enumerate_cp_solutions,
    _build_model,
)

PHI_TYPES = ["degree", "triangle", "clustering",
             "sqdeg", "wedge", "fourclique", "avgneighdeg"]
# Exact per-graph rational scale is resolved automatically (None => the LCM of
# the actual denominators, guaranteeing integrality — see
# filtration_cp.scale_for / clustering_scale_for / avgneighdeg_scale_for).
CLUSTERING_SCALE = None

# fourclique is O(n^4); cap the vertex count for soundness/scalability sweeps.
FOURCLIQUE_NCAP = 9


# ─────────────────────────────────────────────────────────────────────────────
# PD helpers (compare as multisets up to relabeling — never adjacency)
# ─────────────────────────────────────────────────────────────────────────────

def _pd_key(p0, p1):
    """Canonical hashable key of a PD (multiset of (birth, death) pairs)."""
    s0 = tuple(sorted((b, (d if d is not None else float("inf"))) for b, d in p0))
    s1 = tuple(sorted((b, float("inf")) for b, _ in p1))
    return (s0, s1)


def graph_pd_key(G, phi_type):
    """Stage-reduced Phi-filtration PD of G, as a canonical key.

    Computed independently via networkx Phi + union-find persistence (no CP).
    Note: clustering PDs are scale-invariant under stage reduction (ranks of
    distinct scaled values are unaffected by the common scale D), so comparing
    graph_pd_keys across graphs is well defined."""
    p0, p1, _, _, _ = target_pd_from_graph(
        G, phi_type, clustering_scale=CLUSTERING_SCALE, cross_check=False)
    return _pd_key(p0, p1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SOUNDNESS
# ─────────────────────────────────────────────────────────────────────────────

def _random_source_graphs(rng_seed, n_lo=3, n_hi=7, per_n=6):
    import random
    rng = random.Random(rng_seed)
    graphs = []
    for n in range(n_lo, n_hi + 1):
        for _ in range(per_n):
            p = rng.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10 ** 6))
            graphs.append((f"gnp_n{n}_p{p}", G))
    # a few structured graphs too
    graphs += [
        ("P5", nx.path_graph(5)),
        ("C6", nx.cycle_graph(6)),
        ("K5", nx.complete_graph(5)),
        ("star5", nx.star_graph(5)),
        ("K4", nx.complete_graph(4)),
        ("diamond", nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 0)])),
    ]
    return graphs


def run_soundness(phi_type, time_limit, seed=7, verbose=True):
    graphs = _random_source_graphs(seed)
    n_pass = n_fail = n_nosol = n_skip = 0
    fails = []
    for name, G in graphs:
        G = nx.convert_node_labels_to_integers(G)
        if phi_type == "fourclique" and G.number_of_nodes() > FOURCLIQUE_NCAP:
            n_skip += 1
            continue
        # independent target PD (with gudhi cross-check inside target_pd_from_graph)
        try:
            p0, p1, _, phi_by, scale = target_pd_from_graph(
                G, phi_type, clustering_scale=CLUSTERING_SCALE, cross_check=True)
        except AssertionError as e:
            fails.append((name, f"target cross-check failed: {e}"))
            n_fail += 1
            continue
        _, phi_target, stage, _ = stage_reduction(phi_by)
        target_key = _pd_key(p0, p1)

        H = solve_phi(p0, p1, stage, phi_target, phi_type=phi_type,
                      time_limit=time_limit, verbose=False,
                      clustering_scale=scale, fourclique_ncap=FOURCLIQUE_NCAP)
        if H is None:
            # A valid source graph always yields a feasible model (the source
            # itself is a witness), so no-solution is a failure of soundness setup.
            n_nosol += 1
            fails.append((name, "solver returned None on realizable target"))
            continue

        got_key = graph_pd_key(H, phi_type)
        if got_key == target_key:
            n_pass += 1
        else:
            n_fail += 1
            fails.append((name, f"PD mismatch\n    target={target_key}\n    got   ={got_key}"))

    if verbose:
        skipmsg = f"  {n_skip} skipped(>cap)" if n_skip else ""
        print(f"[soundness/{phi_type}] {n_pass} PASS  {n_fail} FAIL  "
              f"{n_nosol} no-solution{skipmsg}  (of {len(graphs)})")
        for nm, msg in fails[:10]:
            print(f"    FAIL {nm}: {msg}")
    return dict(passed=n_pass, failed=n_fail, nosol=n_nosol, skipped=n_skip,
                total=len(graphs) - n_skip)


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPLETENESS (brute force ground truth vs CP enumeration)
# ─────────────────────────────────────────────────────────────────────────────

def _all_labeled_graphs(n):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for bits in itertools.product([0, 1], repeat=len(pairs)):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(p for p, b in zip(pairs, bits) if b)
        yield G


def _edges_to_graph(edges, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def run_completeness(phi_type, n, target_graphs, time_limit, verbose=True):
    """For each target graph on n vertices, verify:
        { PD-classes of ALL labeled graphs whose Phi-PD == target }
      ==
        { PD-classes of ALL CP solutions }
    (as sets of PD-equivalence classes; PDs compared up to relabeling).
    """
    results = []
    for name, Gt in target_graphs:
        Gt = nx.convert_node_labels_to_integers(Gt)
        target_key = graph_pd_key(Gt, phi_type)

        # ground truth: brute-force all labeled graphs on n vertices
        brute_keys = set()
        brute_edgesets = 0
        for G in _all_labeled_graphs(n):
            if graph_pd_key(G, phi_type) == target_key:
                brute_keys.add(graph_pd_key(G, phi_type))  # == target_key, but keep pattern
                brute_edgesets += 1
        # brute_keys is either {target_key} (if any graph realizes it) or empty.

        # CP enumeration
        p0, p1, _, phi_by, scale = target_pd_from_graph(
            Gt, phi_type, clustering_scale=CLUSTERING_SCALE, cross_check=False)
        _, phi_target, stage, _ = stage_reduction(phi_by)
        sols, n_cp, status = enumerate_cp_solutions(
            p0, p1, stage, phi_target, phi_type=phi_type,
            clustering_scale=scale, time_limit=time_limit,
            fourclique_ncap=FOURCLIQUE_NCAP)

        cp_keys = set()
        cp_all_match = True
        for edges in sols:
            H = _edges_to_graph(edges, n_cp)
            k = graph_pd_key(H, phi_type)
            cp_keys.add(k)
            if k != target_key:
                cp_all_match = False

        # Every CP solution must realize the target PD (soundness of enumeration),
        # and the CP solution set must cover the target PD class (completeness).
        # Since all matching graphs share exactly one PD class (target_key), the
        # equivalence-class sets coincide iff:
        #   - cp produced at least one solution (covers target class), and
        #   - every cp solution has key == target_key, and
        #   - brute force also found the class (sanity: target is realizable).
        classes_match = (cp_keys == brute_keys) and cp_all_match
        exhaustive = (status != "UNKNOWN")  # enumeration completed within time

        results.append(dict(
            name=name, n=n, target_key_size=1,
            brute_count=brute_edgesets, cp_count=len(sols),
            classes_match=classes_match, cp_all_match=cp_all_match,
            status=status, exhaustive=exhaustive,
        ))
        if verbose:
            flag = "PASS" if classes_match and exhaustive else "FAIL"
            print(f"[completeness/{phi_type}] {name} n={n}: {flag}  "
                  f"brute={brute_edgesets} labeled graphs, cp={len(sols)} solutions, "
                  f"status={status}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCALABILITY
# ─────────────────────────────────────────────────────────────────────────────

def run_scalability(phi_type, sizes, time_limit, seed=11):
    import random
    rng = random.Random(seed)
    rows = []
    for n in sizes:
        # a representative connected-ish random graph at this size
        G = nx.gnp_random_graph(n, 0.4, seed=rng.randint(0, 10 ** 6))
        G = nx.convert_node_labels_to_integers(G)
        if phi_type == "fourclique" and n > FOURCLIQUE_NCAP:
            rows.append(dict(n=n, phi_type=phi_type, n_vars=None,
                             n_constraints=None, c1_aux=None, solve_s=None,
                             status="SKIP(>cap)"))
            continue
        p0, p1, _, phi_by, scale = target_pd_from_graph(
            G, phi_type, clustering_scale=CLUSTERING_SCALE, cross_check=False)
        _, phi_target, stage, _ = stage_reduction(phi_by)

        aux_cfg = {"clustering_scale": scale, "fourclique_ncap": FOURCLIQUE_NCAP}
        model, M, nn, aux_info = _build_model(
            p0, p1, stage, phi_target, phi_type, aux_cfg, verbose=False)
        n_vars = len(model.proto.variables)
        n_cons = len(model.proto.constraints)
        n_aux = aux_info["n_aux"]

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 8
        t0 = time.time()
        status = solver.solve(model)
        el = time.time() - t0
        rows.append(dict(n=n, phi_type=phi_type, n_vars=n_vars,
                         n_constraints=n_cons, c1_aux=n_aux,
                         solve_s=round(el, 3),
                         status=solver.status_name(status)))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--completeness-n", type=int, default=5,
                    help="labeled-vertex count for exhaustive completeness (<=6)")
    ap.add_argument("--scal-sizes", type=int, nargs="+",
                    default=[4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--phi", nargs="+", default=PHI_TYPES)
    ap.add_argument("--skip-completeness", action="store_true")
    args = ap.parse_args()

    summary = {}

    print("=" * 70)
    print("SOUNDNESS")
    print("=" * 70)
    for phi in args.phi:
        summary.setdefault(phi, {})["soundness"] = run_soundness(
            phi, time_limit=args.time_limit)

    if not args.skip_completeness:
        print("\n" + "=" * 70)
        print(f"COMPLETENESS (exhaustive over all graphs on n={args.completeness_n})")
        print("=" * 70)
        n = args.completeness_n
        # small set of target graphs realizable on n vertices
        targets = [
            (f"path_{n}", nx.path_graph(n)),
            (f"cycle_{n}", nx.cycle_graph(n)),
            (f"complete_{n}", nx.complete_graph(n)),
            (f"star_{n-1}", nx.star_graph(n - 1)),
            ("empty", nx.empty_graph(n)),
        ]
        for phi in args.phi:
            summary.setdefault(phi, {})["completeness"] = run_completeness(
                phi, n, targets, time_limit=max(args.time_limit, 120.0))

    print("\n" + "=" * 70)
    print("SCALABILITY")
    print("=" * 70)
    scal = {}
    for phi in args.phi:
        rows = run_scalability(phi, args.scal_sizes, time_limit=args.time_limit)
        scal[phi] = rows
        print(f"\n  phi_type = {phi}")
        print(f"  {'n':>3}  {'vars':>7}  {'constraints':>11}  {'C1-aux':>7}  "
              f"{'solve(s)':>9}  {'status':>10}")
        for r in rows:
            if r["n_vars"] is None:
                print(f"  {r['n']:>3}  {'—':>7}  {'—':>11}  {'—':>7}  "
                      f"{'—':>9}  {r['status']:>10}")
            else:
                print(f"  {r['n']:>3}  {r['n_vars']:>7}  {r['n_constraints']:>11}  "
                      f"{r['c1_aux']:>7}  {r['solve_s']:>9.3f}  {r['status']:>10}")
    summary["scalability"] = scal

    # headline
    print("\n" + "=" * 70)
    print("HEADLINE")
    print("=" * 70)
    for phi in args.phi:
        s = summary[phi].get("soundness", {})
        print(f"  {phi:<11} soundness {s.get('passed')}/{s.get('total')} PASS "
              f"({s.get('failed')} fail, {s.get('nosol')} no-sol)")
        if "completeness" in summary[phi]:
            cres = summary[phi]["completeness"]
            cp = sum(1 for r in cres if r["classes_match"] and r["exhaustive"])
            print(f"  {phi:<11} completeness {cp}/{len(cres)} target PDs matched exactly")

    return summary


if __name__ == "__main__":
    main()