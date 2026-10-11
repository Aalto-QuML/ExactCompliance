"""
Truncation experiment: WHEN and HOW finite-K approximations of the REFUSED
(spectral / random-walk) filtrations fail to preserve exact PD compliance.

Schemes (finite-K surrogates, each a polynomial/rational in M so CP CAN pin it,
whose K->infinity limit is a non-encodable filtration):
    eigvec_K    x = M^K 1            (UNNORMALISED power iteration; true limit is
                                      normalised eigenvector centrality -- the norm
                                      is NOT in-class, so we pin the unnormalised
                                      vector and expect non-monotone true error)
    katz_K      x = sum_{k=1}^K a^k M^k 1     a = 0.1  (denominator 10^k)
    rw_K        x = (M^T D^-1)^K 1            (K-step random-walk occupancy)
    pagerank_K  pi = (1-a)/n sum_{k=0}^K a^k (M^T D^-1)^k 1   a = 0.85

TWO axes, never conflated:
  Axis A  encode feasibility: solve status / time / #vars / #aux / scaling D.
  Axis B  round-trip fidelity: take solved H, recompute the scheme's TRUE
          (converged, networkx) value, build H's TRUE-filtration PD, compare to
          G's TRUE-filtration PD via diagonal-padded 1-Wasserstein (TRUE_W1).
          Control: truncated-PD(H) vs truncated-PD(G) W1 (~0, that's what CP pinned).

Headline: does TRUE_W1 -> 0 as K grows, and does it reach 0 BEFORE the encoding
becomes infeasible/intractable?  Reported honestly per scheme.
"""

import argparse
import csv
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

import networkx as nx

TOPOGEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TOPOGEN_ROOT not in sys.path:
    sys.path.insert(0, TOPOGEN_ROOT)

from topo_gen import filtration_cp as fc

SCHEMES = ["eigvec_K", "katz_K", "rw_K", "pagerank_K"]


def _to_inf(p0):
    return [(b, d if d is not None else float("inf")) for b, d in p0]


def _source_graphs(seed=13):
    """A handful of structured + random small graphs, |V| in 5..8."""
    gs = [
        ("P5", nx.path_graph(5)),
        ("P6", nx.path_graph(6)),
        ("C5", nx.cycle_graph(5)),
        ("C6", nx.cycle_graph(6)),
        ("star5", nx.star_graph(5)),
        ("K4+tail", nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 0), (3, 4)])),
        ("bull", nx.bull_graph()),
        ("house", nx.house_graph()),
    ]
    rng_seeds = [11, 42, 77]
    for n in [6, 7, 8]:
        for s in rng_seeds:
            G = nx.gnp_random_graph(n, 0.45, seed=s)
            if G.number_of_edges() == 0:
                G.add_edge(0, 1)
            gs.append((f"gnp{n}_s{s}", G))
    return gs


def run_scheme(phi_type, Ks, graphs, time_limit, d_cap):
    """One row per (graph, K).  Aggregates the residual TRUE_W1 across graphs."""
    rows = []
    for name, G in graphs:
        G = nx.convert_node_labels_to_integers(G)
        tp0G, tp1G = fc.true_pd(G, phi_type)
        for K in Ks:
            # scale check / D overflow guard (Axis A pre-check)
            try:
                D = fc.truncated_scale_for(G, phi_type, K)
            except Exception as e:
                rows.append(dict(graph=name, n=G.number_of_nodes(), K=K,
                                 status="SCALE_ERR", solve_s=0.0, D=None,
                                 n_vars=None, n_aux=None, trunc_w1=None,
                                 true_w1=None, iso=None, note=str(e)[:40]))
                continue
            if D > d_cap:
                rows.append(dict(graph=name, n=G.number_of_nodes(), K=K,
                                 status="D_OVERFLOW", solve_s=0.0, D=D,
                                 n_vars=None, n_aux=None, trunc_w1=None,
                                 true_w1=None, iso=None,
                                 note=f"D>{d_cap:.0e}"))
                continue
            try:
                H, st, el, nv, na, scale, p0, p1 = fc.solve_truncated_from_graph(
                    G, phi_type, K, time_limit=time_limit, cross_check=False)
            except Exception as e:
                rows.append(dict(graph=name, n=G.number_of_nodes(), K=K,
                                 status="EXC", solve_s=0.0, D=D, n_vars=None,
                                 n_aux=None, trunc_w1=None, true_w1=None,
                                 iso=None, note=f"{type(e).__name__}:{str(e)[:30]}"))
                continue
            if H is None:
                rows.append(dict(graph=name, n=G.number_of_nodes(), K=K,
                                 status=st, solve_s=round(el, 3), D=D,
                                 n_vars=nv, n_aux=na, trunc_w1=None,
                                 true_w1=None, iso=None, note="encode-infeasible"))
                continue
            # Axis B
            tp0H, tp1H = fc.true_pd(H, phi_type)
            true_w1 = (fc.wasserstein1_pd(_to_inf(tp0G), _to_inf(tp0H))
                       + fc.wasserstein1_pd(tp1G, tp1H))
            # control (truncated)
            tphiH, _ = fc.truncated_phi_values(H, phi_type, K, scale=scale)
            p0H, p1H, _, _ = fc._stage_pd_from_phi_by_node(H, tphiH)
            trunc_w1 = (fc.wasserstein1_pd(_to_inf(p0), _to_inf(p0H))
                        + fc.wasserstein1_pd(p1, p1H))
            rows.append(dict(graph=name, n=G.number_of_nodes(), K=K,
                             status=st, solve_s=round(el, 3), D=D, n_vars=nv,
                             n_aux=na, trunc_w1=round(trunc_w1, 4),
                             true_w1=round(true_w1, 4),
                             iso=nx.is_isomorphic(G, H), note=""))
    return rows


def summarize(phi_type, rows, Ks):
    """Per-K aggregate: mean/max TRUE_W1 over graphs where encode succeeded,
    #encode-feasible, #D_overflow, mean solve time, max D."""
    print(f"\n  === {phi_type} : per-K aggregate ===")
    print(f"  {'K':>3}  {'feas':>5}  {'overflow':>8}  {'maxD':>14}  "
          f"{'mean_solve':>10}  {'mean_truncW1':>12}  {'mean_TRUE_W1':>12}  "
          f"{'max_TRUE_W1':>11}  {'#iso':>5}")
    for K in Ks:
        rk = [r for r in rows if r["K"] == K]
        feas = [r for r in rk if r["true_w1"] is not None]
        overflow = [r for r in rk if r["status"] in ("D_OVERFLOW",)]
        infeas = [r for r in rk if r["status"] in ("MODEL_INVALID", "UNKNOWN",
                                                    "INFEASIBLE")]
        maxD = max([r["D"] for r in rk if r["D"] is not None], default=0)
        if feas:
            mean_solve = sum(r["solve_s"] for r in feas) / len(feas)
            mean_tw = sum(r["trunc_w1"] for r in feas) / len(feas)
            mean_true = sum(r["true_w1"] for r in feas) / len(feas)
            max_true = max(r["true_w1"] for r in feas)
            n_iso = sum(1 for r in feas if r["iso"])
            print(f"  {K:>3}  {len(feas):>5}  {len(overflow)+len(infeas):>8}  "
                  f"{maxD:>14}  {mean_solve:>10.3f}  {mean_tw:>12.4f}  "
                  f"{mean_true:>12.4f}  {max_true:>11.4f}  {n_iso:>5}")
        else:
            print(f"  {K:>3}  {0:>5}  {len(overflow)+len(infeas):>8}  "
                  f"{maxD:>14}  {'—':>10}  {'—':>12}  {'—':>12}  {'—':>11}  {'—':>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--schemes", nargs="+", default=SCHEMES)
    ap.add_argument("--d-cap", type=float, default=1e12,
                    help="scaling-integer cap above which encoding is declared "
                         "infeasible (Axis A) without building the model")
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    args = ap.parse_args()

    graphs = _source_graphs()
    print(f"Source graphs ({len(graphs)}): "
          + ", ".join(f"{nm}(n{G.number_of_nodes()})" for nm, G in graphs))
    print(f"K sweep: {args.Ks}   d_cap={args.d_cap:.0e}   "
          f"time_limit={args.time_limit}s")

    all_rows = {}
    for phi in args.schemes:
        print("\n" + "=" * 78)
        print(f"SCHEME {phi}")
        print("=" * 78)
        rows = run_scheme(phi, args.Ks, graphs, args.time_limit, int(args.d_cap))
        all_rows[phi] = rows
        # per-row detail (compact)
        print(f"  {'graph':<10} {'n':>2} {'K':>2} {'status':<13} {'D':>14} "
              f"{'vars':>5} {'aux':>5} {'truncW1':>8} {'TRUE_W1':>8} {'iso':>4}")
        for r in rows:
            D = "" if r["D"] is None else str(r["D"])
            nv = "" if r["n_vars"] is None else r["n_vars"]
            na = "" if r["n_aux"] is None else r["n_aux"]
            tw = "" if r["trunc_w1"] is None else f"{r['trunc_w1']:.3f}"
            tr = "" if r["true_w1"] is None else f"{r['true_w1']:.3f}"
            iso = "" if r["iso"] is None else ("Y" if r["iso"] else "N")
            print(f"  {r['graph']:<10} {r['n']:>2} {r['K']:>2} {r['status']:<13} "
                  f"{D:>14} {nv:>5} {na:>5} {tw:>8} {tr:>8} {iso:>4}")
        summarize(phi, rows, args.Ks)

    # headline
    print("\n" + "=" * 78)
    print("HEADLINE — finite approximation: fail or succeed?")
    print("=" * 78)
    for phi in args.schemes:
        rows = all_rows[phi]
        # largest K that stayed encode-feasible on ANY graph
        feas_Ks = sorted({r["K"] for r in rows if r["true_w1"] is not None})
        if not feas_Ks:
            print(f"  {phi:<11}: no K encode-feasible")
            continue
        maxK = feas_Ks[-1]
        rk = [r for r in rows if r["K"] == maxK and r["true_w1"] is not None]
        mean_true = sum(r["true_w1"] for r in rk) / len(rk)
        max_true = max(r["true_w1"] for r in rk)
        # does TRUE_W1 reach 0 at ANY feasible K on ALL graphs?
        reached0 = False
        for K in feas_Ks:
            rkk = [r for r in rows if r["K"] == K and r["true_w1"] is not None]
            if rkk and all(r["true_w1"] == 0.0 for r in rkk):
                reached0 = K
                break
        # where did encoding first fail (overflow / model_invalid)?
        firstfail = None
        for K in sorted({r["K"] for r in rows}):
            rkk = [r for r in rows if r["K"] == K]
            if rkk and all(r["true_w1"] is None for r in rkk):
                firstfail = K
                break
        print(f"  {phi:<11}: feasible up to K={maxK}; "
              f"at K={maxK} mean TRUE_W1={mean_true:.3f} (max {max_true:.3f}); "
              f"TRUE_W1==0 on all graphs at K={reached0 if reached0 else 'never'}; "
              f"encoding first fully fails at K={firstfail if firstfail else '>sweep'}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "scheme", "graph", "n", "K", "status", "solve_s", "D",
                "n_vars", "n_aux", "trunc_w1", "true_w1", "iso", "note"])
            w.writeheader()
            for phi in args.schemes:
                for r in all_rows[phi]:
                    w.writerow({"scheme": phi, **r})
        print(f"\nCSV written: {args.csv}")


if __name__ == "__main__":
    main()
