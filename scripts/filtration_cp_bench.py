"""Benchmark eval: run every COMPATIBLE (Tier-1 polynomial + Tier-2 rational)
filtration encoding on the non-SBM benchmark test splits (comm20, planar, enzymes).

For each (scheme, dataset, graph): compute the target PD under that filtration,
solve the CP encoding, and verify by INDEPENDENT round-trip (recompute Phi on the
solved H from scratch, rebuild its PD, compare to target as a multiset).

Reports success rate + solve-time (mean/median) per (scheme, dataset), parallel to
the paper's Table 2. Dumps per-graph results to a pickle.

fourclique is skipped where C(n,4) auxiliaries are infeasible (n>NCAP); this is
logged, not silently dropped.

Usage:
  python scripts/filtration_cp_bench.py --time-limit 300 --datasets comm20 planar enzymes
"""
import argparse, os, sys, time, pickle, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import networkx as nx

from topo_gen.filtration_cp import solve_from_graph

# Load benchmark test graphs from plain-networkx pickles (avoids importing
# torch/torch_geometric just to read the splits). Either generate them from the
# PyG splits once, or use the per-graph nx graphs saved by cp_v4_dataset_eval.py.
#   <repo>/nx_dumps/{ds}_test.pkl                       — a plain list of nx.Graph
#   <repo>/dumps/cp_v4/{ds}_cpv4_tl1800s_cw120cw.pkl     — results[*]['ref_graph']
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DUMP = os.path.join(REPO_ROOT, "dumps", "cp_filtrations")


def load_nx_test(ds):
    ndp = os.path.join(REPO_ROOT, "nx_dumps", f"{ds}_test.pkl")
    if os.path.exists(ndp):
        with open(ndp, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, list) and obj and isinstance(obj[0], nx.Graph):
            return obj
    cpp = os.path.join(REPO_ROOT, "dumps", "cp_v4", f"{ds}_cpv4_tl1800s_cw120cw.pkl")
    with open(cpp, "rb") as f:
        d = pickle.load(f)
    return [r["ref_graph"] for r in d["results"]]

SCHEMES = ["degree", "sqdeg", "wedge", "triangle", "clustering", "avgneighdeg", "fourclique"]
FOURCLIQUE_NMAX = 20  # C(20,4)=4845 ok; above this, skip fourclique (infeasible aux)

R = nx.convert_node_labels_to_integers


# ---- independent Phi + PD (from scratch; never trust the solver's PD here) ----
def indep_phi(G, phi):
    G = R(G)
    if phi == "degree":      return {v: G.degree(v) for v in G}
    if phi == "sqdeg":       return {v: G.degree(v) ** 2 for v in G}
    if phi == "wedge":       return {v: G.degree(v) * (G.degree(v) - 1) for v in G}
    if phi == "triangle":    return dict(nx.triangles(G))
    if phi == "clustering":  return dict(nx.clustering(G))
    if phi == "avgneighdeg":
        return {v: (sum(G.degree(u) for u in G[v]) / G.degree(v) if G.degree(v) else 0.0) for v in G}
    if phi == "fourclique":
        import itertools
        out = {v: 0 for v in G}; adj = {v: set(G[v]) for v in G}
        for q in itertools.combinations(list(G), 4):
            if all(b in adj[a] for a, b in itertools.combinations(q, 2)):
                for v in q: out[v] += 1
        return out


def _uf(nt, et):
    parent = {}; born = {}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    ev = [(t, 0, v) for v, t in nt.items()] + [(t, 1, e) for e, t in et.items()]
    ev.sort(key=lambda x: (x[0], x[1]))
    p0 = []; p1 = 0
    for t, dim, s in ev:
        if dim == 0: parent[s] = s; born[s] = t
        else:
            u, w = s; ru, rw = find(u), find(w)
            if ru == rw: p1 += 1
            else:
                if born[ru] < born[rw]: ru, rw = rw, ru
                p0.append((born[ru], t)); parent[ru] = rw
    for r, b in born.items():
        if find(r) == r: p0.append((b, math.inf))
    return (tuple(sorted(p0)), p1)


def indep_key(G, phi):
    G = R(G); ph = indep_phi(G, phi)
    kf = lambda v: round(v, 9) if isinstance(v, float) else v
    vals = sorted(set(kf(v) for v in ph.values())); rk = {v: i for i, v in enumerate(vals)}
    nt = {v: rk[kf(ph[v])] for v in G}
    et = {(min(u, w), max(u, w)): max(nt[u], nt[w]) for u, w in G.edges()}
    return _uf(nt, et)


def run(cfg):
    os.makedirs(DUMP, exist_ok=True)
    summary = {}
    for ds in cfg.datasets:
        graphs = [R(g) for g in load_nx_test(ds)]
        if cfg.max_graphs: graphs = graphs[:cfg.max_graphs]
        print(f"\n{'='*70}\n{ds}: {len(graphs)} graphs (n {min(g.number_of_nodes() for g in graphs)}"
              f"-{max(g.number_of_nodes() for g in graphs)})\n{'='*70}")
        for phi in cfg.schemes:
            recs = []
            skipped = 0
            for idx, G in enumerate(graphs):
                n = G.number_of_nodes()
                if phi == "fourclique" and n > FOURCLIQUE_NMAX:
                    skipped += 1
                    continue
                tgt = indep_key(G, phi)
                t0 = time.perf_counter()
                try:
                    H, _, _ = solve_from_graph(G, phi_type=phi, time_limit=cfg.time_limit,
                                               verbose=False, cp_sat_workers=cfg.workers,
                                               fourclique_ncap=FOURCLIQUE_NMAX)
                except NotImplementedError:
                    H = None
                except ValueError:  # e.g. fourclique aux cap exceeded — count as skip
                    skipped += 1
                    continue
                el = time.perf_counter() - t0
                if H is None:
                    ok = False; outcome = "UNSOLVED"
                else:
                    ok = (indep_key(H, phi) == tgt)
                    outcome = "PASS" if ok else "FAIL"
                recs.append(dict(idx=idx, n=n, e=G.number_of_edges(),
                                 outcome=outcome, ok=ok, elapsed=el))
                if idx % 10 == 0 or not ok:
                    print(f"  [{phi:11s}] g{idx:3d} n={n:3d}: {outcome:8s} {el:7.2f}s", flush=True)
            n_run = len(recs)
            n_pass = sum(1 for r in recs if r["ok"])
            times = sorted(r["elapsed"] for r in recs if r["ok"])
            sr = 100.0 * n_pass / n_run if n_run else float("nan")
            mean_t = sum(times) / len(times) if times else float("nan")
            med_t = times[len(times)//2] if times else float("nan")
            summary[(ds, phi)] = dict(n_run=n_run, n_pass=n_pass, success_rate=sr,
                                      mean_t=mean_t, median_t=med_t, skipped=skipped)
            print(f"  => {phi:11s}: {n_pass}/{n_run} PASS ({sr:.1f}%)  "
                  f"mean={mean_t:.2f}s median={med_t:.2f}s  skipped={skipped}")
            with open(os.path.join(DUMP, f"{ds}_{phi}.pkl"), "wb") as f:
                pickle.dump(dict(dataset=ds, phi=phi, records=recs, summary=summary[(ds, phi)]), f)

    print(f"\n\n{'='*70}\nSUMMARY TABLE (success rate %% | median solve s)\n{'='*70}")
    print(f"  {'scheme':12s} " + " ".join(f"{d:>18s}" for d in cfg.datasets))
    for phi in cfg.schemes:
        row = f"  {phi:12s} "
        for d in cfg.datasets:
            s = summary.get((d, phi))
            if s and s["n_run"]:
                cell = f"{s['success_rate']:.0f}% / {s['median_t']:.1f}s"
                if s["skipped"]: cell += f" (skip{s['skipped']})"
            else:
                cell = "—"
            row += f"{cell:>18s} "
        print(row)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["comm20", "planar", "enzymes"])
    p.add_argument("--schemes", nargs="+", default=SCHEMES)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-graphs", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
