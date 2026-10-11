"""
E1 — Diversity / structural drift within a degree-PD-equivalence class.
========================================================================
Rebuttal for AC #5 / reviewer MY3i Q4.  Question: many non-isomorphic graphs
share a degree-PD -- do repeated CP / local generations produce structurally
DIFFERENT graphs, or collapse to one?  Filtration = DEGREE ONLY.

Three sampling sources of the PD-equivalence class of a target graph G:
  A. CP exhaustive enumeration (enumerate_cp_solutions, capped) -- the TRUE
     labeled class; tractable only on small graphs.  Reports if truncated.
  B. Local Alg-3 sampler (degree_filtration_pd_equivalent, seeds 0..K-1) --
     what a practical stochastic generator delivers.
  C. CP randomized (repeated solves with distinct random_seed +
     randomize_search) -- whether repeated EXACT CP spreads across the class.

Every source is DEDUPED UP TO ISOMORPHISM (bucket by weisfeiler_lehman_graph_hash,
confirm with is_isomorphic).  We report DISTINCT ISO-CLASSES, never labeled counts.

Metrics per class member (summarised as spread across the class):
  beta0 = #components, beta1 = m - n + beta0  (PD-CONSTRAINED: asserted invariant)
  avg clustering, triangle count               (NOT PD-preserved: expect spread)
  planarity rate
  spectral distance (sorted-Laplacian-eigenvalue L2, pairwise)
  pairwise GED (exact, n<=GED_NMAX, per-pair timeout) else spectral proxy (labelled)

Requires ortools / gudhi / networkx (see environment.yml). Run from the repo root:
  python scripts/filtration_cp_diversity.py
"""

import argparse
import itertools
import os
import pickle
import random
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.stdout.reconfigure(line_buffering=True)

import networkx as nx
import numpy as np
from ortools.sat.python import cp_model

from topo_gen import filtration_cp as fc
from topo_gen.degree_filtration_cp_v4 import degree_filtration_ph, ph_equal
from topo_gen.degree_filtration_pd_equivalent import degree_filtration_pd_equivalent

# Benchmark test graphs as plain nx pickles (generate once to avoid a torch import):
#   <repo>/nx_dumps/{ds}_test.pkl  or  <repo>/dumps/cp_v4/{ds}_cpv4_tl1800s_cw120cw.pkl
NX_DUMPS = os.path.join(REPO_ROOT, "nx_dumps")
CPV4_DUMPS = os.path.join(REPO_ROOT, "dumps", "cp_v4")
OUT_DIR = os.path.join(REPO_ROOT, "dumps", "diversity")
OUT_GRAPHS_DIR = os.path.join(REPO_ROOT, "dumps", "diversity_graphs")

GED_NMAX = 12          # exact GED only up to this many vertices
GED_TIMEOUT = 5.0      # seconds per exact-GED pair
GED_MAX_PAIRS = 15     # cap #pairs for exact GED (subsample)
SPEC_MAX_PAIRS = 3000  # cap #pairs for pairwise spectral distance (subsample)
B_MAX_ATTEMPTS = 2000  # Source-B: max sampler seeds to reach --K successes


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _builtin_small_targets():
    """Small structured graphs (n<=8, modest beta1) chosen so CP enumeration
    reaches OPTIMAL (truly exhaustive) -- gives the EXACT anchor for headline #1
    that the large benchmark graphs (all FEASIBLE / lower-bound) cannot.
    Deliberately spans rigid (complete/star) and non-rigid (path/tree/cycle+)."""
    return [
        nx.path_graph(6),
        nx.path_graph(7),
        nx.cycle_graph(6),
        nx.cycle_graph(7),
        nx.star_graph(5),                 # rigid (only star realizes it)
        nx.complete_graph(5),             # rigid
        nx.random_labeled_tree(7, seed=1),
        nx.random_labeled_tree(8, seed=2),
        nx.wheel_graph(6),
        nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5)]),  # tri+tail
        nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),          # diamond-ish
        nx.lollipop_graph(4, 2),
    ]


def load_targets(ds):
    """Return a list of nx.Graph targets for dataset ds, from the plain-networkx
    pickles (no torch)."""
    if ds == "small":
        return [nx.convert_node_labels_to_integers(g)
                for g in _builtin_small_targets()]
    p = os.path.join(NX_DUMPS, f"{ds}_test.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            graphs = pickle.load(f)
        return [nx.convert_node_labels_to_integers(g) for g in graphs]
    # fallback: cp_v4 dump -> ref_graph
    p2 = os.path.join(CPV4_DUMPS, f"{ds}_cpv4_tl1800s_cw120cw.pkl")
    with open(p2, "rb") as f:
        d = pickle.load(f)
    return [nx.convert_node_labels_to_integers(r["ref_graph"]) for r in d["results"]]


# ─────────────────────────────────────────────────────────────────────────────
# PD / betti helpers
# ─────────────────────────────────────────────────────────────────────────────

def betti(G):
    b0 = nx.number_connected_components(G)
    b1 = G.number_of_edges() - G.number_of_nodes() + b0
    return b0, b1


def pd_inf(p0):
    """Convert solver-format p0 (None death) to inf-death form for the sampler."""
    return [(b, d if d is not None else float("inf")) for b, d in p0]


# ─────────────────────────────────────────────────────────────────────────────
# Iso-dedup collector
# ─────────────────────────────────────────────────────────────────────────────

class IsoSet:
    """Collect graphs, deduped up to isomorphism (WL-hash bucket + is_isomorphic)."""
    def __init__(self):
        self.buckets = {}   # wl_hash -> list of representative graphs
        self.members = []   # all distinct-iso representatives

    def add(self, H):
        h = nx.weisfeiler_lehman_graph_hash(H, iterations=3)
        bucket = self.buckets.setdefault(h, [])
        for Hprev in bucket:
            if nx.is_isomorphic(H, Hprev):
                return False   # already have this iso-class
        bucket.append(H)
        self.members.append(H)
        return True

    def __len__(self):
        return len(self.members)


# ─────────────────────────────────────────────────────────────────────────────
# Sources A / B / C
# ─────────────────────────────────────────────────────────────────────────────

def source_A_enumerate(G, cap, time_limit):
    """CP exhaustive enumeration (labeled), deduped to iso-classes.
    Returns (iso_members, n_labeled, truncated, status, elapsed)."""
    p0t, p1t, _, phi_by, _ = fc.target_pd_from_graph(G, "degree", cross_check=False)
    _, phi_target, stage, _ = fc.stage_reduction(phi_by)
    t0 = time.time()
    sols, ncp, status = fc.enumerate_cp_solutions(
        p0t, p1t, stage, phi_target, phi_type="degree", clustering_scale=1,
        time_limit=time_limit, limit=cap)
    el = time.time() - t0
    # Exhaustive ONLY when CP-SAT proved it enumerated the whole space (OPTIMAL).
    # A FEASIBLE status means it stopped early -- either at the labeled cap OR at
    # the time limit -- so the class_size is a LOWER BOUND either way.
    exhaustive = (status == "OPTIMAL")
    hit_cap = (len(sols) >= cap)
    truncated = (not exhaustive)
    iso = IsoSet()
    for edges in sols:
        H = nx.Graph(); H.add_nodes_from(range(ncp)); H.add_edges_from(edges)
        iso.add(H)
    return iso.members, len(sols), truncated, status, el, hit_cap


def source_B_sampler(G, target_successes=20, max_attempts=2000):
    """Alg-3 local sampler.  Iterate seeds until ``target_successes`` compliant
    graphs are collected OR ``max_attempts`` seeds are exhausted (safety cap so
    fully-failing cycle-heavy targets don't loop forever).

    Returns (iso_members, n_success_collected, n_attempts_used, fail_reasons,
             hit_attempt_cap).  ``n_success_collected`` is the raw #successes
    (0..target_successes); dedup-to-iso means len(iso_members) may be smaller.
    The attempts-to-successes ratio is itself a sampler-difficulty signal."""
    p0, p1 = degree_filtration_ph(G)
    p0i = pd_inf(p0)
    iso = IsoSet()
    n_ok = 0
    attempts = 0
    reasons = {}
    while n_ok < target_successes and attempts < max_attempts:
        res = degree_filtration_pd_equivalent(p0i, p1, seed=attempts)
        attempts += 1
        if res.ok:
            n_ok += 1
            iso.add(nx.convert_node_labels_to_integers(res.H))
        else:
            reasons[res.reason.name] = reasons.get(res.reason.name, 0) + 1
    hit_attempt_cap = (n_ok < target_successes)
    return iso.members, n_ok, attempts, reasons, hit_attempt_cap


def source_C_cp_random(G, R, time_limit):
    """Repeated EXACT CP solves with distinct random_seed + randomize_search.
    Returns (iso_members, n_solved)."""
    p0t, p1t, _, phi_by, _ = fc.target_pd_from_graph(G, "degree", cross_check=False)
    _, phi_target, stage, _ = fc.stage_reduction(phi_by)
    aux_cfg = {"clustering_scale": 1, "fourclique_ncap": 9, "K": None}
    iso = IsoSet()
    n_solved = 0
    for rs in range(R):
        model, M, nn, _ = fc._build_model(
            p0t, p1t, stage, phi_target, "degree", aux_cfg)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.random_seed = rs
        solver.parameters.randomize_search = True
        solver.parameters.num_search_workers = 1
        st = solver.solve(model)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        n_solved += 1
        edges = [(i, j) for i in range(nn) for j in range(i + 1, nn)
                 if solver.value(M[i, j]) == 1]
        H = nx.Graph(); H.add_nodes_from(range(nn)); H.add_edges_from(edges)
        iso.add(H)
    return iso.members, n_solved


# ─────────────────────────────────────────────────────────────────────────────
# Metrics over a class
# ─────────────────────────────────────────────────────────────────────────────

def _lap_spectrum(H):
    if H.number_of_nodes() == 0:
        return np.zeros(0)
    L = nx.laplacian_matrix(H).toarray().astype(float)
    return np.sort(np.linalg.eigvalsh(L))


def class_metrics(members, target_b0, target_b1, want_ged=True):
    """Compute per-member metrics + pairwise spreads.  Returns a dict of
    summaries and a list of per-member rows."""
    rows = []
    betti_violation = []
    for H in members:
        b0, b1 = betti(H)
        if (b0, b1) != (target_b0, target_b1):
            betti_violation.append((b0, b1))
        rows.append(dict(
            n=H.number_of_nodes(), m=H.number_of_edges(), b0=b0, b1=b1,
            clustering=nx.average_clustering(H),
            triangles=sum(nx.triangles(H).values()) // 3,
            planar=nx.check_planarity(H)[0],
            spectrum=_lap_spectrum(H),
        ))

    def spread(key):
        vals = [r[key] for r in rows]
        return dict(min=min(vals), max=max(vals),
                    mean=float(np.mean(vals)), std=float(np.std(vals)),
                    n_distinct=len(set(round(v, 6) if isinstance(v, float) else v
                                       for v in vals)))

    planar_rate = sum(1 for r in rows if r["planar"]) / len(rows) if rows else 0.0

    # pairwise spectral distance.  Full coverage when small; random subsample
    # (capped) for large classes to keep runtime bounded (labelled via
    # 'subsampled').
    specs = [r["spectrum"] for r in rows]
    all_pairs = list(itertools.combinations(range(len(specs)), 2))
    subsampled = len(all_pairs) > SPEC_MAX_PAIRS
    if subsampled:
        pairs = random.Random(1).sample(all_pairs, SPEC_MAX_PAIRS)
    else:
        pairs = all_pairs
    sp_pair = []
    for i, j in pairs:
        a, b = specs[i], specs[j]
        L = max(len(a), len(b))
        a2 = np.pad(a, (0, L - len(a))); b2 = np.pad(b, (0, L - len(b)))
        sp_pair.append(float(np.linalg.norm(a2 - b2)))
    spec_summary = dict(
        mean=float(np.mean(sp_pair)) if sp_pair else 0.0,
        max=float(np.max(sp_pair)) if sp_pair else 0.0,
        n_pairs=len(sp_pair), subsampled=subsampled)

    # pairwise GED: exact if small, else spectral proxy
    ged_summary = dict(kind="none", mean=None, max=None, n_pairs=0)
    n = rows[0]["n"] if rows else 0
    if len(members) >= 2:
        if want_ged and n <= GED_NMAX:
            pairs = list(itertools.combinations(range(len(members)), 2))
            random.Random(0).shuffle(pairs)
            pairs = pairs[:GED_MAX_PAIRS]
            geds = []
            for i, j in pairs:
                g = nx.graph_edit_distance(members[i], members[j],
                                           timeout=GED_TIMEOUT)
                if g is not None:
                    geds.append(g)
            if geds:
                ged_summary = dict(kind="exact", mean=float(np.mean(geds)),
                                   max=float(np.max(geds)), n_pairs=len(geds))
            else:
                ged_summary = dict(kind="exact-timeout", mean=None, max=None,
                                   n_pairs=0)
        else:
            # proxy = spectral L2 distance (labelled as proxy)
            ged_summary = dict(kind="spectral-proxy", mean=spec_summary["mean"],
                               max=spec_summary["max"],
                               n_pairs=spec_summary["n_pairs"])

    return dict(
        class_size=len(members),
        betti_invariant=(len(betti_violation) == 0),
        betti_violations=betti_violation,
        clustering=spread("clustering"),
        triangles=spread("triangles"),
        planar_rate=planar_rate,
        spectral=spec_summary,
        ged=ged_summary,
    ), rows


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def process_target(G, idx, do_A, cap, K, R, time_limit):
    n, m = G.number_of_nodes(), G.number_of_edges()
    tb0, tb1 = betti(G)
    # target's degree persistence diagram (p0 uses None for essential deaths)
    tp0, tp1 = degree_filtration_ph(G)
    rec = dict(idx=idx, n=n, m=m, target_b0=tb0, target_b1=tb1,
               target_p0=tp0, target_p1=tp1)

    out = {"target": G}

    if do_A:
        memA, nlabA, truncA, statA, elA, hitcap = source_A_enumerate(
            G, cap, time_limit)
        mA, rowsA = class_metrics(memA, tb0, tb1)
        rec["A"] = dict(iso_classes=len(memA), n_labeled=nlabA,
                        truncated=truncA, hit_cap=hitcap, status=statA,
                        elapsed=round(elA, 2), metrics=mA)
        out["A_rows"] = rowsA
        out["A_members"] = memA

    memB, okB, attemptsB, reasonsB, capB = source_B_sampler(
        G, target_successes=K, max_attempts=B_MAX_ATTEMPTS)
    mB, rowsB = class_metrics(memB, tb0, tb1) if memB else (None, [])
    rec["B"] = dict(iso_classes=len(memB), n_success=okB,
                    n_attempts=attemptsB, hit_attempt_cap=capB,
                    fail_reasons=reasonsB, metrics=mB)
    out["B_rows"] = rowsB
    out["B_members"] = memB

    memC, solvedC = source_C_cp_random(G, R, time_limit)
    mC, rowsC = class_metrics(memC, tb0, tb1) if memC else (None, [])
    rec["C"] = dict(iso_classes=len(memC), n_solved=solvedC, metrics=mC)
    out["C_rows"] = rowsC
    out["C_members"] = memC

    return rec, out


def build_graph_corpus(ds, rec, out):
    """One row per DISTINCT ISO-CLASS per source, storing the representative
    graph as an edge-list, tagged for later metric computation WITHOUT
    regenerating.  Returns a list of dicts."""
    entries = []
    for src in ("A", "B", "C"):
        members = out.get(f"{src}_members")
        if not members:
            continue
        for cid, H in enumerate(members):
            b0, b1 = betti(H)
            entries.append(dict(
                dataset=ds,
                target_idx=rec["idx"],
                target_n=rec["n"],
                target_m=rec["m"],
                target_beta0=rec["target_b0"],
                target_beta1=rec["target_b1"],
                target_p0=rec["target_p0"],      # target degree-PD (H0)
                target_p1=rec["target_p1"],      # target degree-PD (H1)
                source=src,
                iso_class_id=cid,
                n=H.number_of_nodes(),
                m=H.number_of_edges(),
                beta0=b0,
                beta1=b1,
                edges=sorted(map(tuple, (sorted(e) for e in H.edges()))),
                wl_hash=nx.weisfeiler_lehman_graph_hash(H, iterations=3),
            ))
    return entries


def fmt_spread(s):
    if s is None:
        return "n/a"
    return f"[{s['min']}..{s['max']}] mean={s['mean']:.3f} std={s['std']:.3f} #d={s['n_distinct']}"


def print_target(rec):
    print(f"\n  target#{rec['idx']} n={rec['n']} m={rec['m']} "
          f"beta0={rec['target_b0']} beta1={rec['target_b1']}")
    for src in ("A", "B", "C"):
        if src not in rec:
            continue
        r = rec[src]
        mt = r["metrics"]
        if src == "A":
            if r["status"] == "OPTIMAL":
                tag = "EXHAUSTIVE(exact)"
            elif r.get("hit_cap"):
                tag = ">=cap LOWER-BOUND"
            else:
                tag = ">=timelimit LOWER-BOUND"
            head = (f"    [A cp-enum ] iso-classes={r['iso_classes']} "
                    f"(labeled={r['n_labeled']}, {tag}, "
                    f"{r['status']}, {r['elapsed']}s)")
        elif src == "B":
            capnote = " GAVE-UP@2000-attempts" if r.get("hit_attempt_cap") else ""
            head = (f"    [B sampler ] iso-classes={r['iso_classes']} "
                    f"(successes={r['n_success']} in {r['n_attempts']} attempts"
                    f"{capnote}, fails {r['fail_reasons']})")
        else:
            head = (f"    [C cp-rand ] iso-classes={r['iso_classes']} "
                    f"(solved {r['n_solved']})")
        print(head)
        if mt is None:
            print("        (no members)")
            continue
        binv = "INVARIANT ✓" if mt["betti_invariant"] else f"VIOLATED {mt['betti_violations']}"
        print(f"        beta0/beta1: {binv}")
        print(f"        clustering : {fmt_spread(mt['clustering'])}")
        print(f"        triangles  : {fmt_spread(mt['triangles'])}")
        print(f"        planar_rate: {mt['planar_rate']:.2f}")
        print(f"        spectral   : mean={mt['spectral']['mean']:.3f} "
              f"max={mt['spectral']['max']:.3f} ({mt['spectral']['n_pairs']} pairs)")
        g = mt["ged"]
        gv = "n/a" if g["mean"] is None else f"mean={g['mean']:.2f} max={g['max']:.2f}"
        print(f"        GED({g['kind']}): {gv} ({g['n_pairs']} pairs)")


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return 0
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def aggregate(records, ds, b_target=20):
    print("\n" + "=" * 78)
    print(f"AGGREGATE — {ds}")
    print("=" * 78)
    for src in ("A", "B", "C"):
        recs = [r for r in records if src in r and r[src]["metrics"] is not None]
        if not recs:
            print(f"  [{src}] no data")
            continue
        sizes = [r[src]["iso_classes"] for r in recs]
        rigid = [s for s in sizes if s <= 1]
        nonrigid = [s for s in sizes if s > 1]

        # ── HARD CHECK: beta0/beta1 violations counted as FAILURES per source ──
        viol_recs = [r for r in recs
                     if not r[src]["metrics"]["betti_invariant"]]
        n_viol = len(viol_recs)

        # clustering / triangle spread among non-rigid
        nr_recs = [r for r in recs if r[src]["iso_classes"] > 1]
        cl_std = [r[src]["metrics"]["clustering"]["std"] for r in nr_recs]
        tri_d = [r[src]["metrics"]["triangles"]["n_distinct"] for r in nr_recs]
        plan = [r[src]["metrics"]["planar_rate"] for r in nr_recs]
        spec = [r[src]["metrics"]["spectral"]["mean"] for r in nr_recs]
        ged = [r[src]["metrics"]["ged"]["mean"] for r in nr_recs
               if r[src]["metrics"]["ged"]["mean"] is not None]

        # ── Truncation-aware median (Source A only carries a truncated flag) ──
        if src == "A":
            completed = [r for r in recs if not r[src]["truncated"]]
            truncated = [r for r in recs if r[src]["truncated"]]
            comp_nonrigid = [r[src]["iso_classes"] for r in completed
                             if r[src]["iso_classes"] > 1]
            all_nonrigid = [r[src]["iso_classes"] for r in recs
                            if r[src]["iso_classes"] > 1]
            med_completed = _median(comp_nonrigid)
            med_all = _median(all_nonrigid)
            print(f"  [A cp-enum] {len(recs)} targets "
                  f"({len(completed)} EXHAUSTIVE/OPTIMAL, "
                  f"{len(truncated)} lower-bound[cap or time-limit])")
            print(f"        rigid(class==1): {len(rigid)}/{len(recs)} "
                  f"({100*len(rigid)/len(recs):.0f}%)")
            if comp_nonrigid:
                print(f"        non-rigid median iso-classes: "
                      f"EXHAUSTIVE-only={med_completed} (EXACT anchor, "
                      f"{len(comp_nonrigid)} targets) | "
                      f"ALL(lower-bounds incl.)={med_all}")
            else:
                print(f"        non-rigid median iso-classes: "
                      f"EXACT anchor UNAVAILABLE (no target enumerated to "
                      f"OPTIMAL -- all are LOWER BOUNDS) | "
                      f"ALL(lower-bounds)={med_all}")
            if truncated:
                print(f"        NOTE: {len(truncated)} target(s) did NOT exhaust "
                      f"(FEASIBLE -- hit the labeled cap or the time limit); "
                      f"class_size is a LOWER BOUND, and iso-dedup on a partial "
                      f"labeled set can itself undercount iso-classes.")
        else:
            med = _median(nonrigid)
            print(f"  [{src}] {len(recs)} targets | "
                  f"rigid(class==1): {len(rigid)}/{len(recs)} "
                  f"({100*len(rigid)/len(recs):.0f}%) | "
                  f"non-rigid median iso-classes = {med} "
                  f"(range {min(sizes)}..{max(sizes)})")
            if src == "B":
                # Source-B difficulty signal: successes collected & attempts used.
                allB = [r["B"] for r in records if "B" in r]
                gaveup = [b for b in allB if b.get("hit_attempt_cap")]
                reached = [b for b in allB if not b.get("hit_attempt_cap")]
                att_reached = [b["n_attempts"] for b in reached]
                print(f"        sampler difficulty: {len(reached)}/{len(allB)} "
                      f"targets reached {b_target} successes "
                      f"(median {int(_median(att_reached)) if att_reached else 0} "
                      f"attempts), {len(gaveup)}/{len(allB)} GAVE UP at "
                      f"{B_MAX_ATTEMPTS} attempts "
                      f"(successes collected there: "
                      f"{[b['n_success'] for b in gaveup]})")

        # ── beta0/beta1 hard check output ──
        if n_viol == 0:
            print(f"        beta0/beta1 HARD CHECK: 0 violations across all "
                  f"members of all {len(recs)} targets ✓")
        else:
            print(f"        !!! beta0/beta1 VIOLATIONS: {n_viol}/{len(recs)} "
                  f"targets have a member with wrong (beta0,beta1) — BUG !!!")
            ex = viol_recs[0]
            print(f"            example target#{ex['idx']} "
                  f"target(b0,b1)=({ex['target_b0']},{ex['target_b1']}) "
                  f"violations={ex[src]['metrics']['betti_violations'][:3]}")

        if nr_recs:
            print(f"        non-rigid drift: clustering std mean="
                  f"{np.mean(cl_std):.3f} | triangle #distinct mean="
                  f"{np.mean(tri_d):.1f} | planar_rate mean={np.mean(plan):.2f}")
            print(f"        pairwise spectral mean={np.mean(spec):.3f}"
                  + (f" | GED mean={np.mean(ged):.2f}" if ged else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["comm20"])
    ap.add_argument("--max-targets", type=int, default=None,
                    help="cap #targets per dataset (default all)")
    ap.add_argument("--target-indices", type=int, nargs="+", default=None,
                    help="explicit target indices to process (overrides "
                         "--max-targets); preserves original dataset index")
    ap.add_argument("--cap", type=int, default=2000, help="Source-A enum cap")
    ap.add_argument("--K", type=int, default=20,
                    help="Source-B TARGET #successes to collect (iterate seeds "
                         "until this many compliant graphs or 2000 attempts)")
    ap.add_argument("--R", type=int, default=20, help="Source-C CP-random solves")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--no-A", action="store_true",
                    help="skip exhaustive enumeration (for planar/enzymes)")
    ap.add_argument("--A-nmax", type=int, default=20,
                    help="only run Source A when n <= this (enum tractable)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_GRAPHS_DIR, exist_ok=True)

    for ds in args.datasets:
        all_targets = load_targets(ds)
        if args.target_indices is not None:
            indexed = [(i, all_targets[i]) for i in args.target_indices]
        elif args.max_targets:
            indexed = list(enumerate(all_targets[:args.max_targets]))
        else:
            indexed = list(enumerate(all_targets))
        print("=" * 78)
        print(f"DATASET {ds}: {len(indexed)} targets, "
              f"n range {min(g.number_of_nodes() for _, g in indexed)}.."
              f"{max(g.number_of_nodes() for _, g in indexed)}")
        print(f"Sources: A(cap={args.cap}{',SKIP' if args.no_A else ''}) "
              f"B(target={args.K} successes, cap {B_MAX_ATTEMPTS} attempts) "
              f"C(R={args.R})")
        print("=" * 78)

        dump_path = os.path.join(OUT_DIR, f"{ds}_diversity.pkl")
        gpath = os.path.join(OUT_GRAPHS_DIR, f"{ds}_graphs.pkl")

        def _flush(records, dump_rows, graph_corpus, final=False):
            """Write both dumps.  Called after EVERY target so a kill preserves
            all completed targets (large-n runs can be killed mid-sweep)."""
            with open(dump_path, "wb") as f:
                pickle.dump(dict(dataset=ds, records=records, rows=dump_rows,
                                 config=vars(args), complete=final), f)
            by_src = {}
            for e in graph_corpus:
                by_src[e["source"]] = by_src.get(e["source"], 0) + 1
            with open(gpath, "wb") as f:
                pickle.dump(dict(dataset=ds, graphs=graph_corpus,
                                 config=vars(args), complete=final), f)
            return by_src

        records = []
        dump_rows = []
        graph_corpus = []
        for idx, G in indexed:
            do_A = (not args.no_A) and (G.number_of_nodes() <= args.A_nmax)
            rec, out = process_target(G, idx, do_A, args.cap, args.K, args.R,
                                      args.time_limit)
            print_target(rec)
            records.append(rec)
            dump_rows.append(dict(idx=idx, record=rec,
                                  A_rows=out.get("A_rows"),
                                  B_rows=out.get("B_rows"),
                                  C_rows=out.get("C_rows")))
            graph_corpus.extend(build_graph_corpus(ds, rec, out))
            # incremental checkpoint after each target
            _flush(records, dump_rows, graph_corpus, final=False)

        aggregate(records, ds, b_target=args.K)

        by_src = _flush(records, dump_rows, graph_corpus, final=True)
        print(f"\n  dumped metrics -> {dump_path}")
        print(f"  dumped {len(graph_corpus)} iso-distinct graphs "
              f"(by source {by_src}) -> {gpath}")


if __name__ == "__main__":
    main()
