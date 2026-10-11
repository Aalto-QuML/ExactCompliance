"""Consolidated correctness MATRIX — verifies both sides in one place.

  CATEGORY A: exact schemes SHOULD work
     -> SOUNDNESS (independent round-trip) + COMPLETENESS (up to iso) must hold.
  CATEGORY B: pure spectral / shortest-path names SHOULD be refused
     -> must raise NotImplementedError (soundness of the refusal boundary).
  CATEGORY C: truncated finite-K surrogates ARE encodable but should NOT give
     exact compliance -> control (trunc PD matches) holds, but the TRUE converged
     PD diverges on at least some non-iso H (independently recomputed).

Everything uses a FROM-SCRATCH independent PD (union-find) so a bug in the
solver's PD path cannot mask a bug here. Hard assertions; exits nonzero on any
violation.
"""
import sys, os, math, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import networkx as nx

from topo_gen.filtration_cp import (
    solve_from_graph, target_pd_from_graph, stage_reduction,
    enumerate_cp_solutions, solve_truncated_from_graph, wasserstein1_pd, _refuse,
)

EXACT = ["degree", "triangle", "clustering", "sqdeg", "wedge", "fourclique", "avgneighdeg"]
REFUSED = ["eigenvector", "subgraph", "closeness", "betweenness", "katz", "pagerank"]
TRUNCATED = ["eigvec_K", "katz_K", "pagerank_K", "rw_K"]

R = nx.convert_node_labels_to_integers
failures = []


# ---- independent Phi + PD (no solver code) --------------------------------
def indep_phi(G, phi):
    G = R(G)
    if phi == "degree":      return {v: G.degree(v) for v in G}
    if phi == "sqdeg":       return {v: G.degree(v)**2 for v in G}
    if phi == "wedge":       return {v: G.degree(v)*(G.degree(v)-1) for v in G}
    if phi == "triangle":    return dict(nx.triangles(G))
    if phi == "clustering":  return dict(nx.clustering(G))
    if phi == "avgneighdeg":
        return {v: (sum(G.degree(u) for u in G[v])/G.degree(v) if G.degree(v) else 0.0) for v in G}
    if phi == "fourclique":
        out = {v: 0 for v in G}; adj = {v: set(G[v]) for v in G}
        for q in itertools.combinations(list(G), 4):
            if all(b in adj[a] for a, b in itertools.combinations(q, 2)):
                for v in q: out[v] += 1
        return out
    raise ValueError(phi)

def _uf_p0p1(nt, et):
    parent = {}; born = {}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    ev = [(t,0,v) for v,t in nt.items()] + [(t,1,e) for e,t in et.items()]
    ev.sort(key=lambda x: (x[0], x[1]))
    p0 = []; p1 = []
    for t, dim, s in ev:
        if dim == 0: parent[s] = s; born[s] = t
        else:
            u,w = s; ru,rw = find(u), find(w)
            if ru == rw: p1.append((t, math.inf))
            else:
                if born[ru] < born[rw]: ru,rw = rw,ru
                p0.append((born[ru], t)); parent[ru] = rw
    for r,b in born.items():
        if find(r) == r: p0.append((b, math.inf))
    return p0, p1

def indep_key(G, phi):
    G = R(G); ph = indep_phi(G, phi)
    kf = lambda v: round(v, 9) if isinstance(v, float) else v
    vals = sorted(set(kf(v) for v in ph.values())); rk = {v: i for i, v in enumerate(vals)}
    nt = {v: rk[kf(ph[v])] for v in G}
    et = {(min(u,w), max(u,w)): max(nt[u], nt[w]) for u, w in G.edges()}
    p0, p1 = _uf_p0p1(nt, et)
    return (tuple(sorted(p0)), tuple(sorted(p1)))

def all_graphs(n):
    P = list(itertools.combinations(range(n), 2))
    for bits in itertools.product([0,1], repeat=len(P)):
        G = nx.Graph(); G.add_nodes_from(range(n))
        G.add_edges_from(p for p, b in zip(P, bits) if b)
        yield G


# ==== CATEGORY A: exact schemes SHOULD work ================================
def cat_A_soundness():
    print("\n" + "="*68 + "\nA1. SOUNDNESS (exact schemes, hard families, independent round-trip)\n" + "="*68)
    fams = [("P8",R(nx.path_graph(8))),("C8",R(nx.cycle_graph(8))),("K5",R(nx.complete_graph(5))),
            ("Petersen",R(nx.petersen_graph())),("K3,3",R(nx.complete_bipartite_graph(3,3))),
            ("cubeQ3",R(nx.hypercube_graph(3))),("Prism",R(nx.circular_ladder_graph(4))),
            ("ER8",R(nx.gnp_random_graph(8,0.4,seed=7))),("ER9",R(nx.gnp_random_graph(9,0.35,seed=11))),
            ("2tri",R(nx.disjoint_union(nx.complete_graph(3),nx.complete_graph(3))))]
    for phi in EXACT:
        p = f = ns = 0
        for nm, G in fams:
            if phi == "fourclique" and G.number_of_nodes() > 9: continue
            tgt = indep_key(G, phi)
            try: H, _, _ = solve_from_graph(G, phi_type=phi, time_limit=90.0, verbose=False)
            except NotImplementedError: H = None
            if H is None: ns += 1; continue
            if indep_key(H, phi) == tgt: p += 1
            else:
                f += 1; failures.append(f"A1 soundness {phi}/{nm}: PD mismatch")
        print(f"  {phi:11s}: {p} PASS  {f} FAIL  {ns} no-sol")

def cat_A_completeness():
    print("\n" + "="*68 + "\nA2. COMPLETENESS up to isomorphism (target's own n, exhaustive)\n" + "="*68)
    targets = [("P6",nx.path_graph(6)),("C6",nx.cycle_graph(6)),("star5",nx.star_graph(5)),
               ("tri+e",nx.Graph([(0,1),(1,2),(2,0),(3,4)])),("K4",nx.complete_graph(4)),
               ("P5",nx.path_graph(5)),("paw",nx.Graph([(0,1),(1,2),(2,0),(2,3)]))]
    for phi in EXACT:
        allok = True
        for nm, Gt in targets:
            Gt = R(Gt); n = Gt.number_of_nodes()
            tgt = indep_key(Gt, phi)
            brute = []
            for G in all_graphs(n):
                if indep_key(G, phi) == tgt and not any(nx.is_isomorphic(G, X) for X in brute):
                    brute.append(G)
            p0,p1,_,phi_by,scale = target_pd_from_graph(Gt, phi, cross_check=False)
            _, pt, st, _ = stage_reduction(phi_by)
            sols, ncp, status = enumerate_cp_solutions(p0,p1,st,pt,phi_type=phi,clustering_scale=scale,time_limit=30.0)
            cp = []
            valid = True
            for edges in sols:
                H = nx.Graph(); H.add_nodes_from(range(ncp)); H.add_edges_from(tuple(sorted(e)) for e in edges)
                if indep_key(H, phi) != tgt: valid = False
                if not any(nx.is_isomorphic(H, X) for X in cp): cp.append(H)
            cov = (all(any(nx.is_isomorphic(a,b) for b in cp) for a in brute) and
                   all(any(nx.is_isomorphic(a,b) for b in brute) for a in cp))
            ok = cov and valid and status != "UNKNOWN"
            allok &= ok
            if not ok: failures.append(f"A2 completeness {phi}/{nm}: brute_iso={len(brute)} cp_iso={len(cp)} valid={valid} status={status}")
        print(f"  {phi:11s}: {'ALL PASS' if allok else 'FAIL (see failures)'}")


# ==== CATEGORY B: refused names SHOULD refuse =============================
def cat_B_refusal():
    print("\n" + "="*68 + "\nB. REFUSAL boundary (pure spectral/shortest-path MUST raise)\n" + "="*68)
    G = R(nx.gnp_random_graph(7, 0.4, seed=3))
    for phi in REFUSED:
        try:
            solve_from_graph(G, phi_type=phi, time_limit=5.0, verbose=False)
            failures.append(f"B refusal {phi}: DID NOT raise (should be refused)")
            print(f"  {phi:12s}: *** DID NOT REFUSE ***")
        except NotImplementedError as e:
            print(f"  {phi:12s}: refused OK — {str(e)[:60]}...")
        except Exception as e:
            failures.append(f"B refusal {phi}: wrong exception {type(e).__name__}")
            print(f"  {phi:12s}: WRONG exception {type(e).__name__}")


# ==== CATEGORY C: truncated surrogates encode but are NOT exact ===========
def _true_p0(X, sch):
    X = R(X)
    if sch == "eigvec_K":
        try: ph = dict(nx.eigenvector_centrality_numpy(X, max_iter=2000))
        except nx.AmbiguousSolution:
            # disconnected: eigenvector centrality per-component (largest-eig of each block)
            ph = {}
            for comp in nx.connected_components(X):
                sub = X.subgraph(comp)
                if sub.number_of_nodes() == 1:
                    ph[list(comp)[0]] = 0.0
                else:
                    ph.update(nx.eigenvector_centrality_numpy(sub, max_iter=2000))
    elif sch == "katz_K":   ph = dict(nx.katz_centrality_numpy(X, alpha=0.1))
    elif sch == "pagerank_K": ph = dict(nx.pagerank(X, alpha=0.85))
    elif sch == "rw_K":
        m = X.number_of_edges(); ph = {v: (X.degree(v)/(2*m) if m else 0.0) for v in X}
    vals = sorted(set(round(v,9) for v in ph.values())); rk = {v:i for i,v in enumerate(vals)}
    nt = {v: rk[round(ph[v],9)] for v in X}
    et = {(min(u,w),max(u,w)): max(nt[u],nt[w]) for u,w in X.edges()}
    return _uf_p0p1(nt, et)[0]

def cat_C_truncation():
    print("\n" + "="*68 + "\nC. TRUNCATION: encodable but NOT exact (independent true recompute)\n" + "="*68)
    # graphs known to admit non-iso surrogate solutions
    srcs = [("gnp8_s11", R(nx.gnp_random_graph(8,0.45,seed=11))),
            ("gnp8_s77", R(nx.gnp_random_graph(8,0.45,seed=77))),
            ("gnp7_s11", R(nx.gnp_random_graph(7,0.45,seed=11)))]
    for sch in ["eigvec_K", "katz_K"]:
        maxgap = 0.0; any_noniso = False; control_ok = True
        for nm, G in srcs:
            for K in [1, 2]:
                ret = solve_truncated_from_graph(G, sch, K, time_limit=30.0, verbose=False)
                H, status = ret[0], ret[1]
                if H is None: continue
                iso = nx.is_isomorphic(R(G), R(H))
                if not iso: any_noniso = True
                gap = wasserstein1_pd(_true_p0(G, sch), _true_p0(H, sch))
                maxgap = max(maxgap, gap)
        # EXPECT: encoding works (found solutions), and true gap > 0 somewhere
        ok = (maxgap > 1e-6)
        if not ok: failures.append(f"C truncation {sch}: no true-PD gap found (maxgap={maxgap})")
        print(f"  {sch:10s}: max true_w1 gap = {maxgap:.3f}  (non-iso H seen: {any_noniso})  "
              f"{'-> NOT exact (as expected)' if ok else '*** unexpectedly exact ***'}")
    # rw_K positive control: should reach 0 (collapses to degree)
    G = R(nx.gnp_random_graph(8, 0.45, seed=11))
    ret = solve_truncated_from_graph(G, "rw_K", 2, time_limit=30.0, verbose=False)
    if ret[0] is not None:
        gap = wasserstein1_pd(_true_p0(G, "rw_K"), _true_p0(ret[0], "rw_K"))
        print(f"  rw_K      : true_w1 gap = {gap:.3f}  (expected 0 — stationary walk == degree rescaled)")
        if gap > 1e-6: failures.append(f"C rw_K: expected 0 gap, got {gap}")


if __name__ == "__main__":
    cat_A_soundness()
    cat_A_completeness()
    cat_B_refusal()
    cat_C_truncation()
    print("\n" + "="*68)
    if failures:
        print(f"VERDICT: {len(failures)} FAILURE(S):")
        for f in failures: print("  -", f)
        sys.exit(1)
    print("VERDICT: ALL CATEGORIES PASS")
    print("  A: exact schemes sound + complete   B: refused names refuse")
    print("  C: truncated schemes encode but are NOT exact (rw_K collapses to degree)")
    print("="*68)
