"""Persistence diagram computation for vertex-edge filtrations.

Two implementations:
- persistence_diagrams : custom Union-Find (no external dependency)
- gudhi_persistence    : via gudhi SimplexTree (cross-check / ground-truth)

Helper: check_match compares the two outputs.
"""

import math

import gudhi


# ---------------------------------------------------------------------------
# Union-Find implementation
# ---------------------------------------------------------------------------

class _UF:
    """Union-Find with birth tracking and member lists.

    Elder rule: on merge, the component with the earlier birth time survives;
    the younger one (larger birth) dies.  Ties in birth time are broken by
    keeping rx as survivor (arbitrary but consistent within one run).

    Public attributes
    -----------------
    born    : dict  root -> birth time of that component
    members : dict  root -> list of all nodes in that component
    """

    def __init__(self):
        self.parent  = {}
        self.born    = {}   # representative -> birth time of that component
        self.members = {}   # representative -> [node, ...]

    def add(self, x, birth):
        if x not in self.parent:
            self.parent[x]  = x
            self.born[x]    = birth
            self.members[x] = [x]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        """Merge the components containing x and y.

        Elder rule: the component with the smaller birth time survives.

        Returns
        -------
        dying_birth  : birth time of the younger (dying) component,
                       or None if x and y are already in the same component.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return None  # same component → cycle

        # Elder rule: smaller born time survives → ry is the elder (survivor)
        if self.born[rx] < self.born[ry]:
            rx, ry = ry, rx   # ry is now the elder, rx is the younger (dies)

        dying_birth      = self.born[rx]
        self.parent[rx]  = ry
        self.members[ry] = self.members[ry] + self.members[rx]
        del self.born[rx]
        del self.members[rx]

        return dying_birth


def persistence_diagrams(node_time: dict, edge_time: dict) -> tuple[list, list]:
    """Compute H0 and H1 persistence diagrams for a vertex-edge filtration.

    Thin wrapper over persistence_diagrams_tracked that drops provenance, so
    callers comparing bare (birth, death) tuples — notably check_match — keep
    working unchanged.

    Returns
    -------
    pd0 : list of (birth, death) pairs for H0
          death=inf means the component survives to the end
    pd1 : list of (birth, inf) pairs for H1
          (in a 1-complex all H1 classes persist forever)
    """
    pd0, pd1 = persistence_diagrams_tracked(node_time, edge_time)
    return ([(b, d) for b, d, _, _ in pd0], [(b, d) for b, d, _, _ in pd1])


def persistence_diagrams_tracked(
    node_time: dict, edge_time: dict
) -> tuple[list, list]:
    """H0/H1 persistence diagrams, each point carrying the simplex that set it.

    Under a degree filtration every filtration value is a node degree, so the
    provenance recorded here is what lets a differentiable re-evaluation
    substitute a soft degree for each birth/death coordinate.

    Parameters
    ----------
    node_time : dict  node -> int filtration time
    edge_time : dict  (u, v) -> int filtration time

    Returns
    -------
    pd0 : list of (birth, death, birth_src, death_src)
    pd1 : list of (birth, inf, birth_src, None)

    Sources are ('node', v) or ('edge', (u, v)); death_src is None when the
    death is infinite. An ('edge', e) source means the coordinate equals
    max(node_time[u], node_time[v]) — see filtrations._edge_times_max.
    """
    events = []
    for v, t in node_time.items():
        events.append((t, 0, v))
    for e, t in edge_time.items():
        events.append((t, 1, e))
    events.sort(key=lambda x: (x[0], x[1]))

    uf = _UF()
    pd0 = []
    pd1 = []
    # Root -> the node whose insertion gave that component its birth time.
    birth_src = {}

    for t, dim, simplex in events:
        if dim == 0:
            uf.add(simplex, birth=t)
            birth_src[simplex] = simplex
        else:
            u, v = simplex
            ru, rv = uf.find(u), uf.find(v)
            dying_birth = uf.union(u, v)
            if dying_birth is None:
                pd1.append((t, float("inf"), ("edge", simplex), None))
            else:
                # union() applies the elder rule; the dying root is whichever
                # of ru/rv is no longer its own representative afterwards.
                dying_root = ru if uf.find(ru) != ru else rv
                src = birth_src.get(dying_root, dying_root)
                pd0.append((dying_birth, t, ("node", src), ("edge", simplex)))

    for root, birth in uf.born.items():
        src = birth_src.get(root, root)
        pd0.append((birth, float("inf"), ("node", src), None))

    return pd0, pd1


# ---------------------------------------------------------------------------
# Gudhi cross-check
# ---------------------------------------------------------------------------

_SENTINEL = 10 ** 9


def _snap(x):
    """Round a shifted filtration value back to the nearest integer."""
    if x == float("inf"):
        return float("inf")
    return int(math.floor(x + 0.5))


def gudhi_persistence(node_time: dict, edge_time: dict) -> tuple[list, list]:
    """Compute H0 and H1 persistence via gudhi SimplexTree.

    Two fixes applied vs. naive gudhi usage:
    1. Nodes inserted at (t - 0.5) to enforce dimension-first ordering at
       equal filtration times; values snapped back to integers afterward.
    2. A dummy 2-simplex at filtration=inf lifts the complex to dim-2 so
       gudhi computes H1 (it silently omits H_k when max simplex dim == k).
       The resulting spurious (inf, inf) H0 point is filtered out.
    """
    st = gudhi.SimplexTree()
    for v, t in node_time.items():
        st.insert([v], filtration=float(t) - 0.5)
    for (u, v), t in edge_time.items():
        st.insert([u, v], filtration=float(t))
    st.insert([_SENTINEL, _SENTINEL + 1, _SENTINEL + 2], filtration=float("inf"))

    st.compute_persistence(min_persistence=-1)
    pairs = st.persistence()

    pd0 = [(_snap(b), _snap(d)) for dim, (b, d) in pairs
           if dim == 0 and b != float("inf")]
    pd1 = [(_snap(b), _snap(d)) for dim, (b, d) in pairs if dim == 1]
    return pd0, pd1


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def check_match(pd0_ours, pd1_ours, pd0_other, pd1_other, verbose=True) -> bool:
    """Return True if both PDs match (order-independent)."""
    ok0 = sorted(pd0_ours) == sorted(pd0_other)
    ok1 = sorted(pd1_ours) == sorted(pd1_other)

    if verbose:
        print(f"H0 match : {'✓ PASS' if ok0 else '✗ FAIL'}"
              f"   (ours {len(pd0_ours)} pts, other {len(pd0_other)} pts)")
        print(f"H1 match : {'✓ PASS' if ok1 else '✗ FAIL'}"
              f"   (ours {len(pd1_ours)} pts, other {len(pd1_other)} pts)")
        if not ok0:
            s_ours  = set(map(tuple, sorted(pd0_ours)))
            s_other = set(map(tuple, sorted(pd0_other)))
            print("  H0 only in ours  :", s_ours  - s_other)
            print("  H0 only in other :", s_other - s_ours)
        if not ok1:
            s_ours  = set(map(tuple, sorted(pd1_ours)))
            s_other = set(map(tuple, sorted(pd1_other)))
            print("  H1 only in ours  :", s_ours  - s_other)
            print("  H1 only in other :", s_other - s_ours)

    return ok0 and ok1
