"""Algebraic function extraction (backward rewriting) for gate-level adder trees (T1.4).

Proves that a combinational gate netlist computes a given word-level polynomial modulo 2^w, for every
input: start from the output-side polynomial Σ_i 2^i (o_i + p_i) and substitute every gate by its exact
multilinear polynomial over Z (NOT 1-a, AND ab, OR a+b-ab, XOR a+b-2ab, XNOR 1-a-b+2ab, MUX, ANDNOT,
ORNOT), in reverse topological order, reducing coefficients mod 2^w and x² = x. The result is a
polynomial in the primary inputs only, compared coefficient by coefficient with the specification
polynomial. Since multilinear polynomials over Z/2^w are a canonical form for functions {0,1}^n -> Z/2^w,
equality of the normal forms is a complete proof of `outputs ≡ spec (mod 2^w)` (Yu/Ciesielski "function
extraction"; Kaufmann/Biere AMulet use the same rewriting).

Input is the JSON netlist written by Yosys (`techmap; opt_clean; write_json`) of the *RTL file itself*, so
the proof does not depend on the generator's in-memory netlist.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

Poly = dict[frozenset, int]


def _mul(p: Poly, q: Poly, mod: int) -> Poly:
    out: dict[frozenset, int] = defaultdict(int)
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            out[m1 | m2] = (out[m1 | m2] + c1 * c2) % mod
    return {m: c for m, c in out.items() if c}


def _gate_poly(kind: str, ins: dict[str, Poly], mod: int) -> Poly:
    one: Poly = {frozenset(): 1}

    def add(*terms: tuple[int, Poly]) -> Poly:
        out: dict[frozenset, int] = defaultdict(int)
        for k, p in terms:
            for m, c in p.items():
                out[m] = (out[m] + k * c) % mod
        return {m: c for m, c in out.items() if c}

    a, b = ins.get("A"), ins.get("B")
    if kind == "$_NOT_":
        return add((1, one), (-1, a))
    if kind == "$_BUF_":
        return a
    ab = _mul(a, b, mod) if b is not None else None
    if kind == "$_AND_":
        return ab
    if kind == "$_OR_":
        return add((1, a), (1, b), (-1, ab))
    if kind == "$_XOR_":
        return add((1, a), (1, b), (-2, ab))
    if kind == "$_XNOR_":
        return add((1, one), (-1, a), (-1, b), (2, ab))
    if kind == "$_NAND_":
        return add((1, one), (-1, ab))
    if kind == "$_NOR_":
        return add((1, one), (-1, a), (-1, b), (1, ab))
    if kind == "$_ANDNOT_":  # A & ~B
        return add((1, a), (-1, ab))
    if kind == "$_ORNOT_":  # A | ~B
        return add((1, one), (-1, b), (1, ab))  # 1 - (1 - a) b
    if kind == "$_MUX_":  # S ? B : A
        s = ins["S"]
        return add((1, a), (1, _mul(s, b, mod)), (-1, _mul(s, a, mod)))
    raise ValueError(f"unsupported gate {kind}")


def _kahn_reverse(nodes, preds) -> list:
    """Consumers-first order of `nodes` given preds[n] (nodes n depends on, restricted to `nodes`)."""
    users: dict = defaultdict(int)
    for n in nodes:
        for v in preds[n]:
            users[v] += 1
    ready = sorted(n for n in nodes if users[n] == 0)
    order = []
    while ready:
        n = ready.pop()
        order.append(n)
        for v in preds[n]:
            users[v] -= 1
            if users[v] == 0:
                ready.append(v)
    assert len(order) == len(nodes), "combinational loop"
    return order


def _rewrite_order(driver: dict, group: dict) -> list:
    members: dict = defaultdict(list)
    for g in driver:
        members[group[g]].append(g)
    gpred: dict = defaultdict(set)
    for g, (_, ins) in driver.items():
        for v in ins.values():
            if v in driver and group[v] != group[g]:
                gpred[group[g]].add(group[v])
    order = []
    for grp in _kahn_reverse(list(members), {k: gpred[k] for k in members}):
        mem = set(members[grp])
        preds = {g: {v for v in driver[g][1].values() if v in mem} for g in mem}
        order += _kahn_reverse(list(mem), preds)
    return order


def extract(netlist_json: Path, top: str, out_terms: list[tuple[str, int, int]], w: int) -> tuple[Poly, dict]:
    """Rewrite Σ k·2^e·bit over the listed output ports back to primary inputs.

    out_terms: (port, bit index, weight) - the output-side polynomial. Returns (polynomial over input-bit
    variables named "<port>[<i>]", stats).
    """
    mod = 1 << w
    js = json.loads(Path(netlist_json).read_text())
    mod_js = js["modules"][top]
    name_of: dict = {}
    for pname, p in mod_js["ports"].items():
        if p["direction"] == "input":
            for i, bit in enumerate(p["bits"]):
                name_of[bit] = f"{pname}[{i}]"
    driver: dict = {}
    group: dict = {}
    for cname, c in mod_js["cells"].items():
        outs = [k for k, d in c["port_directions"].items() if d == "output"]
        assert len(outs) == 1, cname
        bit = c["connections"][outs[0]][0]
        driver[bit] = (c["type"], {k: v[0] for k, v in c["connections"].items() if k != outs[0]})
        src = c.get("attributes", {}).get("src", cname + ":0")  # "<file>:<line>.<col>-<line>.<col>"
        f, pos = src.rsplit(":", 1)
        group[bit] = f"{f}:{pos.split('.')[0]}"

    def leaf(bit) -> Poly:
        if bit in ("0", "x", "z"):
            return {}
        if bit == "1":
            return {frozenset(): 1}
        return {frozenset([bit]): 1}

    # output polynomial
    poly: dict[frozenset, int] = defaultdict(int)
    for port, i, k in out_terms:
        bit = mod_js["ports"][port]["bits"][i]
        for m, c in leaf(bit).items():
            poly[m] = (poly[m] + k * c) % mod
    poly = {m: c for m, c in poly.items() if c}

    # Rewrite order: gates are grouped by their RTL source line (one generated line = one FA/HA cell), and
    # groups are rewritten consumers-first (Kahn on the reversed group graph), all gates of a group together,
    # so the sum and carry of a cell are substituted back to back and the polynomial stays near-linear.
    order = _rewrite_order(driver, group)
    peak = len(poly)
    for g in order:  # consumers before drivers
        kind, ins = driver[g]
        hit = [m for m in poly if g in m]
        if not hit:
            continue
        gp = _gate_poly(kind, {k: leaf(v) for k, v in ins.items()}, mod)
        new: dict[frozenset, int] = defaultdict(int, {m: c for m, c in poly.items() if g not in m})
        for m in hit:
            c = poly[m]
            rest = m - {g}
            for gm, gc in gp.items():
                nm = rest | gm
                new[nm] = (new[nm] + c * gc) % mod
        poly = {m: c for m, c in new.items() if c}
        peak = max(peak, len(poly))
    named = {frozenset(name_of.get(v, str(v)) for v in m): c for m, c in poly.items()}
    return named, {
        "gates": len(driver),
        "rewritten": len(order),
        "peak_monomials": peak,
        "final_monomials": len(poly),
    }


def poly_spec(terms: list[tuple[tuple[str, ...], int]], w: int) -> Poly:
    """Σ k·Π vars (mod 2^w) as a multilinear polynomial; terms: (variable names "<port>[<i>]", weight).
    An empty tuple is the constant monomial."""
    mod = 1 << w
    out: dict[frozenset, int] = defaultdict(int)
    for vs, k in terms:
        key = frozenset(vs)
        out[key] = (out[key] + k) % mod
    return {m: c for m, c in out.items() if c}
