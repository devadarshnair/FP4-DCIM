"""Column compressor-tree generator: one netlist -> (a) SystemVerilog, (b) bit-level evaluator with
per-net toggle counts. Because both come from the same netlist, the RTL and the L2 power proxy are
structurally identical by construction.

Supported leaf encodings (PLAN v2 WP-24):
    "tc"        two's-complement products, sign-extension elimination (invert MSB + constant)
    "oc"        variant B: sign-magnitude leaf delivered as ones' complement (m XOR s) + s at the LSB,
                with sign-extension elimination - the baseline
    "unsigned"  unsigned inputs (w_in = 1 is the bit-serial baseline's 32-input popcount compressor)
    "dual"      variant C (the blueprint's dual-polarity tree): positive and negative product magnitudes
                (w_in-1 bits each) go to two unsigned trees; the carry-save outputs are combined as
                pos - neg (inverted negative rows + constant) - see build_dual_tree

Column structure (docs/spec/numerics.md): rows 0..15 -> 16:2 Dadda CSA, rows 16..31 -> 16:2 Dadda CSA,
then (MX) a 4:2 merge to one carry-save pair. NVFP4 uses the two halves separately. Outputs are the
carry-save vectors (the pipeline register sits here); the CPA is a separate stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DADDA = [2, 3, 4, 6, 9, 13, 19, 28, 42, 63, 94]


@dataclass
class Cell:
    kind: str  # "FA" | "HA"
    ins: tuple[str, ...]
    sum: str
    carry: str


@dataclass
class Netlist:
    name: str
    n_inputs: int
    w_in: int
    scheme: str
    out_width: int
    inputs: list[str] = field(default_factory=list)  # input bit nets, row-major: in_r_b
    cells: list[Cell] = field(default_factory=list)
    outputs: dict[str, list[str]] = field(default_factory=dict)  # name -> nets LSB first ("1"/"0" consts)
    const_bits: list[int] = field(default_factory=list)
    #: T1.4 compositional formal: slice name -> [start, end) range of ``cells`` (construction order), and
    #: internal boundary rows (name -> nets LSB first) that are not module ports. Not used by emit_sv.
    slices: dict[str, tuple[int, int]] = field(default_factory=dict)
    rows: dict[str, list[str]] = field(default_factory=dict)
    _n: int = 0

    def new(self) -> str:
        self._n += 1
        return f"n{self._n}"

    @property
    def n_fa(self) -> int:
        return sum(c.kind == "FA" for c in self.cells)

    @property
    def n_ha(self) -> int:
        return sum(c.kind == "HA" for c in self.cells)


def _signed_width(n: int, w_in: int, scheme: str, max_abs: int | None = None) -> int:
    """Width of an exact sum of n inputs. ``max_abs`` bounds |input| (e.g. 144 for FP4 products);
    default is the full w_in-bit range."""
    if scheme == "unsigned":
        return int(n * (max_abs if max_abs is not None else (1 << w_in) - 1)).bit_length()
    max_mag = max_abs if max_abs is not None else (1 << (w_in - 1))
    return int(n * max_mag).bit_length() + 1


def _dadda(nl: Netlist, cols: list[list[str]]) -> list[list[str]]:
    cols = [list(c) for c in cols]
    height = max((len(c) for c in cols), default=0)
    for d in reversed([d for d in DADDA if d < height]):
        i = 0
        while i < len(cols):
            while len(cols[i]) > d:
                if i + 1 == len(cols):
                    cols.append([])
                s, c = nl.new(), nl.new()
                if len(cols[i]) == d + 1:
                    a, b = cols[i].pop(0), cols[i].pop(0)
                    nl.cells.append(Cell("HA", (a, b), s, c))
                else:
                    a, b, x = cols[i].pop(0), cols[i].pop(0), cols[i].pop(0)
                    nl.cells.append(Cell("FA", (a, b, x), s, c))
                cols[i].append(s)
                cols[i + 1].append(c)
            i += 1
    return cols


def _rows_to_two(cols: list[list[str]], width: int) -> tuple[list[str], list[str]]:
    a, b = [], []
    for i in range(width):
        c = cols[i] if i < len(cols) else []
        a.append(c[0] if len(c) > 0 else "0")
        b.append(c[1] if len(c) > 1 else "0")
    return a, b


def _input_columns(nl: Netlist, rows: range, width: int) -> list[list[str]]:
    cols: list[list[str]] = [[] for _ in range(width)]
    for r in rows:
        bits = [f"in_{r}_{b}" for b in range(nl.w_in)]
        for b, net in enumerate(bits):
            if nl.scheme in ("tc", "oc") and b == nl.w_in - 1:
                inv = nl.new()  # sign-extension elimination: inverted MSB
                nl.cells.append(Cell("INV", (net,), inv, ""))
                net = inv
            cols[b].append(net)
        if nl.scheme == "oc":
            cols[0].append(f"s_{r}")  # +s completes the ones' complement -> two's complement
    if nl.scheme in ("tc", "oc"):
        const = (-len(rows) * (1 << (nl.w_in - 1))) & ((1 << width) - 1)
        for i in range(width):
            if (const >> i) & 1:
                cols[i].append("1")
    return cols


def build_column_tree(
    w_in: int = 9, scheme: str = "oc", rows: int = 32, max_abs: int | None = 144
) -> Netlist:
    """Two rows/2 half trees + 4:2 merge. Outputs: h0_s, h0_c, h1_s, h1_c (half width), m_s, m_c (full)."""
    if scheme not in ("tc", "oc", "unsigned"):
        raise ValueError(scheme)
    half = rows // 2
    if scheme == "unsigned" and max_abs == 144:
        max_abs = None  # the FP4 product bound only applies to signed 9-bit leaves
    if max_abs is not None and max_abs > (1 << (w_in - (0 if scheme == "unsigned" else 1))) - (
        scheme == "unsigned"
    ):
        raise ValueError("max_abs exceeds the w_in-bit input range")
    w_half = _signed_width(half, w_in, scheme, max_abs)
    w_full = _signed_width(rows, w_in, scheme, max_abs)
    nl = Netlist(f"col_tree_{scheme}_w{w_in}_r{rows}", rows, w_in, scheme, w_full)
    nl.inputs = [f"in_{r}_{b}" for r in range(rows) for b in range(w_in)]
    if scheme == "oc":
        nl.inputs += [f"s_{r}" for r in range(rows)]
    # Half trees are built at the *full* width so each carry-save pair equals its half-sum modulo
    # 2^w_full (|half sum| < 2^(w_full-1), so no wrap); the NVFP4 half outputs are the low w_half bits
    # (still exact: |half sum| < 2^(w_half-1)). Sign-extending a carry-save pair row-by-row would be
    # wrong (the two rows can each wrap), so the 4:2 merge simply adds the four full-width rows.
    halves = []
    for h, rng in enumerate((range(0, half), range(half, rows))):
        start = len(nl.cells)
        cols = _dadda(nl, _input_columns(nl, rng, w_full))
        s, c = _rows_to_two(cols, w_full)
        nl.outputs[f"h{h}_s"], nl.outputs[f"h{h}_c"] = s[:w_half], c[:w_half]
        nl.slices[f"h{h}"] = (start, len(nl.cells))
        nl.rows[f"h{h}_s_full"], nl.rows[f"h{h}_c_full"] = s, c
        halves.append((s, c))
    start = len(nl.cells)
    merge_cols: list[list[str]] = [
        [row[i] for pair in halves for row in pair if row[i] != "0"] for i in range(w_full)
    ]
    s, c = _rows_to_two(_dadda(nl, merge_cols), w_full)
    nl.outputs["m_s"], nl.outputs["m_c"] = s, c
    nl.slices["merge"] = (start, len(nl.cells))
    return nl


# --------------------------------------------------------------------------------- bit-level model
def encode_inputs(nl: Netlist, values: np.ndarray) -> dict[str, np.ndarray]:
    """values (T, rows) signed (or unsigned) ints -> {input net: bool (T,)} for this netlist's scheme."""
    v = np.asarray(values, dtype=np.int64)
    nets: dict[str, np.ndarray] = {}
    mask = (1 << nl.w_in) - 1
    for r in range(nl.n_inputs):
        col = v[:, r]
        if nl.scheme == "dual":
            nets[f"s_{r}"] = col < 0
            for b in range(nl.w_in - 1):
                nets[f"in_{r}_{b}"] = ((np.abs(col) >> b) & 1).astype(bool)
            continue
        if nl.scheme == "oc":
            m, s = np.abs(col), (col < 0)
            enc = np.where(s, (~m) & mask, m)
            nets[f"s_{r}"] = s
        else:
            enc = col & mask
        for b in range(nl.w_in):
            nets[f"in_{r}_{b}"] = ((enc >> b) & 1).astype(bool)
    return nets


def evaluate(nl: Netlist, values: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Zero-delay bit-level simulation over T vectors. Returns (all net values, output ints per output).

    Output ints: carry-save pairs are summed modulo 2^width and sign-interpreted (two's complement)
    except for the unsigned scheme.
    """
    T = values.shape[0]
    one, zero = np.ones(T, bool), np.zeros(T, bool)
    val = {"1": one, "0": zero, **encode_inputs(nl, values)}
    for c in nl.cells:
        if c.kind == "INV":
            val[c.sum] = ~val[c.ins[0]]
        elif c.kind == "AND":
            val[c.sum] = val[c.ins[0]] & val[c.ins[1]]
        elif c.kind == "HA":
            a, b = (val[x] for x in c.ins)
            val[c.sum], val[c.carry] = a ^ b, a & b
        else:
            a, b, x = (val[n] for n in c.ins)
            val[c.sum], val[c.carry] = a ^ b ^ x, (a & b) | (a & x) | (b & x)
    out = {}
    for pair in ("h0", "h1", "m"):
        if f"{pair}_s" not in nl.outputs:
            continue
        s, c = nl.outputs[f"{pair}_s"], nl.outputs[f"{pair}_c"]
        w = len(s)
        tot = sum((val[s[i]].astype(np.int64) + val[c[i]].astype(np.int64)) << i for i in range(w))
        tot = tot & ((1 << w) - 1)
        if nl.scheme != "unsigned":
            tot = np.where(tot >= (1 << (w - 1)), tot - (1 << w), tot)
        out[pair] = tot
    return val, out


def toggle_counts(nl: Netlist, net_values: dict[str, np.ndarray]) -> dict[str, int]:
    """Zero-delay toggles of every cell output net and every input net over the trace."""

    def t(x):
        return int(np.count_nonzero(x[1:] != x[:-1]))

    inputs = sum(t(net_values[n]) for n in nl.inputs)
    fa = sum(t(net_values[c.sum]) + t(net_values[c.carry]) for c in nl.cells if c.kind == "FA")
    ha = sum(t(net_values[c.sum]) + t(net_values[c.carry]) for c in nl.cells if c.kind == "HA")
    inv = sum(t(net_values[c.sum]) for c in nl.cells if c.kind in ("INV", "AND"))
    return {
        "inputs": inputs,
        "fa_outputs": fa,
        "ha_outputs": ha,
        "inv_outputs": inv,
        "cells_total": fa + ha + inv,
    }


# --------------------------------------------------------------------------------- SystemVerilog
def emit_sv(nl: Netlist) -> str:
    bits_per_row = nl.w_in - 1 if nl.scheme == "dual" else nl.w_in
    n_in_bits = nl.n_inputs * bits_per_row
    has_halves = "h0_s" in nl.outputs
    w_full = len(nl.outputs["m_s"])
    w_half = len(nl.outputs["h0_s"]) if has_halves else 0
    lines = [
        f"// AUTO-GENERATED by cimfp4.rtlgen.tree - DO NOT EDIT. {nl.n_fa} FA, {nl.n_ha} HA.",
        f"// scheme={nl.scheme} w_in={nl.w_in} rows={nl.n_inputs}; outputs are carry-save pairs.",
        f"module {nl.name} (",
        f"  input  logic [{n_in_bits - 1}:0] in_bits,  // row r bit b at index r*{bits_per_row}+b",
    ]
    if nl.scheme in ("oc", "dual"):
        lines.append(f"  input  logic [{nl.n_inputs - 1}:0] s_bits,")
    lines += [
        *([f"  output logic [{w_half - 1}:0] h0_s, h0_c, h1_s, h1_c,"] if has_halves else []),
        f"  output logic [{w_full - 1}:0] m_s, m_c",
        ");",
    ]

    def ref(net: str) -> str:
        if net in ("0", "1"):
            return f"1'b{net}"
        if net.startswith("in_"):
            _, r, b = net.split("_")
            return f"in_bits[{int(r) * bits_per_row + int(b)}]"
        if net.startswith("s_"):
            return f"s_bits[{int(net[2:])}]"
        return net

    wires = sorted(
        {c.sum for c in nl.cells} | {c.carry for c in nl.cells if c.carry}, key=lambda s: int(s[1:])
    )
    for w in wires:
        lines.append(f"  logic {w} /*verilator public_flat_rd*/;")
    for c in nl.cells:
        a = [ref(x) for x in c.ins]
        if c.kind == "INV":
            lines.append(f"  assign {c.sum} = ~{a[0]};")
        elif c.kind == "AND":
            lines.append(f"  assign {c.sum} = {a[0]} & {a[1]};")
        elif c.kind == "HA":
            lines.append(f"  assign {c.sum} = {a[0]} ^ {a[1]};  assign {c.carry} = {a[0]} & {a[1]};")
        else:
            lines.append(
                f"  assign {c.sum} = {a[0]} ^ {a[1]} ^ {a[2]};  "
                f"assign {c.carry} = ({a[0]} & {a[1]}) | ({a[0]} & {a[2]}) | ({a[1]} & {a[2]});"
            )
    for name, nets in nl.outputs.items():
        lines.append(f"  assign {name} = {{{', '.join(ref(n) for n in reversed(nets))}}};")
    lines.append(f"endmodule : {nl.name}\n")
    return "\n".join(lines)


def build_dual_tree(w_in: int = 9, rows: int = 32, max_abs: int = 144) -> Netlist:
    """Variant C: dual-polarity trees. Inputs: magnitude bits in_r_b (b < w_in-1) and sign s_r per row; row r
    feeds the positive tree if s_r = 0, else the negative tree (AND-gated, so the other tree sees zeros).
    Each tree is a 32-input unsigned Dadda reduction to carry-save; the root forms pos + ~neg_s + ~neg_c + 2
    (two's-complement subtraction of both negative carry-save rows) and reduces to one carry-save pair.
    Outputs: m_s, m_c (full width; NVFP4 half outputs are not provided for this variant)."""
    wm = w_in - 1
    w_full = _signed_width(rows, w_in, "tc", max_abs)
    nl = Netlist(f"col_tree_dual_w{w_in}_r{rows}", rows, w_in, "dual", w_full)
    nl.inputs = [f"in_{r}_{b}" for r in range(rows) for b in range(wm)] + [f"s_{r}" for r in range(rows)]
    pos_cols: list[list[str]] = [[] for _ in range(w_full)]
    neg_cols: list[list[str]] = [[] for _ in range(w_full)]
    for r in range(rows):
        ns = nl.new()
        nl.cells.append(Cell("INV", (f"s_{r}",), ns, ""))
        for b in range(wm):
            gp, gn = nl.new(), nl.new()
            nl.cells.append(Cell("AND", (f"in_{r}_{b}", ns), gp, ""))
            nl.cells.append(Cell("AND", (f"in_{r}_{b}", f"s_{r}"), gn, ""))
            pos_cols[b].append(gp)
            neg_cols[b].append(gn)
    ps, pc = _rows_to_two(_dadda(nl, pos_cols), w_full)
    ns_, nc_ = _rows_to_two(_dadda(nl, neg_cols), w_full)
    root: list[list[str]] = [[] for _ in range(w_full)]
    for i in range(w_full):
        for net in (ps[i], pc[i]):
            if net != "0":
                root[i].append(net)
        for net in (ns_[i], nc_[i]):  # subtract: invert (constant-0 rows invert to 1)
            if net == "0":
                root[i].append("1")
            else:
                inv = nl.new()
                nl.cells.append(Cell("INV", (net,), inv, ""))
                root[i].append(inv)
    two = 2 & ((1 << w_full) - 1)  # +1 +1 completes both two's complements
    for i in range(w_full):
        if (two >> i) & 1:
            root[i].append("1")
    s, c = _rows_to_two(_dadda(nl, root), w_full)
    nl.outputs["m_s"], nl.outputs["m_c"] = s, c
    return nl
