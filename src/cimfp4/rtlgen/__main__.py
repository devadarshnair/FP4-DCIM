"""Regenerate all generated RTL and spec tables: python -m cimfp4.rtlgen"""

from ..utils.provenance import REPO_ROOT
from .leaf_spec import emit_leaf_spec_sv
from .pkg import write_all
from .tree import build_column_tree, build_dual_tree, emit_sv

#: trees used by the designs: v2 leaf (variant B, baseline), TC ablation, bit-serial baseline
TREES = [(9, "oc"), (9, "tc"), (1, "unsigned")]

for path in write_all():
    print("wrote", path)
gen = REPO_ROOT / "rtl" / "gen"
gen.mkdir(parents=True, exist_ok=True)
for w_in, scheme in TREES:
    nl = build_column_tree(w_in, scheme)
    path = gen / f"{nl.name}.sv"
    path.write_text(emit_sv(nl))
    print("wrote", path, f"({nl.n_fa} FA, {nl.n_ha} HA)")
nl = build_dual_tree(9)
(gen / f"{nl.name}.sv").write_text(emit_sv(nl))
print("wrote", gen / f"{nl.name}.sv", f"({nl.n_fa} FA, {nl.n_ha} HA)")
spec = REPO_ROOT / "tb" / "formal" / "gen" / "leaf_spec.sv"
spec.parent.mkdir(parents=True, exist_ok=True)
spec.write_text(emit_leaf_spec_sv())
print("wrote", spec)
