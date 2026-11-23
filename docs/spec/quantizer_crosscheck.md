# Quantizer cross-check (WP-07)

Our OCP-exact quantizers (`cimfp4.formats.mx_quant`, `nv_quant`) vs library implementations.
Tests: `tests/formats/test_crosscheck.py` (torchao 0.18.0).

| Library | Mode | Ours | Result |
|---|---|---|---|
| torchao `to_mx` | FLOOR (OCP v1.0) | `ocp_floor` | **bit-identical** scales + dequantized values (1e6 elems/rule; 1e7 slow test) |
| torchao `to_mx` | EVEN | `ocp_v11` (expAdj) | **bit-identical** |
| torchao `to_mx` | RCEIL | `rceil` | **bit-identical** |
| torchao `to_mx` | CEIL | `ceil` | **bit-identical** |
| torchao `nvfp4_quantize` (two-level) | - | `recip_mul` | **bit-identical on blocks with normal-range E4M3 scales** (scale codes and elements, 0 disagreements). Convention difference: torchao clamps block scales to the smallest normal E4M3 (2^-6); ours rounds into the subnormal range (NVIDIA recipe). Affects only blocks ≳3·10^4× below the tensor amax (41.9 % of blocks on the deliberately extreme synthetic test data; to be measured on real captures). |
| microxcaling (`round='even'`) | - | - | **deferred**: requires cloning the GitHub repo (not a PyPI package); external code download needs user approval |
| NVIDIA ModelOpt | MXFP4 (ceil scale, ties toward zero per R02/R07) | - | **deferred** (known non-OCP defaults; see above) |

Known library divergences from OCP: microxcaling default rounding is ties-away; ModelOpt MXFP4
uses a ceil scale with ties toward zero; torchao inference default scale rule is not FLOOR/EVEN.
