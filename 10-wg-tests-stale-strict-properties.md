# [mlir][xegpu] WG integration tests do not parse since layout attributes became strict properties

**Labels:** `mlir`, `mlir:gpu`

**Status:** open. Affects tests only, but they are dead weight until fixed.

## Summary

The `layout`, `layout_a`, `layout_b`, `layout_cd` and `layout_*_scale` attributes of `load_nd`,
`store_nd`, `prefetch_nd`, `dpas` and `dpas_mx` are *inherent*, and the assembly format now
enables strict properties, so they no longer parse from the attribute dictionary:

```
error: custom op 'xegpu.load_nd' inherent attribute 'layout' cannot be parsed from attr-dict
when strict properties in assembly format is enabled
```

Most of `mlir/test/Integration/Dialect/XeGPU/WG/` still uses the old spelling and therefore fails
at parse time, before reaching the pipeline it is meant to exercise:

```mlir
%c_init = xegpu.load_nd %cd_tdesc[%m, %n] {layout = #c} : ...        // no longer parses
%c_init = xegpu.load_nd %cd_tdesc[%m, %n] <{layout = #c}> : ...      // properties form
```

`simple_gemm.mlir` and `simple_mxfp_gemm_dequantizeB_F4.mlir` were migrated with the change; the
rest were not. Affected in tree:

| file | occurrences |
|---|---|
| `simple_3d_gemm.mlir` | 12 + 1 `dpas` dict |
| `simple_3d_mxfp_gemm.mlir` | 6 + 1 `dpas_mx` dict |
| `simple_mxfp_gemm.mlir` | 6 + 1 |
| `simple_mxfp_gemm_F8.mlir` | 6 + 1 |
| `simple_mxfp_gemm_dequantizeB_F8.mlir` | 5 + 1 |
| `simple_mxfp_gemm_quantizeA_F4.mlir` | 5 + 1 |
| `simple_mxfp_gemm_quantizeA_F8.mlir` | 5 + 1 |
| `simple_gemm_bf16_k4096.mlir` | 4 + 1 |

## Why CI did not catch it

Every one of these tests either carries `XFAIL: *` or has its execution line commented out as
`RUN-DISABLED`, because running them needs Intel GPU hardware. An `XFAIL` test that fails to parse
is still an expected failure, so the breakage is invisible; and for the `RUN-DISABLED` ones the
active `RUN` line is exactly the `mlir-opt` invocation that now fails, which turns them into
tests that pass no information.

## Suggested fix

Mechanical: move the dictionaries to the properties form. All occurrences are the simple
`{layout = #name}` shape or a single multi-line `{layout_a = ..., layout_b = ...}` dict per file,
so it is a two-rule substitution.

After that, all the mxfp and bf16 GEMM tests lower again through
`--gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri"`, and the seven that have
a numerical check pass on the simulator with 0 mismatching elements.

`simple_3d_mxfp_gemm.mlir` remains `XFAIL`, but for a real reason rather than a parse error: it
then hits `'xegpu.load_nd' op TensorDesc shape is not distributable with the layout` on a 3D scale
load, `!xegpu.tensor_desc<4x128x16xf8E8M0FNU>` with `#a_scale_load`. Worth a separate look.

## Suggestion

Consider whether these tests can keep an always-on lowering-only `RUN` line without `XFAIL`, so
that a parse or legalization regression is caught by ordinary CI, with only the execution line
gated on hardware. That is the arrangement the `*_lengths` tests in
`mlir/test/Integration/Dialect/XeVM/GPU/` use.
