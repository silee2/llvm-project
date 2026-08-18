# [mlir][xegpu] `arith.truncf` wider than one `xevm.truncf` conversion group is not lowered

**Labels:** `mlir`, `mlir:gpu`

**Status:** fixed.
**Fixed by:** [PR #217130](https://github.com/llvm/llvm-project/pull/217130), branch `xegpu-truncf-split-wide`.
**Depends on:** nothing, for landing. Only *reachable* in the mxfp kernel once #ISSUE-3 lets the XeGPU layer lower, and it stays required after #ISSUE-6 is fixed -- see "Interaction with #ISSUE-6".

## Summary

`TruncfToXeVMPattern` in `convert-xegpu-to-xevm` matched an `arith.truncf` only when its source held
*exactly* 16 elements. A wider source was left unconverted, and since nothing downstream can
translate the MX narrow float types, the module then failed to build:

```
cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration
  ... for op: arith.truncf
```

Wider sources do occur, so this is a hard failure rather than a missed optimisation.

## Reproducer

```
mlir-opt --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
  mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F4.mlir
```

Requires the fix from #ISSUE-3 to get this far. Minimally, the op that fails is:

```mlir
%r = arith.truncf %a : vector<32xbf16> to vector<32xf4E2M1FN>
```

## Where the restriction came from

`xevm.truncf` lowers to device builtins that convert a fixed number of elements per call
(`mlir/lib/Conversion/XeVMToLLVM/XeVMToLLVM.cpp`):

* the fp4 path feeds `__builtin_IB_dnscl_bf16` with a `vector<8xi32>`;
* the fp8 paths call the `_16` flavours of the half/bf16 conversion builtins,
  `__builtin_IB_hftobf8_16` and `__builtin_IB_hftohf8_16`.

All of them take exactly 16 f16/bf16 elements. That granularity is captured by
`kXeVMExtfTruncfNumElems = 16` (`mlir/lib/Conversion/XeGPUToXeVM/XeGPUToXeVM.cpp:1220`), and the
match predicate simply required the source to be that wide:

```cpp
if (srcTy.getNumElements() != kXeVMExtfTruncfNumElems)
  return false;
```

So the restriction is a real property of the instruction, not an oversight. What was missing was the
step from *one instruction* to *an operation of arbitrary width*.

## Where the wide source comes from

In a workgroup-level mxfp GEMM that quantizes A in the kernel, A's `inst_data` and lane layout give
each lane 32 bf16 per instruction, and `arith.scaling_truncf` expands into an `arith.truncf` from
`vector<32xbf16>` to `vector<32xf4E2M1FN>`. That is two conversion groups, not one.

This is ordinary, not pathological: the width follows from the layouts the kernel needs for
`dpas_mx`, so any kernel quantizing to fp4 at that `inst_data` hits it.

## Fix

[PR #217130](https://github.com/llvm/llvm-project/pull/217130), branch `xegpu-truncf-split-wide`.

Match any source that is a whole number of groups, and convert a group at a time: slice the source
with `vector.extract_strided_slice`, emit one `xevm.truncf` per group, concatenate the packed `i8`
results with `vector.insert_strided_slice`, then bitcast once to the type-converted result type (an
`i4` vector for fp4).

Two properties worth noting:

* **A source of exactly one group takes the original path**, with no slicing or concatenation, so its
  generated code is unchanged. The splitting only appears when a wider vector is actually converted.
* **Sources that are not a whole number of groups are still not matched** and are left to the regular
  arith-to-LLVM path, rather than being lowered with a ragged tail.

## Interaction with #ISSUE-6

#ISSUE-6 adds a pass that caps the width of elementwise `arith` ops before the XeVM conversions, and
it might look as though that makes this fix unnecessary. It does not. That pass deliberately exempts
ops with a sub-byte, non-`i1` operand or result, because splitting them materialises sub-byte vector
data movement that the backend rejects. `arith.truncf ... to vector<32xf4E2M1FN>` is exactly such an
op, so it is left at 32 components on purpose.

The two changes are therefore complementary, and this one is load-bearing both before and after
#ISSUE-6 is fixed. Removing it reintroduces the failure above.

## `arith.extf` has the same restriction

`isXeVMExtf` requires the *result* to be exactly `kXeVMExtfTruncfNumElems` wide
(`XeGPUToXeVM.cpp:1251`), for the same instruction-granularity reason. No wider case has come up, so
it is left as is, but a dequantizing kernel would hit the mirror image of this issue and the same
grouping would apply.

## Testing

New cases in `mlir/test/Conversion/XeGPUToXeVM/extf_truncf.mlir`:

* `vector<32xbf16>` -> `vector<32xf4E2M1FN>`, checking two groups of 16, each packed into
  `vector<8xi8>`, concatenated into `vector<16xi8>` and bitcast to `vector<32xi4>`;
* `vector<24xbf16>` -> `vector<24xf4E2M1FN>`, checking that a partial group emits no `xevm.truncf`
  and is left as `arith.truncf`.

`mlir/test/Dialect/XeGPU`, `mlir/test/Dialect/Vector`, `mlir/test/Conversion/XeGPUToXeVM`,
`mlir/test/Conversion/XeVMToLLVM`, `mlir/test/Conversion/VectorToLLVM`, `mlir/test/Dialect/GPU` and
`mlir/test/Dialect/Arith` pass with no regressions, and the branch was also checked standalone
against `main` with `check-mlir`.

## Open question

Should the grouping live in the conversion pattern, as here, or should a target-width legalization
pass upstream of `convert-xegpu-to-xevm` be responsible for presenting only single-group ops? The
latter is where #ISSUE-6 operates, but the sub-byte exemption means it cannot own this case, so the
pattern has to cope either way.
