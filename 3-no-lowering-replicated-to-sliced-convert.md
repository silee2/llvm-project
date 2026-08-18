# [mlir][xegpu] sg-to-lane: no lowering for `convert_layout` from a fully replicated layout to a partially replicated (sliced) target

**Labels:** `mlir`, `mlir:gpu`

**Status:** fixed.
**Fixed by:** [PR #217104](https://github.com/llvm/llvm-project/pull/217104), branch `xegpu-convert-layout-broadcast-divisor`, on top of `xegpu-convert-layout-broadcast-redistribute`.
**Depends on:** `xegpu-convert-layout-broadcast-redistribute`, the base pattern it generalizes, which is upstream as [PR #215645](https://github.com/llvm/llvm-project/pull/215645) and must land first.
**Superseded by:** #ISSUE-4 would remove the need for this conversion entirely; the two are complementary, not exclusive.
**Validated:** the redistribution is numerically correct -- `simple_mxfp_gemm_quantizeA_F4.mlir` XPASSes with 0 mismatching elements. See #ISSUE-1.

## Summary

`xegpu-sg-to-lane-distribute` cannot lower a `convert_layout` whose input layout replicates the
value across the whole subgroup and whose target layout is a `SliceAttr` that replicates it
across *groups* of lanes while distributing it across the rest. It fails with
`lowering incompatible convert_layout not yet supported`.

This is the remaining blocker for
`mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F4.mlir` (32 occurrences).

## Reproduction

```mlir
// RUN: mlir-opt --xevm-attach-target='module=xevm_* chip=cri' \
// RUN:   --allow-unregistered-dialect --xegpu-sg-to-lane-distribute %s

gpu.module @xevm_module {
gpu.func @cvt_replicated_to_sliced_target() {
  %src = arith.constant dense<1.0> : vector<8x2xbf16>
  %cvt = xegpu.convert_layout %src
    <{
      input_layout  = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>,
      target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>
    }> : vector<8x2xbf16>
  "some_use"(%cvt) : (vector<8x2xbf16>) -> ()
  gpu.return
}
}
```

```
error: failed to legalize operation 'xegpu.convert_layout' that was explicitly marked illegal
```

## Analysis

Work out what each side assigns to lane `t` for shape `8x2` and a subgroup size of 16:

**Input** `slice<layout<lane_layout=[1,1,16], lane_data=[1,1,1]>, dims=[2]>` — effective
`lane_layout = [1,1]`, so every lane holds all 16 elements, in row-major order. The distributed
type is `vector<8x2xbf16>`.

**Target** `slice<layout<lane_layout=[8,1,2], lane_data=[4,1,1], order=[0,2,1]>, dims=[0]>` —
the parent `order` makes dim 0 (the *sliced*, i.e. replicated, dim) the fastest-varying, so
`d0 = t % 8` and `d2 = (t / 8) % 2`. Lane `t` therefore owns **column `t / 8`**, all 8 rows —
lanes [0,8) get column 0, lanes [8,16) get column 1. The distributed type is `vector<8x1xbf16>`.

So result element `pos` of lane `t` sits at index `2*pos + (t / 8)` of the copy that lane
already holds.

The in-flight broadcast-redistribute pattern derives the extract index as an affine function of
the lane index and rejects anything else:

```cpp
    // The extracted element is computed from the lane id at runtime, so it has
    // to be an affine function of the lane's position in the target layout.
    int64_t offset = elements[0];
    int64_t stride = numTargetLanes > 1 ? elements[1] - offset : 0;
    ...
          return elements[t] == stride * t + offset;
```

`2*pos + t/8` is a step function of `t`, not `stride * t + offset`, so the derivation fails.
The existing supported cases only ever need `t % numTargetLanes` scaled by a stride, which
happens to be the case when the distributed dim is fastest-varying in the parent `order`.

## Suggested fix

Generalise the derived index from `stride * t + offset` to a form that can also express a
division of the lane id, e.g.

```
index = stride * ((t / divisor) % modulus) + offset
```

with `divisor` / `modulus` read off the parent `lane_layout` and `order` (here `divisor = 8`,
`modulus = 2`, `stride = 1`, `offset = 2*pos`). Codegen stays cheap: one extra `arith.divui`
by a constant.

Worth noting: in this case the input is replicated over the *whole* subgroup, so every lane
already holds the element it ends up owning. `needsShuffle` is false for every element and the
op should lower to a plain dynamic `vector.extract` with **no** `gpu.shuffle` at all — the same
shape of code as the existing "broadcast to all lanes" test.

## Related

* This conversion only exists because of the reduction layout rule — see #ISSUE-4.
* The sibling conversion in the same kernel
  (`slice<[8,1,2],[4,1,1],order=[0,2,1],dims=[0]>` → `layout<[8,1],[1,1]>` on
  `vector<8x2xf8E8M0FNU>`) *is* handled by the broadcast-redistribute work.
