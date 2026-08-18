# [mlir][xegpu] Layout conflict resolution uses `isEqualTo` and materializes thousands of no-op `convert_layout` ops

**Labels:** `mlir`, `mlir:gpu`

**Status:** open. Compile time and IR noise; does not block compilation.

## Summary

`ResolveLayoutConflicts::resolveVectorConsumer` decides whether a producer/consumer layout pair
is a conflict with `DistributeLayoutAttr::isEqualTo`, which is *attribute* equality, not layout
equivalence. Two layouts that describe the identical data distribution but are spelled
differently — a `LayoutAttr` vs. an equivalent `SliceAttr`, or an omitted `order` vs. an
explicit default `order` — are reported as conflicting, and a `xegpu.convert_layout` is
inserted that does nothing.

In `mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F4.mlir` this accounts for
**1040 of the 1136** `convert_layout` ops present just before `xegpu-sg-to-lane-distribute`.

They are all subsequently folded away by `SgToLaneConvertLayout`, which correctly uses
`isCompatibleWith`, so this is not a miscompile. But it costs compile time, it bloats every IR
dump of this kernel, and it makes the pipeline look broken: an unrelated failure elsewhere in
the pass rolls the conversion back (see #ISSUE-2) and the 1040 no-op converts reappear in the
output, which is what originally suggested that `isCompatibleWith` was broken. It is not.

## Reproduction

```mlir
// RUN: mlir-opt --test-xegpu-resolve-layout-conflicts -split-input-file %s

#lane_1x16   = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
#lane_1x1x16 = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>

gpu.module @test {
func.func @slice_vs_layout(%arg0: memref<8x16xbf16>) {
  %c0 = arith.constant 0 : index
  %t = xegpu.create_nd_tdesc %arg0 : memref<8x16xbf16> -> !xegpu.tensor_desc<8x16xbf16, #lane_1x16>
  %v = xegpu.load_nd %t[%c0, %c0] {layout = #lane_1x16}
      : !xegpu.tensor_desc<8x16xbf16, #lane_1x16> -> vector<8x16xbf16>
  %b = vector.broadcast %v {layout_result_0 = #lane_1x1x16}
      : vector<8x16xbf16> to vector<1x8x16xbf16>
  %s = vector.shape_cast %b {layout_result_0 = #lane_1x16}
      : vector<1x8x16xbf16> to vector<8x16xbf16>
  xegpu.store_nd %s, %t[%c0, %c0] {layout = #lane_1x16}
      : vector<8x16xbf16>, !xegpu.tensor_desc<8x16xbf16, #lane_1x16>
  return
}
}

// -----

#no_order   = #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>
#with_order = #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1], order = [1, 0]>

gpu.module @test2 {
func.func @default_order_vs_explicit_order(%arg0: memref<32x16xi8>) {
  %c0 = arith.constant 0 : index
  %t = xegpu.create_nd_tdesc %arg0 : memref<32x16xi8> -> !xegpu.tensor_desc<32x16xi8, #no_order>
  %v = xegpu.load_nd %t[%c0, %c0] {layout = #no_order}
      : !xegpu.tensor_desc<32x16xi8, #no_order> -> vector<32x16xi8>
  xegpu.store_nd %v, %t[%c0, %c0] {layout = #with_order}
      : vector<32x16xi8>, !xegpu.tensor_desc<32x16xi8, #no_order>
  return
}
}
```

Both cases get a `convert_layout`:

```mlir
%2 = xegpu.convert_layout %1
  <{input_layout  = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
    target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16],
                                               lane_data = [1, 1, 1]>, dims = [0]>}>
  : vector<8x16xbf16>

%3 = xegpu.convert_layout %2
  <{input_layout  = #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>,
    target_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1],
                                  order = [1, 0]>}>
  : vector<32x16xi8>
```

Running either through `--xegpu-sg-to-lane-distribute` erases them without emitting a single
instruction, confirming they were no-ops.

## Analysis

`mlir/lib/Dialect/XeGPU/Transforms/XeGPUPropagateLayout.cpp:1658`:

```cpp
  // If layouts are same, no conflict exists, return success.
  if (consumerLayout.isEqualTo(producerLayout))
    return success();
```

`mlir/lib/Dialect/XeGPU/IR/XeGPUDialect.cpp:388`:

```cpp
bool LayoutAttr::isEqualTo(const xegpu::DistributeLayoutAttr &other) {
  if (dyn_cast<xegpu::SliceAttr>(other))
    return false;                       // <-- any slice is "different"

  return *this == dyn_cast<xegpu::LayoutAttr>(other);   // <-- structural equality
}
```

and symmetrically at `:1203` for `SliceAttr`.

* **Case 1** (1024 occurrences in the kernel): the producer is a `LayoutAttr` and the consumer
  layout is the equivalent `SliceAttr` — which arises naturally, because backward propagation
  through a `vector.broadcast` / `vector.shape_cast` that introduces a unit dim produces a slice
  over that dim. `LayoutAttr::isEqualTo` returns `false` on sight of any `SliceAttr`, before
  looking at the distribution at all.
* **Case 2** (16 occurrences): both sides are `LayoutAttr`, but one spells out
  `order = [1, 0]` and the other omits it. `order = [1, 0]` *is* the default for rank 2, so
  `getEffectiveOrderAsInt()` returns `[1, 0]` for both; only the raw attribute comparison
  differs.

`DistributeLayoutAttr::isCompatibleWith(other, shape, level)` already answers the right
question — it compares effective order and, when needed, the actual distributed coordinates.

## Suggested fix

Use `isCompatibleWith` for the no-conflict test in `resolveVectorConsumer`:

```cpp
  auto shape = llvm::to_vector(cast<VectorType>(vectorValue.getType()).getShape());
  if (consumerLayout.isCompatibleWith(producerLayout, shape, level))
    return success();
```

The `level` is already known to the pass (it runs once per `LayoutKind`).

Two smaller alternatives, if changing the predicate is considered too broad:

* Make `LayoutAttr::isEqualTo` / `SliceAttr::isEqualTo` compare *effective* fields
  (`getEffectiveLaneLayoutAsInt`, `getEffectiveLaneDataAsInt`, `getEffectiveOrderAsInt`, …)
  across the `LayoutAttr` / `SliceAttr` boundary, rather than short-circuiting on the attribute
  class.
* At minimum, normalise `order` so an explicit default-order attribute compares equal to an
  omitted one; that alone fixes case 2.

The first option is preferable: `isEqualTo` is also used by the dataflow lattice
(`XeGPUPropagateLayout.cpp:115`), where treating equivalent layouts as distinct can cost extra
fixpoint iterations.

## Where the 1024 come from

The 1024 identical no-ops all convert `layout<[1,16],[1,1]>` to
`slice<layout<[1,1,16],[1,1,1]>,dims=[1]>`, and they exist because of the layout the
`multi_reduction` result rule picks. Removing the `consumerReductionDims.empty()` term from the
`verticalLaneLayout` predicate in `setupMultiReductionResultLayout`
(`mlir/lib/Dialect/XeGPU/Transforms/XeGPULayoutImpl.cpp`, InstData and Lane branches) makes **all
1024 of them disappear** -- the census of `convert_layout` before `xegpu-sg-to-lane-distribute` drops
from 1136 ops to 112.

That is a diagnostic result, not a proposed fix: the same change regresses the conversion in
#ISSUE-3 so that the kernel no longer lowers at all. See #ISSUE-4 for the measurement. It does
establish that these no-ops are a downstream consequence of the reduction result layout rather than
of anything specific to conflict resolution, so #ISSUE-4 and this issue are likely to be fixed
together: the `isEqualTo` predicate discussed above is what turns a benign layout difference into a
materialized op, and the reduction rule is what creates the difference in the first place.

## Impact

* Removes ~1040 dead ops per compile of this kernel and correspondingly shrinks intermediate IR.
* Makes IR dumps of the layout pipeline legible, which matters for diagnosing the real problems
  (#ISSUE-3, #ISSUE-4).
* No functional change expected: everything removed is currently folded away later anyway.
