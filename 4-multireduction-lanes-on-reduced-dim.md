# [mlir][xegpu] `multi_reduction` layout rule places lanes on the reduced dim when the consumer layout is a `SliceAttr` for an unrelated reason

**Labels:** `mlir`, `mlir:gpu`

**Status:** open. Source of the layout #ISSUE-3 has to convert *from*, and of 2048 cross-lane butterfly reductions in the lowered kernel. Fixing it is necessary but **not sufficient** to remove all the scale redistribution -- see "A second conversion this does not remove".

## Summary

`setupMultiReductionResultLayout` decides whether to put the subgroup's lanes on the reduced
dimension or on a surviving one. The heuristic looks at whether the *consumer's* layout is a
`SliceAttr`, and treats the slice's `dims` as "dimensions the consumer already reduced". That
inference is wrong whenever the consumer layout is a slice for some other reason — for example
because the value also feeds a `vector.broadcast`, whose backward layout propagation naturally
produces a slice over the broadcast (prepended) dimension.

When it misfires, lanes end up **on the reduced dimension**. The reduction then becomes a
cross-lane butterfly whose result is replicated across all 16 lanes, and a `convert_layout` has
to be inserted to redistribute it into the layout the real consumer wants. That
`convert_layout` is currently the blocker for
`mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F4.mlir` (see #ISSUE-3).

## Reproduction

```mlir
// RUN: mlir-opt --xevm-attach-target='module=xevm_* chip=cri' \
// RUN:   --xegpu-propagate-layout='layout-kind=inst' %s

#dpas_a_scale = #xegpu.layout<inst_data = [8, 2], lane_layout = [8, 1], lane_data = [1, 1]>
#dpas_b_scale = #xegpu.layout<inst_data = [2, 16], lane_layout = [1, 16], lane_data = [1, 1]>
#a = #xegpu.layout<inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 4]>
#b = #xegpu.layout<inst_data = [64, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#c = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>

gpu.module @xevm_module {
  gpu.func @reduce_then_broadcast(%A: memref<16x1024xbf16>,
                                  %B: memref<1024x16xf4E2M1FN>,
                                  %BS: memref<32x16xf8E8M0FNU>,
                                  %C: memref<16x16xf32>) kernel {
    %c0 = arith.constant 0 : index
    %at  = xegpu.create_nd_tdesc %A  : memref<16x1024xbf16>       -> !xegpu.tensor_desc<16x1024xbf16>
    %bt  = xegpu.create_nd_tdesc %B  : memref<1024x16xf4E2M1FN>   -> !xegpu.tensor_desc<1024x16xf4E2M1FN>
    %bst = xegpu.create_nd_tdesc %BS : memref<32x16xf8E8M0FNU>    -> !xegpu.tensor_desc<32x16xf8E8M0FNU>
    %ct  = xegpu.create_nd_tdesc %C  : memref<16x16xf32>          -> !xegpu.tensor_desc<16x16xf32>

    %a_bf16  = xegpu.load_nd %at[%c0, %c0]                          : !xegpu.tensor_desc<16x1024xbf16>     -> vector<16x1024xbf16>
    %b       = xegpu.load_nd %bt[%c0, %c0]  {layout = #b}           : !xegpu.tensor_desc<1024x16xf4E2M1FN> -> vector<1024x16xf4E2M1FN>
    %scale_b = xegpu.load_nd %bst[%c0, %c0] {layout = #dpas_b_scale}: !xegpu.tensor_desc<32x16xf8E8M0FNU>  -> vector<32x16xf8E8M0FNU>
    %c       = xegpu.load_nd %ct[%c0, %c0]  {layout = #c}           : !xegpu.tensor_desc<16x16xf32>        -> vector<16x16xf32>

    // per-32-element abs-max along K
    %neg_inf = arith.constant dense<0xFF80> : vector<16x32xbf16>
    %abs = math.absf %a_bf16 : vector<16x1024xbf16>
    %r = vector.shape_cast %abs : vector<16x1024xbf16> to vector<16x32x32xbf16>
    %amax = vector.multi_reduction <maximumf>, %r, %neg_inf [2]
        : vector<16x32x32xbf16> to vector<16x32xbf16>

    // amax -> power-of-two scale
    %i16 = arith.bitcast %amax : vector<16x32xbf16> to vector<16x32xi16>
    %mask = arith.constant dense<0x7F80> : vector<16x32xi16>
    %pow2_i16 = arith.andi %i16, %mask : vector<16x32xi16>
    %pow2 = arith.bitcast %pow2_i16 : vector<16x32xi16> to vector<16x32xbf16>
    %four = arith.constant dense<4.0> : vector<16x32xbf16>
    %sc_bf16 = arith.divf %pow2, %four : vector<16x32xbf16>
    %sc = arith.truncf %sc_bf16 : vector<16x32xbf16> to vector<16x32xf8E8M0FNU>

    // %sc has two consumers: dpas_mx's scale_a (wants #dpas_a_scale), and a
    // broadcast back over K used to scale A.
    %lead = vector.broadcast %sc : vector<16x32xf8E8M0FNU> to vector<32x16x32xf8E8M0FNU>
    %t = vector.transpose %lead, [1, 2, 0] : vector<32x16x32xf8E8M0FNU> to vector<16x32x32xf8E8M0FNU>
    %full = vector.shape_cast %t : vector<16x32x32xf8E8M0FNU> to vector<16x1024xf8E8M0FNU>

    %af4 = arith.scaling_truncf %a_bf16, %full
        : vector<16x1024xbf16>, vector<16x1024xf8E8M0FNU> to vector<16x1024xf4E2M1FN>

    %res = xegpu.dpas_mx %af4, %b, %c scale_a = %sc scale_b = %scale_b
        {layout_a = #a, layout_b = #b, layout_cd = #c,
         layout_a_scale = #dpas_a_scale, layout_b_scale = #dpas_b_scale}
        : (vector<16x1024xf4E2M1FN>, vector<1024x16xf4E2M1FN>, vector<16x16xf32>,
           vector<16x32xf8E8M0FNU>, vector<32x16xf8E8M0FNU>) -> vector<16x16xf32>
    xegpu.store_nd %res, %ct[%c0, %c0] {layout = #c}
        : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32>
    gpu.return
  }
}
```

This is the quantize-A chain of `simple_mxfp_gemm_quantizeA_F4.mlir` reduced to a single
subgroup; it produces exactly the layouts the full pipeline produces for that test.

Relevant output:

```mlir
%11 = vector.multi_reduction <maximumf>, %10, %cst
  {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16],
                                                lane_layout = [1, 1, 16],
                                                lane_data = [1, 1, 1]>, dims = [2]>}
  [2] : vector<16x32x32xbf16> to vector<16x32xbf16>

%12 = xegpu.convert_layout %11
  <{input_layout  = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16],
                                               lane_layout = [1, 1, 16],
                                               lane_data = [1, 1, 1]>, dims = [2]>,
    target_layout = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 2],
                                               lane_layout = [8, 1, 2],
                                               lane_data = [4, 1, 1],
                                               order = [0, 2, 1]>, dims = [0]>}>
  : vector<16x32xbf16>
```

`lane_layout = [1, 1, 16]` on a reduction over dim 2 means the 16 lanes are spread **across the
dim being reduced**. `inst_data = [1, 1, 16]` also means one instruction only covers a single
`(m, block)` pair, so the reduction is unrolled 512 times per subgroup.

`%12` (after unrolling: `slice<layout<[1,1,16],[1,1,1]>,dims=[2]>` →
`slice<layout<[8,1,2],[4,1,1],order=[0,2,1]>,dims=[0]>` on `vector<8x2xbf16>`) is the op that
has no sg-to-lane lowering, see #ISSUE-3.

## Where the target layout comes from

The layout `%12` converts *to*,
`slice<layout<[8,1,2],[4,1,1],order=[0,2,1]>,dims=[0]>`, is not produced by this rule. It comes from
the other consumer of the scale. After the scale is computed it is broadcast along K and fed to
`arith.scaling_truncf`, which is what actually divides A by it:

```mlir
%a_scale      = arith.truncf %a_scale_bf16 : vector<32x32xbf16> to vector<32x32xf8E8M0FNU>
%a_scale_lead = vector.broadcast %a_scale  : vector<32x32xf8E8M0FNU> to vector<32x32x32xf8E8M0FNU>
%a_scale_t    = vector.transpose %a_scale_lead, [1, 2, 0]
%a_scale_full = vector.shape_cast %a_scale_t : ... to vector<32x1024xf8E8M0FNU>
%a            = arith.scaling_truncf %a_bf16, %a_scale_full
```

The rank-3 shape is the `vector.broadcast`, and the `order = [0, 2, 1]` permutation is the
`vector.transpose [1, 2, 0]`. So this layout is the broadcast/transpose chain's, propagated back onto
the scale.

## A second conversion this does not remove

`%a_scale` has **two** consumers with different distributions, and only one of them is the broadcast
chain:

```
%6166 = arith.truncf ... {layout_result_0 = slice<[8,1,2],[4,1,1],order=[0,2,1]>, dims=[0]>}
  used by: xegpu.convert_layout  -> layout<[8, 1], [1, 1]>          (dpas_mx scale_a)
  used by: vector.broadcast      -> layout<[8,1,2],...,order=[0,2,1]> (scaling_truncf)
```

Conflict resolution picks the broadcast chain's layout for the value and inserts a `convert_layout`
for `dpas_mx`. Measured on the kernel, that is a second redistribution of 32 ops, distinct from the
32 of #ISSUE-3:

| count | in | out | |
|---|---|---|---|
| 32 | `slice<[1,1,16],dims=[2]>` | `slice<[8,1,2],order=[0,2,1],dims=[0]>` | #ISSUE-3; shuffle-free |
| 32 | `slice<[8,1,2],order=[0,2,1],dims=[0]>` | `layout<[8,1],[1,1]>` | needs cross-lane movement |

The second one needs cross-lane movement for a physical reason: under that layout each lane's
distributed source is `vector<8x1>`, 8 of the 16 elements, so lanes `[0, 8)` hold one column and
lanes `[8, 16)` the other. A lane cannot produce a column it does not hold.

It is worth being precise about what fixing this issue would and would not achieve. The two
conversions form a chain -- the first *creates* the layout the second has to undo. Collapsing them
into one conversion straight from the reduction's broadcast layout to `layout<[8,1]>` is
shuffle-free today. But that requires resolving the two-consumer conflict on `%a_scale` in favour of
`dpas_mx`, which is a separate decision from the reduction result layout this issue is about.

## Analysis

Two separate problems, both in `mlir/lib/Dialect/XeGPU/Transforms/XeGPULayoutImpl.cpp`.

### 1. The `verticalLaneLayout` predicate over-reads the consumer's `SliceAttr`

`setupMultiReductionResultLayout`, InstData branch (~L2445) and Lane branch (~L2485):

```cpp
    xegpu::SliceAttr consumerSliceLayout =
        dyn_cast_if_present<xegpu::SliceAttr>(consumerLayout);
    auto consumerReductionDims =
        consumerSliceLayout
            ? SmallVector<int64_t>(consumerSliceLayout.getDims().asArrayRef())
            : SmallVector<int64_t>({});
    // A[i] reduced from A[i, j] is stored out directly, use vertical Lane
    // layout like [16, 1]
    bool verticalLaneLayout = consumerReductionDims.empty() &&
                              reductionDims.size() == 1 &&
                              reductionDims[0] == (srcRank - 1);
```

In the repro, `reductionDims == [2]` and `srcRank - 1 == 2`, so the only thing standing between
us and the sensible layout is `consumerReductionDims.empty()`. The consumer layout here is
`slice<..., dims = [0]>`, which came from the **`vector.broadcast`** further down the chain
(a broadcast prepends dim 0, so its operand layout is a slice over dim 0) — it has nothing to
do with a reduction. `consumerReductionDims` is `[0]`, the predicate is false, and lanes land on
the reduced dim.

Note the equivalent Subgroup branch (~L2378) does not use this heuristic: it either reuses the
consumer's slice when `dims == reductionDims`, or explicitly matches the consumer's layout on
the non-reduction dims and only then spends leftover subgroups on the reduction dims. That
"match the consumer on surviving dims" formulation is the one that generalises.

### 2. `computeReductionLaneLayoutAndData` ignores `reductionDims` entirely

```cpp
/// ... To minimize cross-lane reduction, lanes
/// are spread across a non-reduction dim when possible so the reduction happens
/// within a lane.
static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
computeReductionLaneLayoutAndData(ArrayRef<int64_t> srcShape,
                                  ArrayRef<int64_t> reductionDims,
                                  int subgroupSize, int64_t maxReduceVectorSize,
                                  bool verticalLaneLayout = false) {
  ...
  int laneDim = innermost;      // reductionDims is never read
```

The `reductionDims` parameter is dead. The function does exactly what its own doc comment says
it avoids, unless the caller happens to pass `verticalLaneLayout = true`. Even in the "vertical"
case the choice is a hardcoded swap of the two innermost dims rather than a search for a
non-reduced dim, so it cannot express e.g. `reductionDims = [1]` on a rank-3 source.

## Suggested fix

Make the InstData and Lane branches consumer-aware, the way the Subgroup branch already is:

1. Never place lanes on a dim in `reductionDims` if any non-reduced dim can hold them. Pass
   `reductionDims` into `computeReductionLaneLayoutAndData` and use it, instead of the
   `verticalLaneLayout` bool.
2. On the non-reduced dims, follow the consumer's `lane_layout` / `lane_data` when it is
   compatible with `srcShape`, so the reduction result is produced directly in the layout the
   consumer wants.

For the repro, that means picking `lane_layout = [16, 1, 1]` / `inst_data = [8, 2, 32]`-ish for
the source, giving the result `lane_layout = [8, 1]` — exactly `#dpas_a_scale`. The reduction
becomes fully in-lane (no `gpu.shuffle`), `%12` disappears, and so does the surrounding
broadcast/transpose chain's layout mismatch.

### What a minimal attempt at step 1 actually does

Dropping `consumerReductionDims.empty()` from the `verticalLaneLayout` predicate — step 1 in its
smallest form — is **not** enough, and on its own makes things worse. Measured on the kernel:

* the second conversion above is unaffected, still 32 ops, since it does not originate here;
* the #ISSUE-3 conversion regresses. Its source layout changes from `slice<[1,1,16],dims=[2]>`,
  which distributes 16 of 16 elements to every lane, to `slice<[1,16,1],dims=[2]>`, which gives each
  lane 1. It stops being a broadcast, so it now needs cross-lane movement too and no longer lowers;
* the kernel therefore fails to lower, on a *new* op.

So the fix has to do step 2 as well: the reduction result must land in a layout the consumer actually
wants, not merely off the reduced dimension. Step 1 alone moves the lanes without fixing where they
land.

One useful side effect of that experiment: it removed **all 1024** of the no-op `convert_layout` ops
described in #ISSUE-5. See that issue for the detail.

## Why this kernel and not its sibling

`simple_mxfp_gemm_quantizeA_F4.mlir` and `simple_mxfp_gemm_dequantizeB_F4.mlir` are identical in
shape: same operand sizes (256x4096 bf16 A, 2048x256 i8 packed B, 256x256 f32 C), same tiling
(`sg_layout = [2, 2]`, `sg_data = [16, 1024]`), same K loop (`0` to `4096` step `1024`, four
iterations), same `dpas_mx`. The only difference is that `quantizeA_F4` computes the A scales in the
kernel instead of loading them precomputed.

That single difference is what pulls in this reduction layout, the resulting `convert_layout`
(#ISSUE-3), and the cross-lane data movement described above. The sibling kernel never reaches any
of it, so all of the extra work this issue describes is attributable to the in-kernel scale
computation rather than to the matrix multiplication.

## Impact

* Addresses #ISSUE-3 at the source rather than teaching sg-to-lane to lower it. It does **not**
  remove the second redistribution described above, which comes from `%a_scale`'s other consumer.
* Replaces the butterfly reductions with in-lane reductions. In the lowered kernel these are
  **2048 of the 2080** `gpu.shuffle` ops, i.e. 98% of all cross-lane traffic, so this is by far the
  largest codegen win available here. The scale redistribution accounts for the other 32.
* `#ISSUE-3` and this issue are complementary, not exclusive: sg-to-lane should still be able
  to lower the conversion when it is genuinely needed.

## Why the cost class matters

A `lane_data`-only layout mismatch is free in hardware: it lowers to `xegpu.lane_shuffle`, whose
pack/unpack forms are a register reinterpretation rather than data movement. A replicated →
distributed mismatch is not, and cannot be made so, because it needs a *lane-dependent* selection
-- lane `i` must end up with element `i` -- which no fixed permutation can supply. It costs one
dynamic `vector.extract` per element, plus a `gpu.shuffle` per element that is not in the lane's
own fragment.

So the layout choices in this file partition into a free class and an expensive one, and this
issue is about landing in the free one. Optimizing the lowering of #ISSUE-3 chases a constant
factor on code that ideally is not emitted at all.

One clarification, since it is easy to overstate: a reduction result layout is always a
`SliceAttr` -- `setupMultiReductionResultLayout` returns one by construction -- but that is not
itself the problem. The result is only *replicated* when the sliced dims carry more than one lane.
With `lane_layout = [16, 1, 1]` and a reduction over dim 2, the result is
`slice<layout<[16, 1, 1]>, dims = [2]>`, whose sliced dim holds a single lane: distributed over M,
nothing replicated, no conversion needed.

## Note: this cannot be worked around from the test

Annotating the source with an M-distributed layout and inserting an explicit
`xegpu.convert_layout` to `#dpas_a_scale` before `xegpu.dpas_mx` has no effect — the reduction
rule overrides the annotation and the resulting IR is the same multiset of operations.
