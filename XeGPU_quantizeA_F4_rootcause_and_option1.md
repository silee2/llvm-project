# Root-cause analysis: `simple_mxfp_gemm_quantizeA_F4.mlir` lowering crash
## …and how it relates to PR #203156 (layout propagation refactor)

**To:** Jianhui-Li
**Re:** `mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F4.mlir`
**Pipeline:** `mlir-opt --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri"`

---

## 1. TL;DR

- The test aborts on an assertion during **lane-level** `XeGPUPropagateLayout`, in `setupInsertStridedSliceResultLayout` (`XeGPULayoutImpl.cpp:1162`).
- Root cause is **not** `insert_strided_slice` — it is the **`vector.multi_reduction` layout rule**. For the A abs-max reduction (reduction over the innermost K dim), the rule places the SIMT lanes **on the reduced dimension**, producing a lane-replicated `[1,1]` result that then must be cross-lane redistributed to the `dpas_mx` scale layout `[8,1]`.
- I verified **no test-side layout change can fix this** (the rule overrides any operand layout we supply).
- I empirically measured **PR #203156**: it removes the *crash* (good) but does **not** enable lowering — the reduction still resolves to `[1,1]`, so the cross-lane convert and reduction become "failed to legalize" errors instead.
- **Option 1 (proposed):** make the InstData/Lane branches of `setupMultiReductionResultLayout` honor the **consumer's** lane distribution on non-reduced dims and keep reduced dims lane-local — exactly mirroring what the **Subgroup** branch already does for `sg_layout`.

---

## 2. The crash

```
XeGPULayoutImpl.cpp:1162: setupInsertStridedSliceResultLayout(...):
Assertion `srcShape[dim] % consumerLaneLayout[dim] == 0 &&
           "srcShape must be divisible by laneLayout for all dimensions"' failed.
```

Call path: lane-level `XeGPUPropagateLayout` → `visitInsertStridedSliceOp` → `setupInsertStridedSliceResultLayout` (Lane branch).

Offending ops (512 of them) build a `16x32` from **`1x1`** pieces:

```mlir
%r    = vector.multi_reduction <maximumf>, %x, %acc [2] : vector<1x1x16xbf16> to vector<1x1xbf16>
%acc2 = vector.insert_strided_slice %r, %acc {offsets=[i,j]} : vector<1x1xbf16> into vector<16x32xbf16>
```

`srcShape = [1,1]`, `consumerLaneLayout = [8,1]` → `1 % 8 != 0` → abort.

---

## 3. Where the `1x1` comes from (the real cause)

Quantize-A computes a per-K-block abs-max:

```mlir
%amax = vector.multi_reduction <maximumf>, %a_abs_r, %neg_inf [2]
        : vector<16x32x32xbf16> to vector<16x32xbf16>   // (M, block, Kinner) -> (M, block)
```

Inst-level layout assigned to it (before blocking):

```mlir
%amax  = vector.multi_reduction ... {layout_result_0 =
   #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16]>, dims = [2]>}   // result inst = [1,1]
%scale = xegpu.convert_layout %amax_chain
   : slice<[1,1,16]>   ->   #xegpu.layout<inst_data=[8,2], lane_layout=[8,1], lane_data=[1,1]>
```

- Reduction result is `[1,1]` / **lane-replicated** (lanes sit on the reduced K dim).
- Its consumer is `dpas_mx` `scale_a`, which requires `#dpas_a_scale` = `inst=[8,2], lane=[8,1]`.
- ⇒ a `convert_layout [1,1] -> [8,1]` is inserted. **Blocking** lowers it as *insert `1x1` → assemble `16x32` → extract `8x2`*. Lane propagation then sees a `1x1` insert under a `[8,1]` lane layout → assert.

---

## 4. Root cause stated plainly

A **layout transpose tension**:

| Tensor | Distribution | Why |
|---|---|---|
| A operand (for `dpas_mx`) | `lane=[1,16]` — **16 lanes along K** | matches dpas A operand |
| abs-max reduction | reduces **along K** | ⇒ reduction runs *across* the lanes (cross-lane) |
| reduction result rule | lanes on innermost (= reduced) dim ⇒ result **`[1,1]` replicated** | `computeReductionLaneLayoutAndData` |
| scale (for `dpas_mx`) | `lane=[8,1]` — **8 lanes along M** | HW scale layout `#dpas_a_scale` |

The reduction result can never equal `[8,1]`, so a cross-lane `[1,1]→[8,1]` redistribution is **structurally unavoidable** — and that is what crashes today.

---

## 5. This is not fixable from the test

I tried the obvious test-side routes; all fail because **the reduction's layout is assigned by the rule, not inferred from its operand**:

1. **Annotate** `absf`/`shape_cast`/`multi_reduction` with M-distributed layouts → `setupMultiReductionResultLayout` recomputes and overrides them.
2. **Load A twice** (a second, M-distributed `[8,1]` load feeding only the reduction) → propagation inserts a `convert_layout` that **immediately undoes** the M-distribution, because the reduction *demands* a K-distributed source:
   ```mlir
   %25 = xegpu.convert_layout %24 : [8,64]/lane[8,1]  ->  inst_data=[1,16]   // M-dist discarded
   %28 = vector.multi_reduction ... slice<[1,1,16],dims=[2]>                 // still [1,1]
   ```
3. **Relax `#dpas_a_scale` to `[1,1]`** → only `[1,1]` satisfies the assert, but it violates the `dpas_mx` HW scale constraint.

Conclusion: the fix must live in the **reduction layout rule**.

---

## 6. What PR #203156 changes here (measured, not guessed)

I grafted the PR's `XeGPULayoutImpl.{h,cpp}`, `XeGPUPropagateLayout.cpp`, `XeGPUUnroll.cpp` onto the branch (they were identical to the merge-base, so this is faithful) and rebuilt `mlir-opt`.

**Result: crash → graceful failure (improvement), but still does not lower.**

| | Before | With PR #203156 |
|---|---|---|
| Exit | abort (134) | error (1) |
| Diagnostic | `assert(...)` in `setupInsertStridedSliceResultLayout` | `failed to legalize 'xegpu.convert_layout'` (`lane[1,1]→[8,1]`) and `'vector.multi_reduction'` (`1x1x16→1x1`) |

Why it doesn't lower:

- `setupInsertStridedSliceResultLayout` keeps the **same Lane-branch assert** (only the InstData branch was stubbed).
- `computeReductionLaneLayoutAndData` still sets `laneDim = innermost` unconditionally ⇒ for an innermost-dim reduction the result is still `[1,1]` ⇒ the same cross-lane convert + cross-lane reduction remain, now surfacing as legalization failures.

---

## 7. The bug is in the new helper (and contradicts its own comment)

`computeReductionLaneLayoutAndData` (PR #203156):

```cpp
int laneDim    = innermost;        // lanes always on innermost...
int vectorDim  = secondInnermost;  // ...within-lane packing on 2nd-innermost
laneLayout[laneDim]  = min(subgroupSize, srcShape[laneDim]);
laneData[vectorDim]  = min(maxReduceVectorSize, srcShape[vectorDim]);
```

Its doc comment states the intent:

> *"…picks a layout that minimizes cross-lane reduction (reducing within a lane when only one of the innermost two dims is a reduction dim)."*

But the implementation ignores `reductionDims`. When the **innermost dim is the reduction dim** (our case), it puts lanes **on** the reduced dim — i.e. it *maximizes* cross-lane reduction, the opposite of the stated intent.

---

## 8. Option 1: consumer-aware, lane-local reduction layout

Make the **InstData/Lane** branches of `setupMultiReductionResultLayout` do what the **Subgroup** branch already does — except for `lane_layout`/`lane_data` instead of `sg_layout`/`sg_data`:

1. **Reduced dims → lane-local.** Never place lanes on a reduction dim; set `lane_layout = 1` there (pack into `lane_data` when `maxReduceVectorSize > 1`). This makes the reduction intra-lane and distributable.
2. **Non-reduced dims → follow the consumer.** Take `lane_layout`/`lane_data` from `consumerLayoutAttr` on the non-reduced dims (it is already a parameter and already used by the Subgroup branch).
3. Keep `inst_data = lane_layout * lane_data` (the Cat-A invariant for non-anchor ops).

For this kernel — consumer `[8,1]` over `(M, block)`, reduce `Kinner`:

```
source lane  = [8, 1, 1]   (8 lanes on M, 1 on block, Kinner within-lane)
result lane  = [8, 1]      == #dpas_a_scale   ⇒ NO convert_layout, reduction is lane-local
```

Both failures vanish at the source: no `[1,1]→[8,1]` convert, and the reduction is lane-local (legal in `SgToLane`).

---

## 9. Option 1 pairs naturally with the "two-load A" pattern

- With today's rule, "load A twice (M-distributed copy for the scale path)" **fails** — propagation reverts the M-distribution (Section 5).
- With Option 1, the reduction *requests* an M-distributed source, so the M-distributed A load flows straight in **with zero cross-lane converts** end-to-end:

```mlir
%a_bf16       = load_nd ... {layout = #a_ld}        // K-distributed -> dpas_mx A operand
%a_bf16_scale = load_nd ... {layout = #a_scale_ld}  // M-distributed -> abs-max reduction (lane-local)
```

So Option 1 is the enabling change; the two-load pattern is the (cheap) test-side companion that avoids re-shuffling A in registers.

---

## 10. Open questions / things to sanity-check

1. **Consumer selection.** When the consumer is not a plain `LayoutAttr` (e.g. a `SliceAttr`, or multiple consumers with conflicting lane layouts), how should the non-reduced dims pick their lane layout? The Subgroup branch already has precedent for the slice case.
2. **Rank > "innermost two".** The current helper only reasons about the innermost two dims and `assert`s `leadingDimsAreUnit`. Option 1 needs to place lanes on an arbitrary non-reduced dim (here M is the *leading* dim), so that assumption has to be relaxed for this pattern.
3. **The other scale convert.** There is a second redistribution in this kernel: scale `[8,1]` → broadcast → `scaling_truncf` on the K-distributed A (`%33` in the dump). Option 1 removes the *reduction→scale* convert (the crash); please confirm whether this broadcast convert is already lowerable or needs separate attention for full end-to-end success.
4. **Keep the assert, or clamp?** Independent of Option 1, `setupInsertStridedSliceResultLayout`'s Lane branch still `assert`s on `srcShape[dim] < laneLayout[dim]`. Even with Option 1 correct, a defensive diagnostic (graceful failure instead of `abort`) would be friendlier than an assertion.

---

## 11. Suggested next steps

1. Implement Option 1 in `computeReductionLaneLayoutAndData` + `setupMultiReductionResultLayout` (InstData & Lane), reusing the Subgroup branch's consumer-matching structure.
2. Re-run `simple_mxfp_gemm_quantizeA_F4.mlir` through `--gpu-lower-to-xevm-pipeline` (workgroup); expect the reduction to become lane-local and the `[1,1]→[8,1]` convert to disappear.
3. Update `propagate-layout-inst-data.mlir` reduction CHECK lines (they currently encode the `[1,1,*]` behavior).
4. Decide on the broadcast-convert item (Section 10.3) and whether to land the two-load test edit alongside.

---

### Appendix — reproduction notes

- Repro: `mlir-opt mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F4.mlir --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri"` aborts (exit 134).
- The sibling `simple_mxfp_gemm_dequantizeB_F4.mlir` lowers cleanly (exit 0) — it has no A-side reduction, which isolates the reduction layout rule as the cause.
- PR measurement: the local playground branch is `merge-base + 343` commits; PR #203156 is `merge-base + 22`. The PR's four core files (`XeGPULayoutImpl.{h,cpp}`, `XeGPUPropagateLayout.cpp`, `XeGPUUnroll.cpp`) are byte-identical to the merge-base on the playground, so grafting the PR versions of just those files and rebuilding `mlir-opt` faithfully reproduces the PR's behavior without a full-tree rebuild. The `XeGPUDialect.cpp::expandDim` change in the PR is already present (identically) on the playground.
