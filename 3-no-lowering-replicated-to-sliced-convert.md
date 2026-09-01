# [mlir][xegpu] sg-to-lane: no lowering for `convert_layout` from a fully replicated layout to a partially replicated (sliced) target

**Labels:** `mlir`, `mlir:gpu`

**Status:** fixed.
**Fixed by:** [PR #215645](https://github.com/llvm/llvm-project/pull/215645), branch `xegpu-convert-layout-broadcast-redistribute`. The generalization to non-equal divisors that was PR #217104 (branch `xegpu-convert-layout-broadcast-divisor`) has been folded into #215645, and #217104 is closed.
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

## Known limits of the implementation

Recorded from the review of #215645, in rough order of how likely they are to matter.

**One shuffle per moved element; no packing.** `gpu.shuffle` exchanges a whole 32-bit lane word,
but the lowering moves one element per shuffle, so an `f8E8M0FNU` element uses 8 of 32 bits. A
fragment with four bytes to move emits four shuffles where one would do. The sibling pattern in
the same file, `shuffleDataAsLaneLayoutChange`, already packs into `vector<Nxi32>` and issues
`vectorBitWidth / 32` shuffles.

This cannot be delegated to `xegpu.lane_shuffle` or `xevm.bitcast_shuffle`, even though both do
pack sub-word data: they are bit-preserving bijections (`AllTypesMatch<["source", "result"]>` on
the former, "total number of bits of `res` must equal the total number of bits of `src`" on the
latter), whereas this conversion is a size-changing gather from replicated data -- 16 lanes x 8
elements in, 16 x 2 out. Packing would have to happen inside the pattern: group element sources
that share a donor and whose fragment indices form a contiguous, aligned run, and move a word.

None of the three tests would benefit. In the partial case a lane needs exactly one element from
the other donor group, so the current form costs 1 shuffle where broadcasting the donor's whole
8-byte fragment would cost 2; the fully broadcast case emits no shuffle at all; and the
sliced-target case reads indices strided by 2, which no run covers. Choosing between the
strategies needs a cost model the pass does not have.

**Non-unit `lane_data` is rejected.** `isBroadcastRedistribution` requires the two layouts to
agree on `lane_data`, but `computeOwnedCoords` then rejects anything that is not all-ones, because
`computeStaticDistributedCoords` gives distribution-unit *starts* rather than every element. The
unit would have to be expanded, which is what the file-static `expandBlockCoords` in
`XeGPUDialect.cpp` does for layout comparison; it is not exposed in a header.

**Conversions that differ in both `lane_layout` and `lane_data` are handled by nothing.** The
repack path requires equal effective `lane_layout`, this pattern requires equal effective
`lane_data`, so a conversion differing in both falls through to
`lowering incompatible convert_layout not yet supported`. Since a `lane_data`-only repack is free
in hardware -- it lowers to `xegpu.lane_shuffle` -- the natural handling is composition:
redistribute to the target `lane_layout` keeping the input's `lane_data`, then one `lane_shuffle`.
No kernel in hand produces such a conversion, so this is speculative until one does.

**Lanes outside the target's active set may hold stale elements.** Dropping the shuffle when the
donor is the lane itself is only sound because nothing reads the lanes past `numActiveLanes`. The
emitted extract is uniform, so those lanes evaluate it against their own fragment and can land on
a different element. The first revision of the pattern shuffled unconditionally, which also
repaired them.

**The `needsShuffle` predicate cannot fire.** With the range narrowed to the active lanes it is
tautological: `needed[slot]` is *defined* as the index of the wanted element in the fragment of
lane `slot + delta`, so at `delta == 0` the check
`inputOwned[slot][index->at(slot)] == targetOwned[slot][pos]` holds by construction, and the fit
has already verified `index->at(slot) == needed[slot]`. So `needsShuffle` reduces to
`donorDelta != 0`. Either simplify it to that, or restore the full-subgroup range -- which is the
same decision as the previous point.
