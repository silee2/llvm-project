# [mlir][xegpu] array-length narrowing only ever targets the subgroup size, so `lane_data > 1` descriptors keep their wide FCD

**Labels:** `mlir`, `mlir:gpu`

**Status:** open. Missed optimization, not a correctness problem.
**Related:** supersedes the local fix that used to live on branch `xegpu-array-length-lane-data`; see "History" below.

## Summary

`XeGPUArrayLengthOptimization` rewrites a descriptor whose fastest-changing dimension (FCD) is a
multiple of the subgroup size into a narrower descriptor with the factor folded into
`array_length`. It only ever considers the *subgroup size* as the new FCD:

```cpp
/// TODO: Currently, we are only allowing subgroupSize as our new FCD for LANE
/// level distribution simplicity. But it can be different, and in the future,
/// we can add that support.
static int64_t computeArrayLength(int64_t fcdSize, int64_t subgroupSize) {
```

When the descriptor's layout has `lane_data > 1` on the FCD, that candidate is not distributable,
so the rewrite is declined outright:

```cpp
    SmallVector<int64_t> newShape = {shape[0], shape[1] / arrayLength};
    if (auto layout = tdescType.getLayoutAttr();
        layout && !layout.isDistributable(newShape))
      return failure();
```

For `tensor_desc<8x32xf8E5M2, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>>` the
candidate FCD is 16, each lane takes 2 elements along the FCD so the layout needs 32, and the
descriptor is left alone. No `array_length` is used, even though narrowing to **32** would have
been legal and would have folded a factor of 2 into `array_length` for a `8x64` descriptor.

This is exactly the shape the mx-fp8 operands have, so those loads never benefit from
`array_length` while the equivalent f16 operands (with `lane_data = [1, 1]`) do.

## Suggested fix

Use the distribution unit along the FCD, `lane_layout[FCD] * lane_data[FCD]`, as the narrowing
granularity instead of the subgroup size, which is what the `TODO` above anticipates:

```cpp
    int64_t fcdUnit = subgroupSize;
    if (auto layout = tdescType.getLayoutAttr()) {
      SmallVector<int64_t> laneLayout = layout.getEffectiveLaneLayoutAsInt();
      SmallVector<int64_t> laneData = layout.getEffectiveLaneDataAsInt();
      if (laneLayout.size() == 2 && laneData.size() == 2)
        fcdUnit = laneLayout[1] * laneData[1];
    }
```

The existing `isDistributable(newShape)` guard then still backs it up, so a granularity that does
not work out is declined as before rather than asserting. Descriptors with `lane_data = 1` on the
FCD -- every case in tree before the mx-fp types -- are unaffected, since `fcdUnit` reduces to the
subgroup size.

Needs a test with a non-unit FCD `lane_data` on both sides: one descriptor that can now be
narrowed (`8x64`, `lane_data = [1, 2]` -> `8x32` with `array_length = 2`) and one that still
cannot (`8x32`, which is already one distribution unit wide).

## History

This started as a crash: the pass narrowed to the subgroup size, carried the layout over
unchanged, and built the result with `TensorDescType::get` rather than a verifier-reporting path,
so `cannot distribute [8, 16] using ...` asserted instead of declining. The local fix on
`xegpu-array-length-lane-data` widened the granularity as above, which fixed the crash *and*
kept the optimization.

Upstream then rewrote the pass and fixed the same crash the other way, by adding the
`isDistributable(newShape)` check and declining. So the crash is gone in tree, and what remains
is only the missed optimization described here. Keeping the local fix on top of the upstream one
is not viable as is: it makes the pass narrow a descriptor that upstream's own
`test_incompatible_descriptor_layout` asserts is left alone, so the branch fix was dropped from
the integration branch in favour of upstream's version.
