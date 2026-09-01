# [mlir][XeGPU] Mixed-size `vector.shuffle` is scalarized: 70% of the emitted LLVM IR is `insertelement`/`extractelement`

**Labels:** `mlir`, `mlir:gpu`

**Status:** fixed.
**Fixed by:** [PR #217141](https://github.com/llvm/llvm-project/pull/217141), branch `xegpu-linearize-promote-shuffle`.
**Depends on:** nothing. Independent of the others; can land on its own.
**Validated:** the promoted shuffles are numerically correct. See #ISSUE-1.

## Summary

`gpu-lower-to-xevm-pipeline` does not run `vector::populateVectorShuffleLoweringPatterns`. As a
result every `vector.shuffle` whose two operands have *different* lengths is scalarized by
`VectorToLLVM` into one `extractelement` + one `insertelement` per result element.

For `simple_mxfp_gemm_quantizeA_F4.mlir` this turns 1024 shuffle ops into ~48k scalar ops, which is
**70% of the entire emitted LLVM IR**:

| | count |
|---|---|
| `llvm.insertelement` | 25850 |
| `llvm.extractelement` | 22253 |
| `llvm.shufflevector` | 1520 |
| total lines of LLVM IR | 68613 |

The upstream pattern that fixes this already exists and is fully general. It is simply not wired
into any production pipeline.

## Root cause

`mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp:1086` takes the fast path only when the
two operand types are *identical*:

```cpp
// For rank 0 and 1, where both operands have *exactly* the same vector
// type, there is direct shuffle support in LLVM. Use it!
if (rank <= 1 && v1Type == v2Type) {
  Value llvmShuffleOp = LLVM::ShuffleVectorOp::create(
      rewriter, loc, adaptor.getV1(), adaptor.getV2(),
      llvm::to_vector_of<int32_t>(mask));
  rewriter.replaceOp(shuffleOp, llvmShuffleOp);
  return success();
}

// For all other cases, insert the individual values individually.
...
for (int64_t extPos : mask) {
  ...
  Value extract = extractOne(...);
  insert = insertOne(...);
}
```

This is correct but unconditional: MLIR's `vector.shuffle` permits operands of different lengths
(the mask indexes the concatenation), whereas LLVM's `shufflevector` requires both operands to have
the same type. Without a prior legalization step the fallback fires.

The mixed-size shuffles are created by the linearizer itself:
`LinearizeVectorInsertStridedSlice` (`VectorLinearize.cpp:286`) and `LinearizeVectorInsert` (`:502`)
both emit `vector.shuffle(dest, src)` where `dest` is the wide vector and `src` the narrow inserted
chunk, so mixed sizes are guaranteed by construction.

## Measured breakdown

All `vector.shuffle` ops immediately before `convert-xegpu-to-xevm`:

```
  x1024 (vector<8xbf16>, vector<8xbf16>)              -> vector<1xbf16>        llvm.shufflevector
  x512  (vector<16xbf16>, vector<1xbf16>)             -> vector<16xbf16>       SCALARIZED
  x256  (vector<32xbf16>, vector<32xbf16>)            -> vector<4xbf16>        llvm.shufflevector
  x256  (vector<32xbf16>, vector<4xbf16>)             -> vector<32xbf16>       SCALARIZED
  x128  (vector<32xbf16>, vector<8xbf16>)             -> vector<32xbf16>       SCALARIZED
  x128  (vector<32xf8E8M0FNU>, vector<8xf8E8M0FNU>)   -> vector<32xf8E8M0FNU>  SCALARIZED
  x32   (vector<32xf8E8M0FNU>, vector<32xf8E8M0FNU>)  -> vector<32xf8E8M0FNU>  llvm.shufflevector
  x16   (vector<32xf8E8M0FNU>, vector<32xf8E8M0FNU>)  -> vector<2xf8E8M0FNU>   llvm.shufflevector
```

1024 mixed-size shuffles. Predicted `insertelement` count:

```
512 x 16 (result vector<16xbf16>)      =  8192
256 x 32 + 128 x 32 + 128 x 32         = 16384
                                 total = 24576
```

Measured 25850 total `llvm.insertelement`, of which exactly **16384** are on 32-component vectors
and 8192 on 16-component vectors -- an exact match, with the ~1274 remainder coming from other
sources. `extractelement` is slightly lower than predicted (22253 vs 24576) because extracts from
identical positions CSE.

## The fix already exists upstream

`mlir/lib/Dialect/Vector/Transforms/LowerVectorShuffle.cpp:44`,
`MixedSizeInputShuffleOpRewrite`, exposed as `vector::populateVectorShuffleLoweringPatterns`
(`LoweringPatterns.h:317`):

```
/// Lowers a `vector.shuffle` operation with mixed-size inputs to a new
/// `vector.shuffle` which promotes the smaller input to the larger vector size
/// and an updated version of the original `vector.shuffle`.
///
///     %0 = vector.shuffle %v1, %v2 [0, 2, 1, 3] : vector<2xf32>, vector<4xf32>
///
///   is lowered to:
///
///     %0 = vector.shuffle %v1, %v1 [0, 1, -1, -1] : vector<2xf32>, vector<2xf32>
///     %1 = vector.shuffle %0, %v2 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
///
/// Note: This transformation helps legalize vector.shuffle ops when lowering
/// to SPIR-V/LLVM, which don't support shuffle operations with mixed-size
/// inputs.
```

It handles both directions (smaller operand in v1 or v2) and remaps the mask accordingly.

**It has exactly one caller in the entire tree**, the test pass
`TestVectorShuffleLowering` (`mlir/test/lib/Dialect/Vector/TestVectorTransforms.cpp:1081`).
No production pipeline invokes it, including `GPUToXeVMPipeline.cpp`.

## Verification

```mlir
func.func @c16(%a: vector<16xbf16>, %b: vector<1xbf16>) -> vector<16xbf16> {
  %r = vector.shuffle %a, %b [0, 1, 2, 16, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
     : vector<16xbf16>, vector<1xbf16>
  return %r : vector<16xbf16>
}
func.func @v1smaller(%a: vector<2xf32>, %b: vector<4xf32>) -> vector<4xf32> {
  %r = vector.shuffle %a, %b [0, 2, 1, 3] : vector<2xf32>, vector<4xf32>
  return %r : vector<4xf32>
}
func.func @f8(%a: vector<32xf8E8M0FNU>, %b: vector<8xf8E8M0FNU>) -> vector<32xf8E8M0FNU> {
  %r = vector.shuffle %a, %b [32, 1, 2, 3, 33, 5, 6, 7, 34, 9, 10, 11, 35, 13, 14, 15,
                              36, 17, 18, 19, 37, 21, 22, 23, 38, 25, 26, 27, 39, 29, 30, 31]
     : vector<32xf8E8M0FNU>, vector<8xf8E8M0FNU>
  return %r : vector<32xf8E8M0FNU>
}
```

```
$ mlir-opt --convert-vector-to-llvm --reconcile-unrealized-casts
  insertelement + extractelement : 104
  shufflevector                  : 0

$ mlir-opt --test-vector-shuffle-lowering --convert-vector-to-llvm --reconcile-unrealized-casts
  insertelement + extractelement : 0
  shufflevector                  : 6
```

All three signatures present in the kernel are covered, including the `v1`-smaller direction.

## Fix

[PR #217141](https://github.com/llvm/llvm-project/pull/217141), branch `xegpu-linearize-promote-shuffle`.

`vector::populateVectorShuffleLoweringPatterns` already promotes the smaller operand to the size of
the larger one and remaps the mask; it is declared in
`mlir/include/mlir/Dialect/Vector/Transforms/LoweringPatterns.h` and, before this change, its only
caller was the `--test-vector-shuffle-lowering` test pass. So nothing had to be written -- the
patterns only had to be run.

Run them at the end of `xegpu-vector-linearize`, which is the pass that *produces* these shuffles,
so a mixed-size shuffle never escapes in the first place. That is a 17-line addition of a fourth
greedy pattern application to `XeGPUVectorLinearize.cpp`; the pass already includes
`LoweringPatterns.h` and already drives several `populateVector*LoweringPatterns` sets the same way,
and `MLIRXeGPUTransforms` already links `MLIRVectorTransforms`.

The whole change is therefore two files:

```
mlir/lib/Dialect/XeGPU/Transforms/XeGPUVectorLinearize.cpp | 17 ++
mlir/test/Dialect/XeGPU/xegpu-vector-linearize.mlir        | 84 +++++-----
```

No new pass, no new public API, no Vector-dialect change and no pipeline change.

### Fixing it at the producer is cheaper than fixing it later

Running the promotion inside the linearizer, before canonicalization, is not merely tidier. The
promoting shuffle usually folds into whatever produced the narrow operand, so it often costs nothing
at all:

* a `vector.broadcast` feeding the shuffle is emitted at the wide type directly, with no promoting
  shuffle left behind;
* a splat `arith.constant` is materialized at the wide width instead.

Over `mlir/test/Dialect/XeGPU/xegpu-vector-linearize.mlir` this takes the number of mixed-size
shuffles from **46 to 0**. Several existing expected outputs in that file change as a result, and the
new output is better IR rather than merely different, which is why the checks were updated rather
than worked around.

### Measured result on the kernel

| | before | after | |
|---|---|---|---|
| `llvm.insertelement` | 25850 | 1274 | |
| `llvm.extractelement` | 22253 | 1316 | |
| scalar ops total | **48103** | **2590** | **18.6x fewer** |
| total lines of LLVM IR | **68613** | **25636** | |

Mixed-size `vector.shuffle` reaching `gpu-to-llvm`: 1024 -> **0**, with 2768 equal-size shuffles in
their place. MLIR pipeline errors: 0. (Note it is `gpu-to-llvm`, not `convert-vector-to-llvm`, that
performs the scalarization; by the time `convert-vector-to-llvm` runs there are zero
`vector.shuffle` ops left.)

Regression testing: `mlir/test/Dialect/XeGPU`, `mlir/test/Dialect/Vector`,
`mlir/test/Conversion/XeGPUToXeVM`, `mlir/test/Conversion/VectorToLLVM`, `mlir/test/Dialect/GPU` --
191/191 pass. New test case in `xegpu-vector-linearize.mlir` covering an
`insert_strided_slice` of a narrow chunk into a wider tile, asserting that both shuffle operands end
up the same type and that no mixed-size shuffle remains.

This is a code-size and compile-time fix, orthogonal to `#ISSUE-6`: on its own it does not change
the backend outcome, it only makes the failure much easier to see in a 25k-line dump instead of a
69k-line one. `#ISSUE-6` is what unblocks the backend.

### Alternatives not taken

* **A new Vector-dialect pass.** An earlier version of this fix added a `lower-vector-shuffle` pass
  and scheduled it in `GPUToXeVMPipeline.cpp`. Promoting mixed-size shuffles is not obviously a
  general-purpose Vector transform yet -- the only known consumer is the XeVM path -- so a new pass
  in that dialect is not justified. Calling the existing patterns from the XeGPU pass that creates
  the shuffles achieves the same result with no new surface area.
* **Doing it in `VectorToLLVM`.** Arguably the `v1Type == v2Type` fallback in
  `ConvertVectorToLLVM.cpp` should perform the promotion inline, since scalarization is never the
  desired outcome for a 1-D shuffle. That would fix every target at once, but it is a behavioural
  change for all existing targets, so it is left as a separate upstream decision.

## Relationship to the other issues

* Independent of `#ISSUE-6` (vector width legalization). These shuffles are *data movement*, not
  compute; they are exempt from the elementwise unrolling proposed there, and the two fixes compose.
* This is very likely not a correctness blocker on its own -- `insertelement` into a wide vector is
  just register indexing and IGC should accept it. It is a compile-time, code-size and codegen
  quality problem, and it makes the post-`#ISSUE-6` IR much harder to inspect.
* This fix converts ~48k scalar ops into 32-component `llvm.shufflevector`, which the backend
  accepts. With both fixed, 944 `llvm.shufflevector`, 49 `llvm.bitcast`, 2 `poison` and 1
  `constant` on 32-component vectors survive and the module compiles. `vector<32xbf16>` (512 bits)
  and `vector<32xi8>` (256 bits) are fine as payloads.

## Measurement environment

`main` at `6523442d2efe`, which includes the merged PR #210837, plus the `#ISSUE-3` fix
(PR #215645, which now includes the folded #217104) and the `xegpu.lane_shuffle` -> XeVM lowering of PRs #215306 and #215303 (branch
`xegpu-mxfp-combined`). `mlir-opt` Release + `BUILD_SHARED_LIBS=ON`.
