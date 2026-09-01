# [mlir][XeGPU] No vector width legalization before `convert-xegpu-to-xevm`: 32-component `arith` ops reach the backend unlegalized

**Labels:** `mlir`, `mlir:gpu`

**Status:** fixed.
**Fixed by:** [PR #217131](https://github.com/llvm/llvm-project/pull/217131), branch `xegpu-legalize-vector-width`.
**Depends on:** #ISSUE-3 (this issue is only reachable once the XeGPU layer lowers cleanly) and #ISSUE-8, whose fix stays required -- see "Interaction with the `arith.truncf` splitting change".
**Validated:** the legalized kernel is numerically correct. See #ISSUE-1.

## Summary

After the XeGPU-layer blockers are resolved (see `#ISSUE-3`), the whole MLIR pipeline for
`mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F4.mlir` completes with zero
errors, but the resulting LLVM IR contains 32-component vector operations that the backend cannot
legalize:

```
LLVM ERROR: unable to legalize instruction:
  %67797:vfid(<32 x s32>) = G_FPEXT %67796:vfid(<32 x s16>) (in function: gemm_mxfp)
```

Nothing in `gpu-lower-to-xevm-pipeline` caps vector width. `xegpu-vector-linearize` flattens
per-lane tiles such as `vector<8x4xbf16>` into `vector<32xbf16>` with no width limit, and
`arith-expand-ops` then builds an entire arithmetic chain at that width.

## Reproducer

```
mlir-opt --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
  mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F4.mlir
```

Requires the fix from `#ISSUE-3` to get this far.

## Analysis: two disjoint families of wide vectors

There are 964 SSA values with more than 16 vector components immediately before
`convert-xegpu-to-xevm`. They split cleanly into two groups, and only one of them is a problem.

### Family A - payload vectors. Already correct, must not be touched.

| type | components | total bits | reaches LLVM as |
|---|---|---|---|
| `vector<32xf4E2M1FN>` (dpas `$a`) | 32 | 128 | `vector<8xi16>` |
| `vector<64xf4E2M1FN>` (dpas `$b`) | 64 | 256 | `vector<8xi32>` |
| `vector<32xf8E8M0FNU>` | 32 | 256 | bitcast away |
| `vector<32xi8>` (load/store payload) | 32 | 256 | `vector<8xi32>` |

These are wide only in *component count*; they are <= 256 bits and are bitcast down to <= 8
components inside the `load_nd` / `store_nd` / `dpas_mx` lowerings. Evidence from the final LLVM
IR (`--mlir-print-ir-before=gpu-module-to-binary`):

* `f4E2M1FN`: **0** remaining occurrences
* `f8E8M0FNU`: **1** remaining occurrence, a scalar `llvm.mlir.constant`
* `llvm.bitcast vector<32xi8> -> vector<8xi32>` x32, and `vector<8xi32> -> vector<32xi8>` x17

Confirmed against the emitted builtin:

```
__builtin_IB_sub_group16_bdpas_f_f_e2m1_e2m1_8_8(
    vector<8xf32>,   // acc
    vector<8xi16>,   // $a = 128 bits
    vector<8xi32>,   // $b = 256 bits
    vector<2xi8>, vector<2xi8>) -> vector<8xf32>
```

Splitting any of these would be an active pessimization.

### Family B - real arithmetic. This is the bug.

The `f8E8M0` scale expansion produced by `arith-expand-ops`, plus the bf16 scaling divide:

```
xegpu.load_nd    -> vector<32xf8E8M0FNU>   (256b)
arith.bitcast    -> vector<32xi8>          (256b)
arith.extui      -> vector<32xi32>        (1024b)
arith.shli       -> vector<32xi32>        (1024b)
arith.cmpi       -> vector<32xi1>
arith.select     -> vector<32xi32>        (1024b)
arith.bitcast    -> vector<32xf32>        (1024b)   <-- same length, NOT a narrowing
arith.truncf     -> vector<32xbf16>        (512b)
arith.divf       -> vector<32xbf16>        (512b)
arith.truncf     -> vector<32xf4E2M1FN>    (128b)
xegpu.dpas_mx
```

Narrowing happens only at the very last hop. Everything upstream reaches LLVM still 32-component:

| LLVM op | type | count |
|---|---|---|
| `llvm.fpext` | `vector<32xbf16>` -> `vector<32xf32>` | 64 |
| `llvm.fdiv` | `vector<32xf32>` | 32 |
| `llvm.fptrunc` | `vector<32xf32>` -> `vector<32xbf16>` | 32 |
| `llvm.zext` | `vector<32xi8>` -> `vector<32xi32>` | 32 |
| `llvm.shl` | `vector<32xi32>` | 32 |
| `llvm.select` | `vector<32xi1>, vector<32xi32>` | 32 |
| `llvm.icmp` | -> `vector<32xi1>` | 32 |
| `llvm.bitcast` | `vector<32xi32>` -> `vector<32xf32>` | 32 |
| `llvm.shufflevector` | `32xbf16` / `32xf32` / `32xi8` | 368 |
| `llvm.insertelement` / `llvm.extractelement` | on 32-component vectors | 16384 / 10906 |

The reported `G_FPEXT <32 x s16> -> <32 x s32>` is row 1: IGC's own legalization of
`fdiv <32 x bfloat>` (the scaling divide) into fpext / fdiv / fptrunc.

### Why "it all ends in a bitcast anyway" does not rescue this

A tempting reading is that the width is an artifact and the terminal bitcast absorbs it. That is
true for family A and false for family B. At the XeGPU level there is **not a single narrowing
bitcast** anywhere in the wide subgraph -- every bitcast is length-preserving or length-*increasing*:

```
arith.bitcast  vector<32xf8E8M0FNU> -> vector<32xi8>        x32  (same length)
arith.bitcast  vector<32xi32>       -> vector<32xf32>       x32  (same length)
vector.bitcast vector<32xi8>        -> vector<64xf4E2M1FN>  x16  (longer)
```

The narrowing for family A is injected *inside* the `load_nd` / `store_nd` / `dpas_mx` lowering
patterns, not in the visible chain. Family B has no such escape and must actually be split.

### A bit-width threshold is not the right discriminator

`vector<32xbf16>` is exactly 512 bits, so a "> 512 bits" rule declares the failing `fpext`'s source
legal and we still fail. Conversely `vector<64xf4E2M1FN>` is only 256 bits but has 64 components
and must be left alone. The two families are not separated by bit width.

## The pattern in family B

Family B is remarkably uniform:

```
=== FAMILY B (compute) ops ===
  x32   arith.truncf     (vector<32xbf16>) -> vector<32xf4E2M1FN>
  x32   arith.bitcast    (vector<32xi32>) -> vector<32xf32>
  x32   arith.extui      (vector<32xi8>) -> vector<32xi32>
  x32   arith.shli       (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
  x32   arith.truncf     (vector<32xf32>) -> vector<32xbf16>
  x32   arith.cmpi       (vector<32xi8>, vector<32xi8>) -> vector<32xi1>
  x32   arith.select     (vector<32xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
  x32   arith.bitcast    (vector<32xf8E8M0FNU>) -> vector<32xi8>
  x32   arith.divf       (vector<32xbf16>, vector<32xbf16>) -> vector<32xbf16>
  TOTAL 288
  dialects: {'arith'}
```

* **288 ops, 100% `arith` dialect, 100% single-result, 100% elementwise.**
* Exactly **two wide entry points**: `vector.shuffle -> vector<32xf8E8M0FNU>` (x32) and
  `vector.shuffle -> vector<32xbf16>` (x32), plus splat constants (x96 uses).
* Exactly **one wide exit point**: `arith.truncf -> vector<32xf4E2M1FN> -> xegpu.dpas_mx` (x32).

This means family B can be handled uniformly with **no new patterns**, because every op derived from
`Arith_Op` already carries what `vector::UnrollElementwisePattern` needs
(`mlir/include/mlir/Dialect/Arith/IR/ArithOps.td:33`):

```tablegen
class Arith_Op<string mnemonic, list<Trait> traits = []> :
    Op<Arith_Dialect, mnemonic,
       traits #
       [DeclareOpInterfaceMethods<VectorUnrollOpInterface>, NoMemoryEffect] #
       ElementwiseMappable.traits>;
```

and `UnrollElementwisePattern` matches exactly on that
(`mlir/lib/Dialect/Vector/Transforms/VectorUnroll.cpp:476`):

```cpp
if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
  return failure();
```

Conveniently, `Arith_ConstantOp` is declared with plain `Op<Arith_Dialect, "constant", ...>` rather
than `Arith_Op` (`ArithOps.td:223`), so constants are *not* matched; the splats instead fold through
the pattern's `createOrFold<vector::ExtractStridedSliceOp>` on operands.

Equally important, the family A ops are all *non*-elementwise -- `vector.shuffle`,
`vector.bitcast` (length-changing), `xegpu.load_nd`, `xegpu.store_nd`, `xegpu.dpas_mx` -- so the same
predicate leaves them untouched for free. The two families are separated exactly by
`hasElementwiseMappableTraits`.

## Design

Mirror what SPIR-V already does. `mlir/lib/Dialect/SPIRV/Transforms/SPIRVConversion.cpp:1478`:

```cpp
std::optional<SmallVector<int64_t>>
mlir::spirv::getNativeVectorShape(Operation *op) {
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = dyn_cast<VectorType>(op->getResultTypes()[0])) {
      if (vecType.getRank() == 0)
        return std::nullopt;
      SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = mlir::spirv::getComputeVectorSize(vecType.getShape().back());
      return nativeSize;
    }
  }
  ...
}
```

driven from `mlir::spirv::unrollVectorsInFuncBodies` (`:1508`) via
`vector::UnrollVectorOptions().setNativeShapeFn(...)` + `populateVectorUnrollPatterns`.

The XeVM version is the same shape, with the component cap coming from uArch instead of the
hardcoded Vulkan-oriented `{4,3,2}` in `getComputeVectorSize`.

### Rule

> Cap the trailing dimension of every `Elementwise`-mappable op at N components, **except** ops with
> a sub-byte non-`i1` operand or result. Leave every non-elementwise op alone; those are payload
> movement and are bounded by register footprint, not component count.

For the current SPIR-V-backed hardware N = 16. The sub-byte exemption is justified in
"Sub-byte types are exempt" below.

### Empirical validation of the mechanism

Distilling family B into a standalone function and running the *existing* SPIR-V unroller
(`--test-spirv-vector-unrolling`, cap 4) over it:

```mlir
func.func @famB(%ms: memref<32xf8E8M0FNU>, %ma: memref<32xbf16>, %mo: memref<32xf4E2M1FN>) {
  %ci = arith.constant 0 : index
  %scale = vector.load %ms[%ci] : memref<32xf8E8M0FNU>, vector<32xf8E8M0FNU>
  %a = vector.load %ma[%ci] : memref<32xbf16>, vector<32xbf16>
  %c127 = arith.constant dense<127> : vector<32xi8>
  %c23  = arith.constant dense<23> : vector<32xi32>
  %cnan = arith.constant dense<2143289344> : vector<32xi32>
  %0 = arith.bitcast %scale : vector<32xf8E8M0FNU> to vector<32xi8>
  %1 = arith.cmpi eq, %0, %c127 : vector<32xi8>
  %2 = arith.extui %0 : vector<32xi8> to vector<32xi32>
  %3 = arith.shli %2, %c23 : vector<32xi32>
  %4 = arith.select %1, %cnan, %3 : vector<32xi1>, vector<32xi32>
  %5 = arith.bitcast %4 : vector<32xi32> to vector<32xf32>
  %6 = arith.truncf %5 : vector<32xf32> to vector<32xbf16>
  %7 = arith.divf %a, %6 : vector<32xbf16>
  %8 = arith.truncf %7 : vector<32xbf16> to vector<32xf4E2M1FN>
  vector.store %8, %mo[%ci] : memref<32xf4E2M1FN>, vector<32xf4E2M1FN>
  return
}
```

result:

```
16 vector.extract_strided_slice     16 arith.truncf     16 arith.bitcast
 8 vector.insert_strided_slice       8 arith.shli        8 arith.select
                                     8 arith.extui       8 arith.divf
 2 vector.load                       8 arith.cmpi        5 arith.constant
 1 vector.store
```

Every arithmetic op is narrowed by the trait-based predicate alone; the only remaining
32-component values are the `vector.load` / `vector.store` payloads (standing in for the
shuffle-in / dpas-out boundary), glued with `extract_strided_slice` / `insert_strided_slice`.
Splat constants folded to narrow splats. No new patterns were written.

This demonstrates the *mechanism* -- stock unroll patterns plus a native-shape function. It is not
the final predicate: the shipped pass additionally exempts sub-byte element types, so the trailing
`arith.truncf ... to vector<32xf4E2M1FN>` shown narrowed here is in fact left alone. See
"Sub-byte types are exempt".

### Pass ordering constraint

`mlir/lib/Dialect/GPU/Pipelines/GPUToXeVMPipeline.cpp`:

| line | pass |
|---|---|
| 100 | `xegpu-vector-linearize` - flattens `vector<8x4xbf16>` -> `vector<32xbf16>`; no width cap |
| 111-116 | `arith-expand-ops` (`includeF8E8M0=true`) - **creates** family B |
| 121 | `convert-xegpu-to-xevm` |

The legalizer must run **after** line 116 and before line 121; an earlier pass cannot see these ops
because they do not exist yet.

An alternative is to cap the width at line 100 so family B is never created at 32 components in the
first place. That is arguably the more principled root fix, but it is a larger change: it would
require the linearizer to be uArch-aware and to re-concatenate at the `dpas_mx` boundary.

### Missing uArch hook

`mlir/include/mlir/Dialect/XeGPU/uArch/uArchBase.h` has no maximum-vector-length query. The closest
existing entries are unrelated quantities and should not be reused:

* `getSubgroupSize()` -> 16. Coincidentally equal, semantically unrelated.
* `getMaxLaneAccessSizeBytes()` -> 16 **bytes**, a block-IO limit.
* `getGeneralPackedFormatBitSize()`.

Also note `kXeVMExtfTruncfNumElems = 16` in `XeGPUToXeVM.cpp` is an *instruction granularity*
(`__builtin_IB_bftof_16`, `convert_half16`, `__builtin_IB_hftobf8_16`), again a third distinct
quantity that happens to be 16.

## Interaction with the `arith.truncf` splitting change

A separate change (#ISSUE-8, branch `xegpu-truncf-split-wide`) teaches `TruncfToXeVMPattern` to accept
`arith.truncf` whose width is any multiple of `kXeVMExtfTruncfNumElems`, slicing it into 16-element
`xevm.truncf` groups. That is needed because the kernel produces
`arith.truncf : vector<32xbf16> to vector<32xf4E2M1FN>`.

The two changes are complementary, not alternatives. The sub-byte exemption described below
deliberately leaves `arith.truncf : vector<32xbf16> to vector<32xf4E2M1FN>` at 32 components (32
such ops, measured), so `TruncfToXeVMPattern` must accept more than one instruction group's worth
of elements regardless of what this pass does. Dropping `xegpu-truncf-split-wide` reintroduces a
hard pipeline failure:

```
cannot be converted to LLVM IR: ... arith.truncf : (vector<32xbf16>) -> vector<32xf4E2M1FN>
```

because `TruncfToXeVMPattern` leaves the wide op unconverted and nothing downstream can translate
`f4E2M1FN`. `xegpu-truncf-split-wide` is load-bearing and must be kept.

`arith.extf` carries the same `== 16` restriction. It is now reachable by the same reasoning: an
`arith.extf` from a sub-byte type is exempted from unrolling, so a wide one would hit the same
limitation. None appears in this kernel, but a dequantizing kernel would need the matching fix.

## Fix

[PR #217131](https://github.com/llvm/llvm-project/pull/217131), branch `xegpu-legalize-vector-width`.

New `xegpu-legalize-vector-width` pass driving the stock
`vector::populateVectorUnrollPatterns` with a native-shape function, scheduled in
`GPUToXeVMPipeline.cpp` immediately after `arith-expand-ops`. No new rewrite patterns were needed.

The limit lives in `mlir/include/mlir/Dialect/XeGPU/uArch/uArchBase.h` as
`xegpu::uArch::kDefaultMaxVectorComponents = 16`, alongside the other microarchitectural queries
and documented as a property of the SPIR-V-backed chips rather than of XeVM, so it can become a
virtual `uArch` query later without touching callers. A `max-vector-components` pass option
overrides it; the pass resolves `0` to the target default in one place.

Widths that are not a multiple of the limit fall back to the largest divisor that fits (24 -> 12),
so unrolling never produces a ragged tail.

### Sub-byte types are exempt

The trait-based predicate alone is not sufficient. `arith.truncf : vector<32xbf16> to
vector<32xf4E2M1FN>` is elementwise, so a predicate based on the trait alone splits it and then
glues the halves back together with `insert_strided_slice` on `f4E2M1FN`, materializing sub-byte
vector data movement. That introduces **225** `i4` vector references where there were **zero**, and
the SPIR-V backend rejects it:

```
LLVM ERROR: incompatible result and operand types in a bitcast
  (in SPIRVTranslateModule)
```

Sub-byte, non-boolean element types are packed payloads, not compute values: they are bit-packed
into byte vectors when lowered and SPIR-V has no corresponding scalar type. An elementwise op
producing or consuming one is a quantization boundary whose own lowering already handles width.
The pass therefore skips any op with a sub-byte operand or result. `i1` is deliberately excluded
from that exemption, since vector masks are ordinary compute values and the `arith.cmpi` /
`arith.select` pair in the scale expansion must still be split.

### Measured result

The kernel compiles end to end: `mlir-opt` exits 0 with empty stderr and emits a real
`gpu.binary @kernel` ELF object. It also *runs* correctly -- the kernel XPASSes with 0
mismatching elements, so the unrolling and the sub-byte exemption are numerically sound and not
merely accepted by the backend. See #ISSUE-1.

| | before | after |
|---|---|---|
| backend | `LLVM ERROR: unable to legalize G_FPEXT <32 x s16> -> <32 x s32>` | compiles |
| lines of LLVM IR | 68613 | 25380 |
| 32-component arithmetic | `fpext` 64, `fdiv` 32, `fptrunc` 32, `zext` 32, `shl` 32, `select` 32, `icmp` 32, `bitcast` 32 | **none** |
| 32-component ops remaining | - | `shufflevector` 944, `bitcast` 49, `poison` 2, `constant` 1 (all data movement) |
| wide vector types remaining | 8 distinct | `vector<32xbf16>` (512b payload), `vector<32xi8>` (256b payload) |
| `i4` references | 0 | 0 |

Regression testing: `mlir/test/Dialect/XeGPU`, `mlir/test/Dialect/Vector`,
`mlir/test/Conversion/XeGPUToXeVM`, `mlir/test/Conversion/XeVMToLLVM`,
`mlir/test/Conversion/VectorToLLVM`, `mlir/test/Dialect/GPU`, `mlir/test/Dialect/Arith` --
227/227 pass. New test: `mlir/test/Dialect/XeGPU/legalize-vector-width.mlir`.

Note this measurement is taken together with the `#ISSUE-7` fix; the two are independent but were
validated on the same integration branch.

The 32-component *data movement* left over after family B is legalized is accepted by the
backend as-is; only compute needed capping.

## Open questions

1. Is the correct cap a flat 16 components for all element types, or per-element-type? A flat 16 is
   the safe choice given the `<32 x bfloat>` counterexample above.
2. Should this be a standalone `xegpu-legalize-vector-width` pass, or an extra stage inside
   `xegpu-vector-linearize` re-run after `arith-expand-ops`?
3. The pass currently keys off the trailing dimension only. That is sufficient because
   `ElementwiseMappable` guarantees matching shapes, but if the predicate is ever widened beyond
   elementwise ops this needs revisiting.

## Measurement environment

`main` at `6523442d2efe`, which includes the merged PR #210837, plus the `#ISSUE-3` fix
(PR #215645, which now includes the folded #217104) and the `xegpu.lane_shuffle` -> XeVM lowering of PRs #215306 and #215303 (branch
`xegpu-mxfp-combined`). `mlir-opt` Release + `BUILD_SHARED_LIBS=ON`.
