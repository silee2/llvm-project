# [mlir][xegpu] `XeGPUSgToLaneDistribute` silently ignores `applyPartialConversion` failure

**Labels:** `mlir`, `mlir:gpu`

**Status:** open. Diagnostics only; does not block compilation.

## Summary

`XeGPUSgToLaneDistributePass::runOnOperation()` discards the result of
`applyPartialConversion` and never calls `signalPassFailure()`. When any op fails to legalize,
dialect conversion **rolls the entire module back**, and the pass then reports success. The
still-subgroup-level IR flows into the rest of the pipeline, where it produces confusing
secondary errors far from the real cause.

`mlir/lib/Dialect/XeGPU/Transforms/XeGPUSgToLaneDistribute.cpp`:

```cpp
    target.addLegalOp<UnrealizedConversionCastOp>();
    (void)applyPartialConversion(root, target, std::move(patterns));   // <-- failure dropped
  }
  // Fold cancelling cast chains and erase dead casts.
  xegpu::cleanupUnrealizedConversionCasts(root, existingCasts);
  xegpu::removeTemporaryLayoutAttrs(getOperation());
```

## Why this matters

Measured on `mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F4.mlir`
(`--gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri"`):

* 1136 `xegpu.convert_layout` ops before the pass, **1136 after** — a bit-for-bit rollback.
* One genuinely unsupported op is reported; the other 1135 (including ~1040 that fold
  trivially) silently reappear in the output.
* The next diagnostic the user sees is
  `failed to legalize operation 'vector.multi_reduction' ... (vector<1x1x16xbf16>)` emitted by
  `xegpu-vector-linearize`, which marks rank>1 vectors illegal. That op is *not* broken — it
  lowers correctly when the pass succeeds. It only survives to that point because of the
  rollback.

This combination is actively misleading. It led to a bug report claiming that "compatible
`convert_layout` ops persist after sg-to-lane distribution, so `isCompatibleWith()` needs
enhancement", when in fact those ops fold correctly and the real defect was elsewhere.

## Suggested fix

```cpp
    if (failed(applyPartialConversion(root, target, std::move(patterns))))
      return signalPassFailure();
```

Two things to check when doing this:

1. Whether any in-tree test currently relies on the pass being a no-op on failure.
2. Whether the trailing `cleanupUnrealizedConversionCasts` / `removeTemporaryLayoutAttrs`
   should still run on the failure path (probably not — after a rollback they only strip
   attributes off IR that was never converted, which makes the dump harder to interpret).

Independently, it would help to make the partial-conversion diagnostics list *all* remaining
illegal ops rather than only the first — with five distinct unsupported layout conversions in
one kernel, discovering them one rebuild at a time is slow. A pass-local pre-walk that reports
every op the patterns will reject would be enough.
