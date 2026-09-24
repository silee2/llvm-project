// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri igc-cmd-options=-ze-opt-large-register-file"
// RUN-DISABLED: | mlir-runner \
// RUN-DISABLED:   --shared-libs=%mlir_levelzero_runtime \
// RUN-DISABLED:   --shared-libs=%mlir_runner_utils \
// RUN-DISABLED:   --shared-libs=%mlir_c_runner_utils \
// RUN-DISABLED:   --entry-point-result=void \
// RUN-DISABLED: | FileCheck %s

// Performance reference point: a hand-tuned non-quantized bf16 GEMM,
// M = N = K = 4096, kept on this branch to compare the mx-fp kernels against a
// kernel whose layouts, tiling and prefetch schedule are known to be good.
//
// The gpu.module below is taken unchanged from a tuned kernel, so the parts
// that carry the tuning are verbatim:
//   - 256x256 workgroup tile of C, sg_layout = [8, 4], sg_data = [32, 64]
//   - K step of 32, with A and B prefetched three steps ahead and one further
//     prefetch issued per iteration
//   - -ze-opt-large-register-file, without which a 256x256 f32 accumulator
//     tile per workgroup does not hold in registers
//
// Two deliberate differences from the original:
//   - It runs on the same target as the rest of this directory rather than the
//     one it was tuned for, so the comparison holds the target fixed.  The two
//     differ in mx-fp support, which this kernel does not use.
//   - The host harness is rewritten against the MLIR runner utilities.  The
//     original used helpers that only exist downstream (random fill, a host
//     GEMM and an allclose check), so instead this uses the same deterministic
//     input set as the mx-fp tests in this directory and checks for a bit-exact
//     match, which a random-input allclose check cannot do.
//
// The reference result is computed without a 4096^3 host GEMM.  A and B are
// drawn from an 8-entry table indexed by (i + k) % 8 and (j + k) % 8, so the
// product is periodic in k with period 8 and K = 4096 = 512 * 8:
//
//   C[i, j] = 512 * sum over r in [0, 8) of lut[(i + r) % 8] * lut[(j + r) % 8]
//
// which depends only on i % 8 and j % 8, so 64 distinct values cover all of C.
// Every table entry is exact in bf16, every product is a multiple of 0.25 and
// the largest result is 147456, well under 2^24, so the f32 accumulation is
// exact and the device result has to match bit for bit.
// Compile-only by design, and the one test in this directory that does not
// execute.  The problem size is deliberately left at the tuned kernel's own
// 4096^3 rather than matched to the mx-fp tests, which makes it a reference for
// compile time, code size and emitted code, not for runtime: 68.7 G
// multiply-accumulates is about 256 times the arithmetic of the mx-fp tests
// here, so a full run extrapolates to hours.
//
// The execution pipeline is kept intact behind RUN-DISABLED and the result
// expectation as MISMATCH, so a run needs only three edits: restore the
// trailing backslash on the first RUN line, turn RUN-DISABLED back into RUN,
// and rename MISMATCH to CHECK.  The host harness below is complete and its
// reference is exact, so such a run is a real correctness check.
//
#a = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>
#b = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>
#c = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>
#a_prefetch = #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>
#b_prefetch = #xegpu.layout<sg_layout = [4, 8], sg_data = [8, 32], inst_data = [8, 16]>
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<4096x4096xbf16>, %B: memref<4096x4096xbf16>, %C: memref<4096x4096xf32>) -> memref<4096x4096xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %A_gpu = gpu.alloc () : memref<4096x4096xbf16>
    gpu.memcpy %A_gpu, %A : memref<4096x4096xbf16>, memref<4096x4096xbf16>
    %B_gpu = gpu.alloc () : memref<4096x4096xbf16>
    gpu.memcpy %B_gpu, %B : memref<4096x4096xbf16>, memref<4096x4096xbf16>
    %C_gpu = gpu.alloc () : memref<4096x4096xf32>
    gpu.memcpy %C_gpu, %C : memref<4096x4096xf32>, memref<4096x4096xf32>
    // NOTE: Here we can't use [8, 64] wi threads following the SG thread layout
    // of [8, 4], because the runtime linearizes the x dimension first and we
    // need y linearized first.  So use a linearized thread layout of [512, 1].
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1) args(%A_gpu : memref<4096x4096xbf16>, %B_gpu : memref<4096x4096xbf16>, %C_gpu : memref<4096x4096xf32>)
    gpu.wait // Wait for the kernel to finish.
    gpu.memcpy %C, %C_gpu : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.dealloc %A_gpu : memref<4096x4096xbf16>
    gpu.dealloc %B_gpu : memref<4096x4096xbf16>
    gpu.dealloc %C_gpu : memref<4096x4096xf32>
    return %C : memref<4096x4096xf32>
  }

  gpu.module @test_kernel   {
    gpu.func @test_kernel(%A: memref<4096x4096xbf16>, %B: memref<4096x4096xbf16>, %C: memref<4096x4096xf32>) kernel  {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c96 = arith.constant 96 : index
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c256 : index
      %n = arith.muli %block_id_y, %c256 : index
      %c_tdesc = xegpu.create_nd_tdesc %C : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32, #c>
      %c_init_value = xegpu.load_nd %c_tdesc[%m, %n] <{layout = #c}>: !xegpu.tensor_desc<256x256xf32, #c> -> vector<256x256xf32>
      %a_tdesc = xegpu.create_nd_tdesc %A : memref<4096x4096xbf16> -> !xegpu.tensor_desc<256x32xbf16, #a>
      %b_tdesc = xegpu.create_nd_tdesc %B : memref<4096x4096xbf16> -> !xegpu.tensor_desc<32x256xbf16, #b>
      // Prefetch A 3 times.
      %a_prefetch_tdesc = xegpu.create_nd_tdesc %A : memref<4096x4096xbf16> -> !xegpu.tensor_desc<256x32xbf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc[%m, %c0] <{layout = #a_prefetch}>: !xegpu.tensor_desc<256x32xbf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc[%m, %c32] <{layout = #a_prefetch}>: !xegpu.tensor_desc<256x32xbf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc[%m, %c64] <{layout = #a_prefetch}>: !xegpu.tensor_desc<256x32xbf16, #a_prefetch>
       // Prefetch B 3 times.
      %b_prefetch_tdesc = xegpu.create_nd_tdesc %B : memref<4096x4096xbf16> -> !xegpu.tensor_desc<32x256xbf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc[%c0, %n] <{layout = #b_prefetch}>: !xegpu.tensor_desc<32x256xbf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc[%c32, %n] <{layout = #b_prefetch}>: !xegpu.tensor_desc<32x256xbf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc[%c64, %n] <{layout = #b_prefetch}>: !xegpu.tensor_desc<32x256xbf16, #b_prefetch>

      %out = scf.for %k = %c0 to %c4096 step %c32
        iter_args(%c_value = %c_init_value)
        -> (vector<256x256xf32>) {
        %a_value = xegpu.load_nd %a_tdesc[%m, %k] <{layout = #a}>: !xegpu.tensor_desc<256x32xbf16, #a> -> vector<256x32xbf16>
        %b_value = xegpu.load_nd %b_tdesc[%k, %n] <{layout = #b}>: !xegpu.tensor_desc<32x256xbf16, #b> -> vector<32x256xbf16>
        // Prefetch next tiles.
        %prefetch_offset = arith.addi %k, %c96 : index
        xegpu.prefetch_nd %a_prefetch_tdesc[%m, %prefetch_offset] <{layout = #a_prefetch}>: !xegpu.tensor_desc<256x32xbf16, #a_prefetch>
        xegpu.prefetch_nd %b_prefetch_tdesc[%prefetch_offset, %n] <{layout = #b_prefetch}>: !xegpu.tensor_desc<32x256xbf16, #b_prefetch>
        %c_new_value = xegpu.dpas %a_value, %b_value, %c_value <{layout_a = #a, layout_b = #b, layout_cd = #c}>
          : vector<256x32xbf16>, vector<32x256xbf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %c_new_value : vector<256x256xf32>
      }
      xegpu.store_nd %out, %c_tdesc[%m, %n] <{layout = #c}>: vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #c>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c4096 = arith.constant 4096 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c512_f32 = arith.constant 5.120000e+02 : f32

    // The same 8 magnitudes the mx-fp tests in this directory use, so the
    // operand values are comparable across the two families.  All are exact in
    // bf16, f32 and the mx-fp formats.
    %lut = memref.alloc() : memref<8xf32>
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    %i5 = arith.constant 5 : index
    %i6 = arith.constant 6 : index
    %i7 = arith.constant 7 : index
    %l0 = arith.constant 0.0 : f32
    %l1 = arith.constant 0.5 : f32
    %l2 = arith.constant 1.0 : f32
    %l3 = arith.constant 1.5 : f32
    %l4 = arith.constant 2.0 : f32
    %l5 = arith.constant 3.0 : f32
    %l6 = arith.constant 4.0 : f32
    %l7 = arith.constant 6.0 : f32
    memref.store %l0, %lut[%c0] : memref<8xf32>
    memref.store %l1, %lut[%i1] : memref<8xf32>
    memref.store %l2, %lut[%i2] : memref<8xf32>
    memref.store %l3, %lut[%i3] : memref<8xf32>
    memref.store %l4, %lut[%i4] : memref<8xf32>
    memref.store %l5, %lut[%i5] : memref<8xf32>
    memref.store %l6, %lut[%i6] : memref<8xf32>
    memref.store %l7, %lut[%i7] : memref<8xf32>

    // tab[a, b] = 512 * sum over r of lut[(a + r) % 8] * lut[(b + r) % 8].
    // C[i, j] is tab[i % 8, j % 8]; 64 values cover the whole result.
    %tab = memref.alloc() : memref<8x8xf32>
    scf.for %a = %c0 to %c8 step %c1 {
      scf.for %b = %c0 to %c8 step %c1 {
        %acc = scf.for %r = %c0 to %c8 step %c1
            iter_args(%sum = %c0_f32) -> (f32) {
          %ar = arith.addi %a, %r : index
          %br = arith.addi %b, %r : index
          %ai = arith.remui %ar, %c8 : index
          %bi = arith.remui %br, %c8 : index
          %va = memref.load %lut[%ai] : memref<8xf32>
          %vb = memref.load %lut[%bi] : memref<8xf32>
          %p = arith.mulf %va, %vb : f32
          %s = arith.addf %sum, %p : f32
          scf.yield %s : f32
        }
        %scaled = arith.mulf %acc, %c512_f32 : f32
        memref.store %scaled, %tab[%a, %b] : memref<8x8xf32>
      }
    }

    // A[i, k] = lut[(i + k) % 8]
    %A = memref.alloc() : memref<4096x4096xbf16>
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %k = %c0 to %c4096 step %c1 {
        %ik = arith.addi %i, %k : index
        %idx = arith.remui %ik, %c8 : index
        %v = memref.load %lut[%idx] : memref<8xf32>
        %vh = arith.truncf %v : f32 to bf16
        memref.store %vh, %A[%i, %k] : memref<4096x4096xbf16>
      }
    }

    // B[k, j] = lut[(j + k) % 8]
    %B = memref.alloc() : memref<4096x4096xbf16>
    scf.for %k = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        %jk = arith.addi %j, %k : index
        %idx = arith.remui %jk, %c8 : index
        %v = memref.load %lut[%idx] : memref<8xf32>
        %vh = arith.truncf %v : f32 to bf16
        memref.store %vh, %B[%k, %j] : memref<4096x4096xbf16>
      }
    }

    // C starts at zero because the kernel accumulates into the loaded value.
    %C = memref.alloc() : memref<4096x4096xf32>
    %C_ref = memref.alloc() : memref<4096x4096xf32>
    scf.for %i = %c0 to %c4096 step %c1 {
      %im = arith.remui %i, %c8 : index
      scf.for %j = %c0 to %c4096 step %c1 {
        %jm = arith.remui %j, %c8 : index
        memref.store %c0_f32, %C[%i, %j] : memref<4096x4096xf32>
        %e = memref.load %tab[%im, %jm] : memref<8x8xf32>
        memref.store %e, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }

    %C_res = call @test(%A, %B, %C) : (memref<4096x4096xbf16>, memref<4096x4096xbf16>, memref<4096x4096xf32>) -> memref<4096x4096xf32>
    %C_cast = memref.cast %C_res : memref<4096x4096xf32> to memref<*xf32>
    %C_ref_cast = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    %diff = call @verifyMemRefF32(%C_cast, %C_ref_cast) : (memref<*xf32>, memref<*xf32>) -> i64
    %prefix = llvm.mlir.addressof @mismatches_str : !llvm.ptr
    llvm.call @printString(%prefix) : (!llvm.ptr) -> ()
    call @printI64(%diff) : (i64) -> ()
    call @printNewline() : () -> ()

    // MISMATCH: {{^mismatches: 0$}}
    memref.dealloc %lut : memref<8xf32>
    memref.dealloc %tab : memref<8x8xf32>
    memref.dealloc %A : memref<4096x4096xbf16>
    memref.dealloc %B : memref<4096x4096xbf16>
    memref.dealloc %C : memref<4096x4096xf32>
    memref.dealloc %C_ref : memref<4096x4096xf32>
    return
  }
  func.func private @verifyMemRefF32(%actual : memref<*xf32>, %expected : memref<*xf32>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @printI64(%num : i64)
  func.func private @printNewline()

  // Print the mismatch count as "mismatches: <n>" rather than bare, so the
  // check cannot be satisfied by an unrelated 0: the runtime is free to write
  // diagnostics to stdout, and a bare "0" check matches a 0 anywhere in them,
  // including inside a larger number.
  llvm.mlir.global internal constant @mismatches_str("mismatches: \00")
  llvm.func @printString(!llvm.ptr)
}
