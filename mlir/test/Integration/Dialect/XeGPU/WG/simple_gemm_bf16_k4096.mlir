// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// Non-quantized bf16 counterpart of simple_mxfp_gemm.mlir, for comparing a
// quantized (dpas_mx) gemm against a plain dpas gemm at the same problem size.
//
// Identical to simple_mxfp_gemm.mlir in:
//   - problem size            M = 256, N = 256, K = 4096, C is f32
//   - dispatch                blocks (8, 8, 1), threads (64, 1, 1)
//   - workgroup tile of C     32x32, sg_layout = [2, 2], sg_data = [16, 16]
//   - reference result        A and B are all ones, so C = K = 4096
//
// Deliberately different: the K step is 256 rather than 1024. The mxfp kernel
// can afford a 1024-deep K step because fp4 is 4 bits, giving an 8 KB per
// subgroup operand tile. At bf16 the same step would need four times the
// registers and spill, which would make the comparison measure spilling rather
// than dpas throughput. A K step of 256 gives bf16 the same 8 KB per subgroup
// tile, so register pressure matches and only the arithmetic differs.
#a = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 256], inst_data = [8, 16]>
#b = #xegpu.layout<sg_layout = [2, 2], sg_data = [256, 16], inst_data = [16, 16]>
#c = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 16]>

module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @gemm_bf16(%arg0: memref<256x4096xbf16>, %arg1: memref<4096x256xbf16>, %arg2: memref<256x256xf32>) kernel {
      %c0 = arith.constant 0 : index
      %mstep = arith.constant 32 : index
      %nstep = arith.constant 32 : index
      %kstep = arith.constant 256 : index
      %kbound = arith.constant 4096 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %mstep : index
      %n = arith.muli %block_id_y, %nstep : index

      %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<256x4096xbf16> -> !xegpu.tensor_desc<32x256xbf16>
      %b_tdesc = xegpu.create_nd_tdesc %arg1 : memref<4096x256xbf16> -> !xegpu.tensor_desc<256x32xbf16>

      // Load initial C
      %cd_tdesc = xegpu.create_nd_tdesc %arg2 : memref<256x256xf32> -> !xegpu.tensor_desc<32x32xf32, #c>
      %c_init = xegpu.load_nd %cd_tdesc[%m, %n] {layout = #c}: !xegpu.tensor_desc<32x32xf32, #c> -> vector<32x32xf32>

      %res = scf.for %k = %c0 to %kbound step %kstep
        iter_args(%c_partial = %c_init) -> (vector<32x32xf32>) {
        %a = xegpu.load_nd %a_tdesc[%m, %k] {layout = #a}: !xegpu.tensor_desc<32x256xbf16> -> vector<32x256xbf16>
        %b = xegpu.load_nd %b_tdesc[%k, %n] {layout = #b}: !xegpu.tensor_desc<256x32xbf16> -> vector<256x32xbf16>
        %new_c_partial = xegpu.dpas %a, %b, %c_partial
              {layout_a = #a, layout_b = #b, layout_cd = #c}
            : vector<32x256xbf16>, vector<256x32xbf16>, vector<32x32xf32> -> vector<32x32xf32>
        scf.yield %new_c_partial : vector<32x32xf32>
      }

      // store_nd with offset
      xegpu.store_nd %res, %cd_tdesc[%m, %n] {layout = #c} : vector<32x32xf32>, !xegpu.tensor_desc<32x32xf32, #c>
      gpu.return
    }
  }

  func.func @test(%a: memref<256x4096xbf16>, %b: memref<4096x256xbf16>, %c: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index

    %memref_a = gpu.alloc() : memref<256x4096xbf16>
    gpu.memcpy %memref_a, %a : memref<256x4096xbf16>, memref<256x4096xbf16>

    %memref_b = gpu.alloc() : memref<4096x256xbf16>
    gpu.memcpy %memref_b, %b : memref<4096x256xbf16>, memref<4096x256xbf16>

    %memref_c = gpu.alloc() : memref<256x256xf32>
    gpu.memcpy %memref_c, %c : memref<256x256xf32>, memref<256x256xf32>

    gpu.launch_func @kernel::@gemm_bf16 blocks in (%c8, %c8, %c1) threads in (%c64, %c1, %c1)
    args(%memref_a : memref<256x4096xbf16>, %memref_b : memref<4096x256xbf16>, %memref_c : memref<256x256xf32>)
    gpu.dealloc %memref_a : memref<256x4096xbf16>
    gpu.dealloc %memref_b : memref<4096x256xbf16>

    %res = memref.alloc() : memref<256x256xf32>
    gpu.memcpy %res, %memref_c : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc %memref_c : memref<256x256xf32>
    return %res : memref<256x256xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c4K = arith.constant 4096 : index
    %c0f32 = arith.constant 0.0 : f32
    %c1bf16 = arith.constant 1.0 : bf16

    // The 8 magnitudes e2m1 can represent, which are exact in bf16 and f32 too,
    // so this test shares its input set and reference result with the fp4 and
    // fp8 tests.
    %lut = memref.alloc() : memref<8xf32>
    %lutb = memref.alloc() : memref<8xbf16>
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    %i5 = arith.constant 5 : index
    %i6 = arith.constant 6 : index
    %i7 = arith.constant 7 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 0.5 : f32
    %f2 = arith.constant 1.0 : f32
    %f3 = arith.constant 1.5 : f32
    %f4 = arith.constant 2.0 : f32
    %f5 = arith.constant 3.0 : f32
    %f6 = arith.constant 4.0 : f32
    %f7 = arith.constant 6.0 : f32
    memref.store %f0, %lut[%c0] : memref<8xf32>
    memref.store %f1, %lut[%i1] : memref<8xf32>
    memref.store %f2, %lut[%i2] : memref<8xf32>
    memref.store %f3, %lut[%i3] : memref<8xf32>
    memref.store %f4, %lut[%i4] : memref<8xf32>
    memref.store %f5, %lut[%i5] : memref<8xf32>
    memref.store %f6, %lut[%i6] : memref<8xf32>
    memref.store %f7, %lut[%i7] : memref<8xf32>
    %b0 = arith.constant 0.0 : bf16
    %b1 = arith.constant 0.5 : bf16
    %b2 = arith.constant 1.0 : bf16
    %b3 = arith.constant 1.5 : bf16
    %b4 = arith.constant 2.0 : bf16
    %b5 = arith.constant 3.0 : bf16
    %b6 = arith.constant 4.0 : bf16
    %b7 = arith.constant 6.0 : bf16
    memref.store %b0, %lutb[%c0] : memref<8xbf16>
    memref.store %b1, %lutb[%i1] : memref<8xbf16>
    memref.store %b2, %lutb[%i2] : memref<8xbf16>
    memref.store %b3, %lutb[%i3] : memref<8xbf16>
    memref.store %b4, %lutb[%i4] : memref<8xbf16>
    memref.store %b5, %lutb[%i5] : memref<8xbf16>
    memref.store %b6, %lutb[%i6] : memref<8xbf16>
    memref.store %b7, %lutb[%i7] : memref<8xbf16>

    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c2048 = arith.constant 2048 : index

    // f32 shadows of A and B, filled from the same loop that writes the device
    // operands, so the reference cannot drift from what the kernel is given.
    %A_f32 = memref.alloc() : memref<256x4096xf32>
    %B_f32 = memref.alloc() : memref<4096x256xf32>

    %A = memref.alloc() : memref<256x4096xbf16>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %m = %c0 to %c2048 step %c1 {
        %k0 = arith.muli %m, %c2 : index
        %k1 = arith.addi %k0, %c1 : index
        %s0 = arith.addi %i, %k0 : index
        %s1 = arith.addi %i, %k1 : index
        %idx0 = arith.remui %s0, %c8 : index
        %idx1 = arith.remui %s1, %c8 : index
        %vb0 = memref.load %lutb[%idx0] : memref<8xbf16>
        %vb1 = memref.load %lutb[%idx1] : memref<8xbf16>
        %v0 = memref.load %lut[%idx0] : memref<8xf32>
        %v1 = memref.load %lut[%idx1] : memref<8xf32>
        memref.store %vb0, %A[%i, %k0] : memref<256x4096xbf16>
        memref.store %vb1, %A[%i, %k1] : memref<256x4096xbf16>
        memref.store %v0, %A_f32[%i, %k0] : memref<256x4096xf32>
        memref.store %v1, %A_f32[%i, %k1] : memref<256x4096xf32>
      }
    }

    %B = memref.alloc() : memref<4096x256xbf16>
    scf.for %m = %c0 to %c2048 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        %k0 = arith.muli %m, %c2 : index
        %k1 = arith.addi %k0, %c1 : index
        %s0 = arith.addi %j, %k0 : index
        %s1 = arith.addi %j, %k1 : index
        %idx0 = arith.remui %s0, %c8 : index
        %idx1 = arith.remui %s1, %c8 : index
        %vb0 = memref.load %lutb[%idx0] : memref<8xbf16>
        %vb1 = memref.load %lutb[%idx1] : memref<8xbf16>
        %v0 = memref.load %lut[%idx0] : memref<8xf32>
        %v1 = memref.load %lut[%idx1] : memref<8xf32>
        memref.store %vb0, %B[%k0, %j] : memref<4096x256xbf16>
        memref.store %vb1, %B[%k1, %j] : memref<4096x256xbf16>
        memref.store %v0, %B_f32[%k0, %j] : memref<4096x256xf32>
        memref.store %v1, %B_f32[%k1, %j] : memref<4096x256xf32>
      }
    }

    %C = memref.alloc() : memref<256x256xf32>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<256x256xf32>
      }
    }

    // Reference GEMM on the host, over the f32 shadows of the same operands.
    // Every product is a multiple of 0.25 and the largest result is well under
    // 2^24, so the f32 accumulation is exact and independent of summation
    // order: the device result has to match bit for bit.
    %C_ref = memref.alloc() : memref<256x256xf32>
    call @gemm_ref(%A_f32, %B_f32, %C_ref) : (memref<256x4096xf32>, memref<4096x256xf32>, memref<256x256xf32>) -> ()

    %C_res = call @test(%A, %B, %C) : (memref<256x4096xbf16>, memref<4096x256xbf16>, memref<256x256xf32>) -> memref<256x256xf32>
    %C_cast = memref.cast %C_res : memref<256x256xf32> to memref<*xf32>
    %C_ref_cast = memref.cast %C_ref : memref<256x256xf32> to memref<*xf32>
    %diff = call @verifyMemRefF32(%C_cast, %C_ref_cast) : (memref<*xf32>, memref<*xf32>) -> i64
    call @printI64(%diff) : (i64) -> ()

    // CHECK: 0
    memref.dealloc %A : memref<256x4096xbf16>
    memref.dealloc %B : memref<4096x256xbf16>
    memref.dealloc %C : memref<256x256xf32>
    memref.dealloc %C_ref : memref<256x256xf32>
    memref.dealloc %C_res : memref<256x256xf32>
    return
  }
  func.func private @verifyMemRefF32(%acutal : memref<*xf32>, %expected : memref<*xf32>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @printI64(%num : i64)


  // Plain host GEMM, used to build the expected result.
  func.func @gemm_ref(%A: memref<256x4096xf32>, %B: memref<4096x256xf32>,
                      %C: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c4K = arith.constant 4096 : index
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        %acc = scf.for %k = %c0 to %c4K step %c1
            iter_args(%sum = %zero) -> (f32) {
          %a = memref.load %A[%i, %k] : memref<256x4096xf32>
          %b = memref.load %B[%k, %j] : memref<4096x256xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %sum, %p : f32
          scf.yield %s : f32
        }
        memref.store %acc, %C[%i, %j] : memref<256x256xf32>
      }
    }
    return
  }

}
