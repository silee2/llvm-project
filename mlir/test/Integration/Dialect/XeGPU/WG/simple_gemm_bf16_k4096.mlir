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

    %A = memref.alloc() : memref<256x4096xbf16>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c4K step %c1 {
        memref.store %c1bf16, %A[%i, %j] : memref<256x4096xbf16>
      }
    }

    %B = memref.alloc() : memref<4096x256xbf16>
    scf.for %i = %c0 to %c4K step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c1bf16, %B[%i, %j] : memref<4096x256xbf16>
      }
    }

    %C = memref.alloc() : memref<256x256xf32>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<256x256xf32>
      }
    }

    // A and B are all ones, so every element of C accumulates K = 4096 ones.
    %c4Kf = arith.constant 4096.0 : f32
    %C_ref = memref.alloc() : memref<256x256xf32>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c4Kf, %C_ref[%i, %j] : memref<256x256xf32>
      }
    }

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

}
