// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// XFAIL: *
// Note: layouts used by dpas_mx need to match HW constaint. Otherwise dpas_mx is not unrolled.
// Note: the 2x2 subgroup grid tiles only the output dims (M, N). K is the GEMM
// reduction dimension and must NOT be partitioned across subgroups: every
// subgroup reduces over the full K. Therefore each operand's sg_data K extent
// is set to the *full* K (broadcast across subgroups), not K / sg_layout.
#a = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 4096], inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 1]>
#b_packed = #xegpu.layout<sg_layout = [2, 2], sg_data = [2048, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>
#b = #xegpu.layout<sg_layout = [2, 2], sg_data = [4096, 16], inst_data = [64, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#c = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: inst_data is chosen to utilize 2D block load
#a_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 128], inst_data = [16, 32], lane_layout = [16, 1], lane_data = [1, 1]>
#b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [128, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: scales for dpas_mx needs separate layouts with inst_data to match HW constraint. Otherwise dpas_mx is not unrolled
#dpas_a_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 128], inst_data = [8, 2], lane_layout = [8, 1], lane_data = [1, 1]>
#dpas_b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [128, 16], inst_data = [2, 16], lane_layout = [1, 16], lane_data = [1, 1]>


module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @gemm_mxfp(%arg0: memref<32x4096xf4E2M1FN>, %arg1: memref<2048x32xi8>, %arg2: memref<32x128xf8E8M0FNU>, %arg3: memref<128x32xf8E8M0FNU>, %arg4: memref<32x32xf32>) kernel {
      %c0 = arith.constant 0 : index

      %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<32x4096xf4E2M1FN> -> !xegpu.tensor_desc<32x4096xf4E2M1FN>
      %bp_tdesc = xegpu.create_nd_tdesc %arg1 : memref<2048x32xi8> -> !xegpu.tensor_desc<2048x32xi8>
      %a_scale_tdesc = xegpu.create_nd_tdesc %arg2 : memref<32x128xf8E8M0FNU> -> !xegpu.tensor_desc<32x128xf8E8M0FNU>
      %b_scale_tdesc = xegpu.create_nd_tdesc %arg3 : memref<128x32xf8E8M0FNU> -> !xegpu.tensor_desc<128x32xf8E8M0FNU>

      // Load initial C
      %cd_tdesc = xegpu.create_nd_tdesc %arg4 : memref<32x32xf32> -> !xegpu.tensor_desc<32x32xf32, #c>
      %c_init = xegpu.load_nd %cd_tdesc[%c0, %c0] {layout = #c}: !xegpu.tensor_desc<32x32xf32, #c> -> vector<32x32xf32>

        // load_nd with offset
        %a = xegpu.load_nd %a_tdesc[%c0, %c0] {layout = #a}: !xegpu.tensor_desc<32x4096xf4E2M1FN> -> vector<32x4096xf4E2M1FN>
        %bp = xegpu.load_nd %bp_tdesc[%c0, %c0] {layout = #b_packed}: !xegpu.tensor_desc<2048x32xi8> -> vector<2048x32xi8>

        // Bitcast to fp4: 2048x32 uint8 -> 2048x64 fp4 (each uint8 holds 2 fp4 values)
        %b_bitcast = vector.bitcast %bp : vector<2048x32xi8> to vector<2048x64xf4E2M1FN>

        // De-interleave: extract even and odd columns
        // Even columns (indices 0, 2, 4, ..., 62) -> first half
        // Odd columns (indices 1, 3, 5, ..., 63) -> second half
        %b_even, %b_odd = vector.deinterleave %b_bitcast : vector<2048x64xf4E2M1FN> -> vector<2048x32xf4E2M1FN>

        // Reconstruct 4096x32 by interleaving even/odd rows:
        // Transpose to move the row dim to trailing position, interleave, transpose back.
        %b_even_t = vector.transpose %b_even, [1, 0] : vector<2048x32xf4E2M1FN> to vector<32x2048xf4E2M1FN>
        %b_odd_t = vector.transpose %b_odd, [1, 0] : vector<2048x32xf4E2M1FN> to vector<32x2048xf4E2M1FN>
        %b_interleaved = vector.interleave %b_even_t, %b_odd_t : vector<32x2048xf4E2M1FN> -> vector<32x4096xf4E2M1FN>
        %b = vector.transpose %b_interleaved, [1, 0] : vector<32x4096xf4E2M1FN> to vector<4096x32xf4E2M1FN>

        %scale_a = xegpu.load_nd %a_scale_tdesc[%c0, %c0] {layout = #a_scale}: !xegpu.tensor_desc<32x128xf8E8M0FNU> -> vector<32x128xf8E8M0FNU>

        %scale_b = xegpu.load_nd %b_scale_tdesc[%c0, %c0] {layout = #b_scale}: !xegpu.tensor_desc<128x32xf8E8M0FNU> -> vector<128x32xf8E8M0FNU>
        %res = xegpu.dpas_mx %a, %b, %c_init scale_a = %scale_a scale_b = %scale_b
              {layout_a = #a,
               layout_b = #b,
               layout_cd = #c,
               layout_a_scale = #dpas_a_scale,
               layout_b_scale = #dpas_b_scale}
            : (vector<32x4096xf4E2M1FN>, vector<4096x32xf4E2M1FN>,
               vector<32x32xf32>,
               vector<32x128xf8E8M0FNU>, vector<128x32xf8E8M0FNU>)
            -> vector<32x32xf32>

      // store_nd with offset
      xegpu.store_nd %res, %cd_tdesc[%c0, %c0] {layout = #c} : vector<32x32xf32>, !xegpu.tensor_desc<32x32xf32, #c>
      gpu.return
    }
  }

  func.func @test(%a: memref<32x4096xf4E2M1FN>, %b: memref<2048x32xi8>, %a_scale: memref<32x128xf8E8M0FNU>, %b_scale: memref<128x32xf8E8M0FNU>, %c: memref<32x32xf32>) -> memref<32x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index

    %memref_a = gpu.alloc() : memref<32x4096xf4E2M1FN>
    gpu.memcpy %memref_a, %a : memref<32x4096xf4E2M1FN>, memref<32x4096xf4E2M1FN>

    %memref_b = gpu.alloc() : memref<2048x32xi8>
    gpu.memcpy %memref_b, %b : memref<2048x32xi8>, memref<2048x32xi8>

    %memref_c = gpu.alloc() : memref<32x32xf32>
    gpu.memcpy %memref_c, %c : memref<32x32xf32>, memref<32x32xf32>

    %memref_a_scale = gpu.alloc() : memref<32x128xf8E8M0FNU>
    gpu.memcpy %memref_a_scale, %a_scale : memref<32x128xf8E8M0FNU>, memref<32x128xf8E8M0FNU>

    %memref_b_scale = gpu.alloc() : memref<128x32xf8E8M0FNU>
    gpu.memcpy %memref_b_scale, %b_scale : memref<128x32xf8E8M0FNU>, memref<128x32xf8E8M0FNU>

    gpu.launch_func @kernel::@gemm_mxfp blocks in (%c1, %c1, %c1) threads in (%c64, %c1, %c1)
    args(%memref_a : memref<32x4096xf4E2M1FN>, %memref_b : memref<2048x32xi8>, %memref_a_scale : memref<32x128xf8E8M0FNU>, %memref_b_scale : memref<128x32xf8E8M0FNU>, %memref_c : memref<32x32xf32>)
    gpu.dealloc %memref_a : memref<32x4096xf4E2M1FN>
    gpu.dealloc %memref_b : memref<2048x32xi8>
    gpu.dealloc %memref_a_scale : memref<32x128xf8E8M0FNU>
    gpu.dealloc %memref_b_scale : memref<128x32xf8E8M0FNU>

    %res = memref.alloc() : memref<32x32xf32>
    gpu.memcpy %res, %memref_c : memref<32x32xf32>, memref<32x32xf32>
    gpu.dealloc %memref_c : memref<32x32xf32>
    return %res : memref<32x32xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c2K = arith.constant 2048 : index
    %c4K = arith.constant 4096 : index
    %c64K = arith.constant 65536 : index
    %c1packed_e2m1 = arith.constant 0x22 : i8
    %c0f32 = arith.constant 0.0 : f32
    %c1f8E8M0FNU = arith.constant 1.0 : f8E8M0FNU

    %A_flatbytes = memref.alloc() : memref<65536xi8>
    %A = memref.view %A_flatbytes[%c0][] : memref<65536xi8> to memref<32x4096xf4E2M1FN>
    scf.for %i = %c0 to %c64K step %c1 {
      memref.store %c1packed_e2m1, %A_flatbytes[%i] : memref<65536xi8>
    }

    %B = memref.alloc() : memref<2048x32xi8>
    scf.for %i = %c0 to %c2K step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        memref.store %c1packed_e2m1, %B[%i, %j] : memref<2048x32xi8>
      }
    }

    %C = memref.alloc() : memref<32x32xf32>
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<32x32xf32>
      }
    }

    %A_scale = memref.alloc() : memref<32x128xf8E8M0FNU>
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        memref.store %c1f8E8M0FNU, %A_scale[%i, %j] : memref<32x128xf8E8M0FNU>
      }
    }

    %B_scale = memref.alloc() : memref<128x32xf8E8M0FNU>
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        memref.store %c1f8E8M0FNU, %B_scale[%i, %j] : memref<128x32xf8E8M0FNU>
      }
    }


    %c4Kf = arith.constant 4096.0 : f32
    %C_ref = memref.alloc() : memref<32x32xf32>
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        memref.store %c4Kf, %C_ref[%i, %j] : memref<32x32xf32>
      }
    }

    %C_res = call @test(%A, %B, %A_scale, %B_scale, %C) : (memref<32x4096xf4E2M1FN>, memref<2048x32xi8>, memref<32x128xf8E8M0FNU>, memref<128x32xf8E8M0FNU>, memref<32x32xf32>) -> memref<32x32xf32>
    %C_cast = memref.cast %C_res : memref<32x32xf32> to memref<*xf32>
    %C_ref_cast = memref.cast %C_ref : memref<32x32xf32> to memref<*xf32>
    %diff = call @verifyMemRefF32(%C_cast, %C_ref_cast) : (memref<*xf32>, memref<*xf32>) -> i64
    call @printI64(%diff) : (i64) -> ()
    //call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // CHECK: 0
    memref.dealloc %A_flatbytes : memref<65536xi8>
    memref.dealloc %B : memref<2048x32xi8>
    memref.dealloc %A_scale : memref<32x128xf8E8M0FNU>
    memref.dealloc %B_scale : memref<128x32xf8E8M0FNU>
    memref.dealloc %C : memref<32x32xf32>
    memref.dealloc %C_res : memref<32x32xf32>
    return
  }
  func.func private @verifyMemRefF32(%acutal : memref<*xf32>, %expected : memref<*xf32>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @printI64(%num : i64)
  //func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

}
