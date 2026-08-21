// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// Note: layouts used by dpas_mx need to match HW constaint. Otherwise dpas_mx is not unrolled.
#a = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 1024], inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 4]>
#b_packed = #xegpu.layout<sg_layout = [2, 2], sg_data = [512, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>
#b = #xegpu.layout<sg_layout = [2, 2], sg_data = [1024, 16], inst_data = [64, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#c = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: inst_data is chosen to utilize 2D block load
#a_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [16, 32], lane_layout = [16, 1], lane_data = [1, 1]>
#b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: scales for dpas_mx needs separate layouts with inst_data to match HW constraint. Otherwise dpas_mx is not unrolled
#dpas_a_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 2], lane_layout = [8, 1], lane_data = [1, 1]>
#dpas_b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [2, 16], lane_layout = [1, 16], lane_data = [1, 1]>


module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @gemm_mxfp(%arg0: memref<256x4096xf4E2M1FN>, %arg1: memref<2048x256xi8>, %arg2: memref<256x128xf8E8M0FNU>, %arg3: memref<128x256xf8E8M0FNU>, %arg4: memref<256x256xf32>) kernel {
      %c0 = arith.constant 0 : index
      %mstep = arith.constant 32 : index
      %nstep = arith.constant 32 : index
      %kstep = arith.constant 1024 : index
      %mbound = arith.constant 256 : index
      %nbound = arith.constant 256 : index
      %kbound = arith.constant 4096 : index
      %kbstep = arith.constant 512 : index
      %kscalestep = arith.constant 32 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %mstep : index
      %n = arith.muli %block_id_y, %nstep : index

      %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<256x4096xf4E2M1FN> -> !xegpu.tensor_desc<32x1024xf4E2M1FN>
      %bp_tdesc = xegpu.create_nd_tdesc %arg1 : memref<2048x256xi8> -> !xegpu.tensor_desc<512x32xi8>
      %a_scale_tdesc = xegpu.create_nd_tdesc %arg2 : memref<256x128xf8E8M0FNU> -> !xegpu.tensor_desc<32x32xf8E8M0FNU>
      %b_scale_tdesc = xegpu.create_nd_tdesc %arg3 : memref<128x256xf8E8M0FNU> -> !xegpu.tensor_desc<32x32xf8E8M0FNU>

      // Load initial C
      %cd_tdesc = xegpu.create_nd_tdesc %arg4 : memref<256x256xf32> -> !xegpu.tensor_desc<32x32xf32, #c>
      %c_init = xegpu.load_nd %cd_tdesc[%m, %n] {layout = #c}: !xegpu.tensor_desc<32x32xf32, #c> -> vector<32x32xf32>

      %res:3 = scf.for %k = %c0 to %kbound step %kstep
        iter_args(%c_partial = %c_init, %kb = %c0, %kscale = %c0) -> (vector<32x32xf32>, index, index) {
        // load_nd with offset
        %a = xegpu.load_nd %a_tdesc[%m, %k] {layout = #a}: !xegpu.tensor_desc<32x1024xf4E2M1FN> -> vector<32x1024xf4E2M1FN>
        %bp = xegpu.load_nd %bp_tdesc[%kb, %n] {layout = #b_packed}: !xegpu.tensor_desc<512x32xi8> -> vector<512x32xi8>

        // Bitcast to fp4: 512x32 uint8 -> 512x64 fp4 (each uint8 holds 2 fp4 values)
        %b_bitcast = vector.bitcast %bp : vector<512x32xi8> to vector<512x64xf4E2M1FN>

        // De-interleave: extract even and odd columns
        // Even columns (indices 0, 2, 4, ..., 62) -> first half
        // Odd columns (indices 1, 3, 5, ..., 63) -> second half
        %b_even, %b_odd = vector.deinterleave %b_bitcast : vector<512x64xf4E2M1FN> -> vector<512x32xf4E2M1FN>

        // Reconstruct 1024x32 by interleaving even/odd rows:
        // Transpose to move the row dim to trailing position, interleave, transpose back.
        %b_even_t = vector.transpose %b_even, [1, 0] : vector<512x32xf4E2M1FN> to vector<32x512xf4E2M1FN>
        %b_odd_t = vector.transpose %b_odd, [1, 0] : vector<512x32xf4E2M1FN> to vector<32x512xf4E2M1FN>
        %b_interleaved = vector.interleave %b_even_t, %b_odd_t : vector<32x512xf4E2M1FN> -> vector<32x1024xf4E2M1FN>
        %b = vector.transpose %b_interleaved, [1, 0] : vector<32x1024xf4E2M1FN> to vector<1024x32xf4E2M1FN>

        %scale_a = xegpu.load_nd %a_scale_tdesc[%m, %kscale] {layout = #a_scale}: !xegpu.tensor_desc<32x32xf8E8M0FNU> -> vector<32x32xf8E8M0FNU>

        %scale_b = xegpu.load_nd %b_scale_tdesc[%kscale, %n] {layout = #b_scale}: !xegpu.tensor_desc<32x32xf8E8M0FNU> -> vector<32x32xf8E8M0FNU>
        %new_c_partial = xegpu.dpas_mx %a, %b, %c_partial scale_a = %scale_a scale_b = %scale_b
              {layout_a = #a,
               layout_b = #b,
               layout_cd = #c,
               layout_a_scale = #dpas_a_scale,
               layout_b_scale = #dpas_b_scale}
            : (vector<32x1024xf4E2M1FN>, vector<1024x32xf4E2M1FN>,
               vector<32x32xf32>,
               vector<32x32xf8E8M0FNU>, vector<32x32xf8E8M0FNU>)
            -> vector<32x32xf32>

        // b, a_scale and b_scale take different steps compared to a
        // compute adjusted k index for those tiles.
        %new_kb = arith.addi %kb, %kbstep : index
        %new_kscale = arith.addi %kscale, %kscalestep : index
        scf.yield %new_c_partial, %new_kb, %new_kscale : vector<32x32xf32>, index, index
      }

      // store_nd with offset
      xegpu.store_nd %res#0, %cd_tdesc[%m, %n] {layout = #c} : vector<32x32xf32>, !xegpu.tensor_desc<32x32xf32, #c>
      gpu.return
    }
  }

  func.func @test(%a: memref<256x4096xf4E2M1FN>, %b: memref<2048x256xi8>, %a_scale: memref<256x128xf8E8M0FNU>, %b_scale: memref<128x256xf8E8M0FNU>, %c: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index

    %memref_a = gpu.alloc() : memref<256x4096xf4E2M1FN>
    gpu.memcpy %memref_a, %a : memref<256x4096xf4E2M1FN>, memref<256x4096xf4E2M1FN>

    %memref_b = gpu.alloc() : memref<2048x256xi8>
    gpu.memcpy %memref_b, %b : memref<2048x256xi8>, memref<2048x256xi8>

    %memref_c = gpu.alloc() : memref<256x256xf32>
    gpu.memcpy %memref_c, %c : memref<256x256xf32>, memref<256x256xf32>

    %memref_a_scale = gpu.alloc() : memref<256x128xf8E8M0FNU>
    gpu.memcpy %memref_a_scale, %a_scale : memref<256x128xf8E8M0FNU>, memref<256x128xf8E8M0FNU>

    %memref_b_scale = gpu.alloc() : memref<128x256xf8E8M0FNU>
    gpu.memcpy %memref_b_scale, %b_scale : memref<128x256xf8E8M0FNU>, memref<128x256xf8E8M0FNU>

    gpu.launch_func @kernel::@gemm_mxfp blocks in (%c8, %c8, %c1) threads in (%c64, %c1, %c1)
    args(%memref_a : memref<256x4096xf4E2M1FN>, %memref_b : memref<2048x256xi8>, %memref_a_scale : memref<256x128xf8E8M0FNU>, %memref_b_scale : memref<128x256xf8E8M0FNU>, %memref_c : memref<256x256xf32>)
    gpu.dealloc %memref_a : memref<256x4096xf4E2M1FN>
    gpu.dealloc %memref_b : memref<2048x256xi8>
    gpu.dealloc %memref_a_scale : memref<256x128xf8E8M0FNU>
    gpu.dealloc %memref_b_scale : memref<128x256xf8E8M0FNU>

    %res = memref.alloc() : memref<256x256xf32>
    gpu.memcpy %res, %memref_c : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc %memref_c : memref<256x256xf32>
    return %res : memref<256x256xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c2K = arith.constant 2048 : index
    %c4K = arith.constant 4096 : index
    %c512K = arith.constant 524288 : index
    %c1packed_e2m1 = arith.constant 0x22 : i8
    %c0f32 = arith.constant 0.0 : f32
    %c1f8E8M0FNU = arith.constant 1.0 : f8E8M0FNU

    // The 8 magnitudes e2m1 can represent, indexed by their e2m1 bit pattern:
    // code c encodes lut[c], so a packed byte holding two copies of code c is
    // c * 0x11. Every one of these is also exact in f8E5M2, f8E4M3FN, bf16 and
    // f32, so the fp4, fp8 and bf16 tests can share one input set and one
    // reference result.
    %lut = memref.alloc() : memref<8xf32>
    %l0 = arith.constant 0.0 : f32
    %l1 = arith.constant 0.5 : f32
    %l2 = arith.constant 1.0 : f32
    %l3 = arith.constant 1.5 : f32
    %l4 = arith.constant 2.0 : f32
    %l5 = arith.constant 3.0 : f32
    %l6 = arith.constant 4.0 : f32
    %l7 = arith.constant 6.0 : f32
    memref.store %l0, %lut[%c0] : memref<8xf32>
    %i1 = arith.constant 1 : index
    memref.store %l1, %lut[%i1] : memref<8xf32>
    %i2 = arith.constant 2 : index
    memref.store %l2, %lut[%i2] : memref<8xf32>
    %i3 = arith.constant 3 : index
    memref.store %l3, %lut[%i3] : memref<8xf32>
    %i4 = arith.constant 4 : index
    memref.store %l4, %lut[%i4] : memref<8xf32>
    %i5 = arith.constant 5 : index
    memref.store %l5, %lut[%i5] : memref<8xf32>
    %i6 = arith.constant 6 : index
    memref.store %l6, %lut[%i6] : memref<8xf32>
    %i7 = arith.constant 7 : index
    memref.store %l7, %lut[%i7] : memref<8xf32>

    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c2048 = arith.constant 2048 : index
    // 0x11: both nibbles of the byte get the same code, so the packing order
    // within a byte does not affect the result.
    %c0x11 = arith.constant 17 : i8

    // f32 shadows of A and B, filled from the same loop that writes the device
    // operands, so the reference cannot drift from what the kernel is given.
    %A_f32 = memref.alloc() : memref<256x4096xf32>
    %B_f32 = memref.alloc() : memref<4096x256xf32>

    // A is row major fp4, two values per byte along K, so byte m of row i holds
    // K elements 2m and 2m+1.
    %A_flatbytes = memref.alloc() : memref<524288xi8>
    %A = memref.view %A_flatbytes[%c0][] : memref<524288xi8> to memref<256x4096xf4E2M1FN>
    scf.for %i = %c0 to %c256 step %c1 {
      %row = arith.muli %i, %c2048 : index
      scf.for %m = %c0 to %c2048 step %c1 {
        %sum = arith.addi %i, %m : index
        %idx = arith.remui %sum, %c8 : index
        %code = arith.index_cast %idx : index to i8
        %byte = arith.muli %code, %c0x11 : i8
        %pos = arith.addi %row, %m : index
        memref.store %byte, %A_flatbytes[%pos] : memref<524288xi8>
        %v = memref.load %lut[%idx] : memref<8xf32>
        %k0 = arith.muli %m, %c2 : index
        %k1 = arith.addi %k0, %c1 : index
        memref.store %v, %A_f32[%i, %k0] : memref<256x4096xf32>
        memref.store %v, %A_f32[%i, %k1] : memref<256x4096xf32>
      }
    }

    // B is fp4 packed along K as well, so byte row m holds K elements 2m, 2m+1.
    %B = memref.alloc() : memref<2048x256xi8>
    scf.for %m = %c0 to %c2K step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        %sum = arith.addi %j, %m : index
        %idx = arith.remui %sum, %c8 : index
        %code = arith.index_cast %idx : index to i8
        %byte = arith.muli %code, %c0x11 : i8
        memref.store %byte, %B[%m, %j] : memref<2048x256xi8>
        %v = memref.load %lut[%idx] : memref<8xf32>
        %k0 = arith.muli %m, %c2 : index
        %k1 = arith.addi %k0, %c1 : index
        memref.store %v, %B_f32[%k0, %j] : memref<4096x256xf32>
        memref.store %v, %B_f32[%k1, %j] : memref<4096x256xf32>
      }
    }

    %C = memref.alloc() : memref<256x256xf32>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<256x256xf32>
      }
    }

    %A_scale = memref.alloc() : memref<256x128xf8E8M0FNU>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        memref.store %c1f8E8M0FNU, %A_scale[%i, %j] : memref<256x128xf8E8M0FNU>
      }
    }

    %B_scale = memref.alloc() : memref<128x256xf8E8M0FNU>
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c1f8E8M0FNU, %B_scale[%i, %j] : memref<128x256xf8E8M0FNU>
      }
    }


    // Reference GEMM on the host, over the f32 shadows of the same operands.
    // Every product is a multiple of 0.25 and the largest result is well under
    // 2^24, so the f32 accumulation is exact and independent of summation
    // order: the device result has to match bit for bit.
    %C_ref = memref.alloc() : memref<256x256xf32>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        %acc = scf.for %k = %c0 to %c4K step %c1
            iter_args(%sum = %c0f32) -> (f32) {
          %a = memref.load %A_f32[%i, %k] : memref<256x4096xf32>
          %b = memref.load %B_f32[%k, %j] : memref<4096x256xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %sum, %p : f32
          scf.yield %s : f32
        }
        memref.store %acc, %C_ref[%i, %j] : memref<256x256xf32>
      }
    }

    %C_res = call @test(%A, %B, %A_scale, %B_scale, %C) : (memref<256x4096xf4E2M1FN>, memref<2048x256xi8>, memref<256x128xf8E8M0FNU>, memref<128x256xf8E8M0FNU>, memref<256x256xf32>) -> memref<256x256xf32>
    %C_cast = memref.cast %C_res : memref<256x256xf32> to memref<*xf32>
    %C_ref_cast = memref.cast %C_ref : memref<256x256xf32> to memref<*xf32>
    %diff = call @verifyMemRefF32(%C_cast, %C_ref_cast) : (memref<*xf32>, memref<*xf32>) -> i64
    call @printI64(%diff) : (i64) -> ()
    //call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // CHECK: 0
    memref.dealloc %A_flatbytes : memref<524288xi8>
    memref.dealloc %A_f32 : memref<256x4096xf32>
    memref.dealloc %B_f32 : memref<4096x256xf32>
    memref.dealloc %lut : memref<8xf32>
    memref.dealloc %C_ref : memref<256x256xf32>
    memref.dealloc %B : memref<2048x256xi8>
    memref.dealloc %A_scale : memref<256x128xf8E8M0FNU>
    memref.dealloc %B_scale : memref<128x256xf8E8M0FNU>
    memref.dealloc %C : memref<256x256xf32>
    memref.dealloc %C_res : memref<256x256xf32>
    return
  }
  func.func private @verifyMemRefF32(%acutal : memref<*xf32>, %expected : memref<*xf32>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @printI64(%num : i64)
  //func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

}
