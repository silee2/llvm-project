// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// XFAIL: *
module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @block_scaled_dpas_bf8(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>, %c: !llvm.ptr<1>) kernel {
      %base_width_a = arith.constant 32 : i32
      %base_height_a = arith.constant 8 : i32
      %base_pitch_a = arith.constant 32 : i32
      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32
      %loaded_a = xevm.blockload2d %a, %base_width_a, %base_height_a, %base_pitch_a, %x, %y
          <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32, v_blocks=1 : i32,
            transpose=false, pack_register=false}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %loaded_a_casted = vector.bitcast %loaded_a : vector<8xi32> to vector<16xf16>
      %a_trunc = xevm.truncf %loaded_a_casted { src_etype = f16, dst_etype = bf8 } : (vector<16xf16>) -> vector<16xi8>
      %a_trunc_casted = vector.bitcast %a_trunc : vector<16xi8> to vector<8xi16>

      %base_width_b = arith.constant 16 : i32
      %base_height_b = arith.constant 32 : i32
      %base_pitch_b = arith.constant 16 : i32
      %loaded_b = xevm.blockload2d %b, %base_width_b, %base_height_b, %base_pitch_b, %x, %y
          <{elem_size_in_bits=8 : i32, tile_width=16 : i32, tile_height=32 : i32, v_blocks=1 : i32,
            transpose=false, pack_register=true}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>

      // Note: scale is not computed. Constant values are used for simplifying the example
      %scale_a = arith.constant dense<1.0> : vector<2xf8E8M0FNU>
      %scale_b = arith.constant dense<1.0> : vector<2xf8E8M0FNU>
      %scale_a_casted = vector.bitcast %scale_a : vector<2xf8E8M0FNU> to vector<2xi8>
      %scale_b_casted = vector.bitcast %scale_b : vector<2xf8E8M0FNU> to vector<2xi8>
      // Note: c is not loaded. constant vector is used for simplifying the example
      %loaded_c_casted = arith.constant dense<0.0> : vector<8xf32>

      %c_result = xevm.mma_mx %a_trunc_casted, %loaded_b, %scale_a_casted, %scale_b_casted, %loaded_c_casted
          {shape=<m=8, n=16, k=32>, types=<d=f32, a=bf8, b=bf8, c=f32>}
          : (vector<8xi16>, vector<8xi32>, vector<2xi8>, vector<2xi8>, vector<8xf32>) -> vector<8xf32>
      %c_result_casted = vector.bitcast %c_result : vector<8xf32> to vector<8xi32>

      %base_width_c = arith.constant 16 : i32
      %base_height_c = arith.constant 8 : i32
      %base_pitch_c = arith.constant 16 : i32
      xevm.blockstore2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %c_result_casted
          <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32}>
          : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      gpu.return
    }
  }

  func.func @test(%a : memref<8x32xf16>, %b : memref<32x16xf8E5M2>, %c : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %memref_a = gpu.alloc() : memref<8x32xf16>
    gpu.memcpy %memref_a, %a : memref<8x32xf16>, memref<8x32xf16>
    %a_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_a : memref<8x32xf16> -> index
    %a_ptr_as_i64 = arith.index_cast %a_ptr_as_idx : index to i64
    %a_ptr = llvm.inttoptr %a_ptr_as_i64 : i64 to !llvm.ptr
    %a_ptr_casted = llvm.addrspacecast %a_ptr : !llvm.ptr to !llvm.ptr<1>

    %memref_b = gpu.alloc() : memref<32x16xf8E5M2>
    gpu.memcpy %memref_b, %b : memref<32x16xf8E5M2>, memref<32x16xf8E5M2>
    %b_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_b : memref<32x16xf8E5M2> -> index
    %b_ptr_as_i64 = arith.index_cast %b_ptr_as_idx : index to i64
    %b_ptr = llvm.inttoptr %b_ptr_as_i64 : i64 to !llvm.ptr
    %b_ptr_casted = llvm.addrspacecast %b_ptr : !llvm.ptr to !llvm.ptr<1>

    %memref_c = gpu.alloc() : memref<8x16xf32>
    gpu.memcpy %memref_c, %c : memref<8x16xf32>, memref<8x16xf32>
    %c_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_c : memref<8x16xf32> -> index
    %c_ptr_as_i64 = arith.index_cast %c_ptr_as_idx : index to i64
    %c_ptr = llvm.inttoptr %c_ptr_as_i64 : i64 to !llvm.ptr
    %c_ptr_casted = llvm.addrspacecast %c_ptr : !llvm.ptr to !llvm.ptr<1>

    gpu.launch_func @kernel::@block_scaled_dpas_bf8 blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%a_ptr_casted : !llvm.ptr<1>, %b_ptr_casted : !llvm.ptr<1>, %c_ptr_casted : !llvm.ptr<1>)
    gpu.dealloc %memref_a : memref<8x32xf16>
    gpu.dealloc %memref_b : memref<32x16xf8E5M2>
    %res = memref.alloc() : memref<8x16xf32>
    gpu.memcpy %res, %memref_c : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc %memref_c : memref<8x16xf32>
    return %res : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x32xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c0f32 = arith.constant 0.0 : f32
    %c1bf8 = arith.constant 1.0 : f8E5M2
    %c1mxscale = arith.constant 1.0 :f8E8M0FNU

    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %row_idx = arith.index_cast %i : index to i32
        %row = arith.sitofp %row_idx : i32 to f16
        memref.store %row, %A[%i, %j] : memref<8x32xf16>
      }
    }
    %B = memref.alloc() : memref<32x16xf8E5M2>
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        //%col_idx = arith.index_cast %j : index to i32
        //%col = arith.sitofp %col_idx : i32 to f16
        memref.store %c1bf8, %B[%i, %j] : memref<32x16xf8E5M2>
      }
    }
    %C = memref.alloc() : memref<8x16xf32>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<8x16xf32>
      }
    }

    %C_res = call @test(%A, %B, %C) : (memref<8x32xf16>, memref<32x16xf8E5M2>, memref<8x16xf32>) -> memref<8x16xf32>
    %C_cast = memref.cast %C_res : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    memref.dealloc %A : memref<8x32xf16>
    memref.dealloc %B : memref<32x16xf8E5M2>
    memref.dealloc %C : memref<8x16xf32>
    memref.dealloc %C_res : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

}
