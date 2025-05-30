// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: func.func @block_dpas(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) {
  // CHECK: %c32_i32 = arith.constant 32 : i32
  // CHECK: %c8_i32 = arith.constant 8 : i32
  // CHECK: %c32_i32_0 = arith.constant 32 : i32
  // CHECK: %c0_i32 = arith.constant 0 : i32
  // CHECK: %c0_i32_1 = arith.constant 0 : i32
  // CHECK: %0 = xevm.blockload2d %arg0, %c32_i32, %c8_i32, %c32_i32_0, %c0_i32, %c0_i32_1 {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, l1_cache_control =  Default, l3_cache_control =  Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  // CHECK:  %c32_i32_2 = arith.constant 32 : i32
  // CHECK:  %c16_i32 = arith.constant 16 : i32
  // CHECK:  %c32_i32_3 = arith.constant 32 : i32
  // CHECK:  %1 = xevm.blockload2d %arg1, %c32_i32_2, %c16_i32, %c32_i32_3, %c0_i32, %c0_i32_1 {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 1, transpose = false, vnni_transform = false, l1_cache_control =  Default, l3_cache_control =  Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  // CHECK:  %2 = vector.bitcast %1 : vector<16xi16> to vector<8xi32>
  // CHECK:  %c64_i32 = arith.constant 64 : i32
  // CHECK:  %c8_i32_4 = arith.constant 8 : i32
  // CHECK:  %c64_i32_5 = arith.constant 64 : i32
  // CHECK:  %3 = xevm.blockload2d %arg2, %c64_i32, %c8_i32_4, %c64_i32_5, %c0_i32, %c0_i32_1 {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, l1_cache_control =  Default, l3_cache_control =  Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  // CHECK:  %4 = vector.bitcast %3 : vector<8xi32> to vector<8xf32>
  // CHECK:  %5 = xevm.dpas %4, %0, %2 {pa = f16, pb = f16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  // CHECK:  %6 = vector.bitcast %5 : vector<8xf32> to vector<8xi32>
  // CHECK:  xevm.blockstore2d %arg2, %c64_i32, %c8_i32_4, %c64_i32_5, %c0_i32, %c0_i32_1, %6 {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, l1_cache_control =  Default, l3_cache_control =  Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
func.func @block_dpas(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>, %c: !llvm.ptr<1>)  {
  %base_width_a = arith.constant 32 : i32
  %base_height_a = arith.constant 8 : i32
  %base_pitch_a = arith.constant 32 : i32
  %x = arith.constant 0 : i32
  %y = arith.constant 0 : i32
  %loaded_a = xevm.blockload2d %a, %base_width_a, %base_height_a, %base_pitch_a, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>

  %base_width_b = arith.constant 32 : i32
  %base_height_b = arith.constant 16 : i32
  %base_pitch_b = arith.constant 32 : i32
  %loaded_b1 = xevm.blockload2d %b, %base_width_b, %base_height_b, %base_pitch_b, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  %loaded_b_casted = vector.bitcast %loaded_b1 : vector<16xi16> to vector<8xi32>

  %base_width_c = arith.constant 64 : i32
  %base_height_c = arith.constant 8 : i32
  %base_pitch_c = arith.constant 64 : i32
  %loaded_c = xevm.blockload2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>

  %loaded_c_casted = vector.bitcast %loaded_c : vector<8xi32> to vector<8xf32>
  %c_result = xevm.dpas %loaded_c_casted, %loaded_a, %loaded_b_casted {pa = f16, pb = f16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  %c_result_casted = vector.bitcast %c_result : vector<8xf32> to vector<8xi32>

  xevm.blockstore2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %c_result_casted {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
  return
}
