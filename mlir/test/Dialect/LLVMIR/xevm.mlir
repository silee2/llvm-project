// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: func.func @blockload2d(%[[ARG0:.*]]: !llvm.ptr<1>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32)
func.func @blockload2d(%a: !llvm.ptr<1>, %base_width_a: i32, %base_height_a: i32, %base_pitch_a: i32, %x: i32, %y: i32) -> vector<8xi16> {
  // CHECK: %[[VAR0:.*]] = xevm.blockload2d %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, l1_cache_control = Default, l3_cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  %loaded_a = xevm.blockload2d %a, %base_width_a, %base_height_a, %base_pitch_a, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  return %loaded_a : vector<8xi16>
}

// -----
// CHECK: func.func @blockstore2d(%[[ARG0:.*]]: !llvm.ptr<1>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: vector<8xi32>)
func.func @blockstore2d(%c: !llvm.ptr<1>, %base_width_c: i32, %base_height_c: i32, %base_pitch_c: i32, %x: i32, %y: i32, %c_result_casted: vector<8xi32>) {
  // CHECK: xevm.blockstore2d %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]] {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, l1_cache_control = Default, l3_cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
  xevm.blockstore2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %c_result_casted {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
  return
}

// -----
// CHECK: func.func @dpas(%[[ARG0:.*]]: vector<8xf32>, %[[ARG1:.*]]: vector<8xi16>, %[[ARG2:.*]]: vector<8xi32>)
func.func @dpas(%loaded_c_casted: vector<8xf32>, %loaded_a: vector<8xi16>, %loaded_b_casted: vector<8xi32>) -> vector<8xf32> {
  // CHECK: %0 = xevm.dpas %[[ARG0]], %[[ARG1]], %[[ARG2]] {pa = f16, pb = f16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  %c_result = xevm.dpas %loaded_c_casted, %loaded_a, %loaded_b_casted {pa = f16, pb = f16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  return %c_result : vector<8xf32>
}
