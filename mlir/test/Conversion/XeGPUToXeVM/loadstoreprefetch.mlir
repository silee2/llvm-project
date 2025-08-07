// RUN: mlir-opt %s --split-input-file -convert-xegpu-to-xevm | FileCheck %s

gpu.module @test {
// CHECK-LABEL: @simt_load
// CHECK-SAME: %[[ARG0:.*]]: ui64
gpu.func @simt_load(%src: ui64) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<4xi1>
  // CHECK: %[[VAR0:.*]] = index.castu %[[ARG0]] : ui64 to index
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[VAR1:.*]] = arith.muli %[[C4]], %[[C0]] : index
  // CHECK: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: %[[VAR2:.*]] = arith.muli %[[C4]], %[[C8]] : index
  // CHECK: %[[VAR3:.*]] = gpu.lane_id
  // CHECK: %[[VAR4:.*]] = arith.muli %[[VAR3]], %[[VAR2]] : index
  // CHECK: %[[VAR5:.*]] = arith.addi %[[VAR4]], %[[VAR1]] : index
  // CHECK: %[[VAR6:.*]] = arith.addi %[[VAR0]], %[[VAR5]] : index
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // CHECK: %[[VAR7:.*]] = arith.index_cast %[[VAR6]] : index to i64
  // CHECK: %[[VAR8:.*]] = llvm.inttoptr %[[VAR7]] : i64 to !llvm.ptr<1>
  // CHECK: %[[VAR9:.*]] = llvm.load %[[VAR8]] : !llvm.ptr<1> -> vector<2xf32>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1> -> vector<2xf32>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @simt_load_2
gpu.func @simt_load_2(%src: ui64) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>, vector<4xi1> -> vector<1xf32>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @simt_load_3
gpu.func @simt_load_3(%src: ui64) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<4xi1> -> vector<8xf16>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @simt_store
gpu.func @simt_store(%src: ui64) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<4xi1>
  %2 = arith.constant dense<2.9>: vector<2xf32>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf32>, !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @simt_store_2
gpu.func @simt_store_2(%src: ui64) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<4xi1>
  %2 = arith.constant dense<2.9>: vector<2xf16>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf16>, !xegpu.tensor_desc<4x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @simt_store_3
gpu.func @simt_store_3(%src: ui64) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<4xi1>
  %2 = arith.constant dense<2.9>: vector<1xf32>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<1xf32>, !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>>, vector<4xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @prefetch
gpu.func @prefetch(%src: ui64) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}
}
