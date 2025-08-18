// RUN: mlir-opt %s --split-input-file -convert-xegpu-to-xevm | FileCheck %s

gpu.module @test {
// CHECK-LABEL: @load_gather_ui64_src_constant_offset
// CHECK-SAME: %[[ARG0:.*]]: ui64
gpu.func @load_gather_ui64_src_constant_offset(%src: ui64) {
  %0 = arith.constant dense<0> : vector<1xindex>
  %1 = arith.constant dense<1>: vector<1xi1>
  // CHECK: %[[VAR0:.*]] = index.castu %[[ARG0]] : ui64 to index
  // CHECK: %[[VAR1:.*]] = gpu.lane_id
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[VAR2:.*]] = arith.muli %[[C4]], %[[C0]] : index
  // CHECK: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: %[[VAR3:.*]] = arith.muli %[[C4]], %[[C8]] : index
  // CHECK: %[[VAR4:.*]] = arith.muli %[[VAR1]], %[[VAR3]] : index
  // CHECK: %[[VAR5:.*]] = arith.addi %[[VAR4]], %[[VAR2]] : index
  // CHECK: %[[VAR6:.*]] = arith.addi %[[VAR0]], %[[VAR5]] : index
  %2 = xegpu.create_tdesc %src, %0 : ui64, vector<1xindex> -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // CHECK: %[[VAR7:.*]] = arith.index_cast %[[VAR6]] : index to i64
  // CHECK: %[[VAR8:.*]] = llvm.inttoptr %[[VAR7]] : i64 to !llvm.ptr<1>
  // CHECK: %[[VAR9:.*]] = llvm.load %[[VAR8]] : !llvm.ptr<1> -> vector<2xf32>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<1xi1> -> vector<2xf32>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @load_gather_memref_src_constant_offset
// CHECK-SAME: %[[ARG0:.*]]: memref<256xf32>
gpu.func @load_gather_memref_src_constant_offset(%src: memref<256xf32>) {
  %0 = arith.constant dense<0> : vector<1xindex>
  %1 = arith.constant dense<1>: vector<1xi1>
  // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<256xf32> -> index
  // CHECK: %[[VAR0:.*]] = gpu.lane_id
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[VAR1:.*]] = arith.muli %[[C4]], %[[C0]] : index
  // CHECK: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: %[[VAR2:.*]] = arith.muli %[[C4]], %[[C8]] : index
  // CHECK: %[[VAR3:.*]] = arith.muli %[[VAR0]], %[[VAR2]] : index
  // CHECK: %[[VAR4:.*]] = arith.addi %[[VAR3]], %[[VAR1]] : index
  // CHECK: %[[VAR5:.*]] = arith.addi %[[INTPTR]], %[[VAR4]] : index
  %2 = xegpu.create_tdesc %src, %0 : memref<256xf32>, vector<1xindex> -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK: %[[VAR6:.*]] = arith.index_cast %[[VAR5]] : index to i64
  // CHECK: %[[VAR7:.*]] = llvm.inttoptr %[[VAR6]] : i64 to !llvm.ptr<1>
  // CHECK: %[[VAR8:.*]] = llvm.load %[[VAR7]] : !llvm.ptr<1> -> vector<1xf32>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>, vector<1xi1> -> vector<1xf32>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @load_gather_memref_src_value_offset
gpu.func @load_gather_memref_src_value_offset(%src: memref<256xf16>, %offset: vector<1xindex>) {
  %1 = arith.constant dense<1>: vector<1xi1>
  %2 = xegpu.create_tdesc %src, %offset : memref<256xf16>, vector<1xindex> -> !xegpu.tensor_desc<1x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  %3 = xegpu.load %2, %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<1x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<1xi1> -> vector<8xf16>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @load_gather_memref_src_load_offset
gpu.func @load_gather_memref_src_load_offset(%src: memref<256xf16>, %offset1: vector<1xindex>, %offset2: vector<1xindex>) {
  %1 = arith.constant dense<1>: vector<1xi1>
  %2 = xegpu.create_tdesc %src, %offset1 : memref<256xf16>, vector<1xindex> -> !xegpu.tensor_desc<1x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  %3 = xegpu.load %2[%offset2], %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<1x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<1xindex>, vector<1xi1> -> vector<8xf16>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @store_scatter_ui64_src_constant_offset
gpu.func @store_scatter_ui64_src_constant_indices(%src: ui64) {
  %0 = arith.constant dense<0> : vector<1xindex>
  %1 = arith.constant dense<1>: vector<1xi1>
  %2 = arith.constant dense<2.9>: vector<2xf32>
  %3 = xegpu.create_tdesc %src, %0 : ui64, vector<1xindex> -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf32>, !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<1xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @store_scatter_memref_src_constant_offset
gpu.func @store_scatter_memref_src_constant_offset(%src: memref<256xf32>) {
  %0 = arith.constant dense<0> : vector<1xindex>
  %1 = arith.constant dense<1>: vector<1xi1>
  %2 = arith.constant dense<2.9>: vector<2xf16>
  %3 = xegpu.create_tdesc %src, %0 : memref<256xf32>, vector<1xindex> -> !xegpu.tensor_desc<1x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<2xf16>, !xegpu.tensor_desc<1x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<1xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @store_scatter_memref_src_value_offset
gpu.func @store_scatter_memref_src_value_offset(%src: memref<256xf32>, %offset: vector<1xindex>) {
  %1 = arith.constant dense<1>: vector<1xi1>
  %2 = arith.constant dense<2.9>: vector<1xf32>
  %3 = xegpu.create_tdesc %src, %offset : memref<256xf32>, vector<1xindex> -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
  xegpu.store %2, %3, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<1xf32>, !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>, vector<1xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @store_scatter_memref_src_store_offset
gpu.func @store_scatter_memref_src_store_offset(%src: memref<256xf32>, %offset: vector<1xindex>, %offset2: vector<1xindex>) {
  %1 = arith.constant dense<1>: vector<1xi1>
  %2 = arith.constant dense<2.9>: vector<1xf32>
  %3 = xegpu.create_tdesc %src, %offset : memref<256xf32>, vector<1xindex> -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
  xegpu.store %2, %3[%offset2], %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<1xf32>, !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>, vector<1xindex>, vector<1xi1>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @prefetch_ui64_src_constant_offset
gpu.func @prefetch_ui64_src_constant_offset(%src: ui64) {
  %0 = arith.constant dense<0> : vector<1xindex>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<1xindex> -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @prefetch_memref_src_constant_offset
gpu.func @prefetch_memref_src_constant_offset(%src: memref<256xf32>) {
  %0 = arith.constant dense<0> : vector<1xindex>
  %1 = xegpu.create_tdesc %src, %0 : memref<256xf32>, vector<1xindex> -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @prefetch_memref_src_value_offset
gpu.func @prefetch_memref_src_value_offset(%src: memref<256xf32>, %offset: vector<1xindex>) {
  %1 = xegpu.create_tdesc %src, %offset : memref<256xf32>, vector<1xindex> -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  gpu.return
}
}
// -----

gpu.module @test {
// CHECK-LABEL: @prefetch_memref_src_prefetch_offset
gpu.func @prefetch_memref_src_value_offset(%src: memref<256xf32>, %offset: vector<1xindex>, %offset2: vector<1xindex>) {
  %1 = xegpu.create_tdesc %src, %offset : memref<256xf32>, vector<1xindex> -> !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  xegpu.prefetch %1[%offset2] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<1x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<1xindex>
  gpu.return
}
}
