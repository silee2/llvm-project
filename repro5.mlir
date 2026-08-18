// RUN: mlir-opt --test-xegpu-resolve-layout-conflicts -split-input-file %s

#lane_1x16 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
#lane_1x1x16 = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>

gpu.module @test {
func.func @slice_vs_layout(%arg0: memref<8x16xbf16>) {
  %c0 = arith.constant 0 : index
  %t = xegpu.create_nd_tdesc %arg0 : memref<8x16xbf16> -> !xegpu.tensor_desc<8x16xbf16, #lane_1x16>
  %v = xegpu.load_nd %t[%c0, %c0] {layout = #lane_1x16} : !xegpu.tensor_desc<8x16xbf16, #lane_1x16> -> vector<8x16xbf16>
  %b = vector.broadcast %v {layout_result_0 = #lane_1x1x16} : vector<8x16xbf16> to vector<1x8x16xbf16>
  %s = vector.shape_cast %b {layout_result_0 = #lane_1x16} : vector<1x8x16xbf16> to vector<8x16xbf16>
  xegpu.store_nd %s, %t[%c0, %c0] {layout = #lane_1x16} : vector<8x16xbf16>, !xegpu.tensor_desc<8x16xbf16, #lane_1x16>
  return
}
}

// -----

#no_order = #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>
#with_order = #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1], order = [1, 0]>

gpu.module @test2 {
func.func @default_order_vs_explicit_order(%arg0: memref<32x16xi8>) {
  %c0 = arith.constant 0 : index
  %t = xegpu.create_nd_tdesc %arg0 : memref<32x16xi8> -> !xegpu.tensor_desc<32x16xi8, #no_order>
  %v = xegpu.load_nd %t[%c0, %c0] {layout = #no_order} : !xegpu.tensor_desc<32x16xi8, #no_order> -> vector<32x16xi8>
  xegpu.store_nd %v, %t[%c0, %c0] {layout = #with_order} : vector<32x16xi8>, !xegpu.tensor_desc<32x16xi8, #no_order>
  return
}
}
