// C-F4 TARGET layout: layout<[8,1],[1,1]> on vector<8x2>
gpu.module @xevm_module {
gpu.func @probe_c_f4_target(%arg0: memref<8x2xbf16>) {
  %c0 = arith.constant 0 : index
  %td = xegpu.create_nd_tdesc %arg0 : memref<8x2xbf16> -> !xegpu.tensor_desc<8x2xbf16, #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>>
  %v = xegpu.load_nd %td[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>}> : !xegpu.tensor_desc<8x2xbf16, #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>> -> vector<8x2xbf16>
  "test.some_sink"(%v) : (vector<8x2xbf16>) -> ()
  gpu.return
}
}
