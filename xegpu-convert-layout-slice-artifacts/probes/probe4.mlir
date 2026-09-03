// (a) B/F4 SOURCE layout on vector<8x2xbf16>
gpu.module @xevm_module {
gpu.func @bf4_source() {
  %0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>} dense<1.0> : vector<8x2xbf16>
  %1 = xegpu.convert_layout %0 <{input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>, target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>}> : vector<8x2xbf16>
  "test.some_sink"(%1) : (vector<8x2xbf16>) -> ()
  gpu.return
}
}

// -----
// (b) B/F4 TARGET layout on vector<8x2xbf16>
gpu.module @xevm_module {
gpu.func @bf4_target() {
  %0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>} dense<1.0> : vector<8x2xbf16>
  %1 = xegpu.convert_layout %0 <{input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>, target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>}> : vector<8x2xbf16>
  "test.some_sink"(%1) : (vector<8x2xbf16>) -> ()
  gpu.return
}
}
