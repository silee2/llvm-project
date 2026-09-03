// probe 1: distributed shape of group C F8 INPUT layout on vector<8x1>
gpu.module @xevm_module {
gpu.func @probe_input() {
  %src = "test.some_op"() {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>} : () -> vector<8x1xbf16>
  %neg = arith.negf %src {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<8x1xbf16>
  "test.some_sink"(%neg) : (vector<8x1xbf16>) -> ()
  gpu.return
}
}

// -----
// probe 2: distributed shape of group C F8 TARGET layout on vector<8x1>
gpu.module @xevm_module {
gpu.func @probe_target() {
  %src = "test.some_op"() {layout_result_0 = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>} : () -> vector<8x1xbf16>
  %neg = arith.negf %src {layout_result_0 = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>} : vector<8x1xbf16>
  "test.some_sink"(%neg) : (vector<8x1xbf16>) -> ()
  gpu.return
}
}
