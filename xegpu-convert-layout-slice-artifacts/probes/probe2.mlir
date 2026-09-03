// Group B F8 (succeeds); keep result live so the materialized cast shows the
// per-lane shape of C-F8's INPUT layout (== B-F8's target layout).
gpu.module @xevm_module {
gpu.func @probe_c_f8_input() {
  %src = "test.some_op"() : () -> vector<8x1xbf16>
  %cvt = xegpu.convert_layout %src
    <{
      input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>,
      target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>
    }> : vector<8x1xbf16>
  "test.some_sink"(%cvt) : (vector<8x1xbf16>) -> ()
  gpu.return
}
}

// -----
// C-F8's TARGET layout, via a load_nd that the pass distributes.
gpu.module @xevm_module {
gpu.func @probe_c_f8_target(%arg0: memref<8x1xbf16>) {
  %c0 = arith.constant 0 : index
  %td = xegpu.create_nd_tdesc %arg0 : memref<8x1xbf16> -> !xegpu.tensor_desc<8x1xbf16, #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>>
  %v = xegpu.load_nd %td[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>}> : !xegpu.tensor_desc<8x1xbf16, #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>> -> vector<8x1xbf16>
  "test.some_sink"(%v) : (vector<8x1xbf16>) -> ()
  gpu.return
}
}
