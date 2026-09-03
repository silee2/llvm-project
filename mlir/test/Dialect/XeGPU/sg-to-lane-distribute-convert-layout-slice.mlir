// RUN: mlir-opt  --xevm-attach-target='module=xevm_* chip=cri' \
// RUN: --test-xegpu-sg-to-lane-distribute --split-input-file %s | FileCheck %s

// Slice-attributed `xegpu.convert_layout` conversions collected from the
// workgroup-level mx-fp GEMM quantize-A integration tests
// (`simple_mxfp_gemm_quantizeA_F4.mlir` and `simple_mxfp_gemm_quantizeA_F8.mlir`),
// as they appear in the IR right before `xegpu-sg-to-lane-distribute`.
//
// These all sit on the boundary between the rank-3 reduction/broadcast layouts
// used by the in-kernel quantization of A (expressed as `#xegpu.slice` of a 3-D
// layout) and the rank-2 layouts used by `load_nd` / `dpas_mx`:
//
//   load_nd (plain 2D) --A--> slice dims=[1] --> reduce --> slice dims=[2]
//                                                                |
//                                                                B
//                                                                v
//                                        slice dims=[0], order=[0,2,1]
//                                                                |
//                                                                C
//                                                                v
//                                            plain 2D lane_layout=[8,1]
//                                                                |
//                                                                v
//                                                     dpas_mx scale_a
//
// Each case is in its own split so that one failure does not mask another.

// -----
// Group A, common to the F4 and F8 kernels (1024 occurrences in each).
// Source: `xegpu.load_nd` of the A tile.
// Consumer: `vector.extract_strided_slice` feeding the 3-D
// `vector.multi_reduction` that computes the per-block amax.
gpu.module @xevm_module {
gpu.func @convert_layout_slice_group_a() {
  %src = "test.some_op"() : () -> vector<8x16xbf16>
  %cvt = xegpu.convert_layout %src
    <{
      input_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
      target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [1]>
    }> : vector<8x16xbf16>
  gpu.return
}
}

// -----
// Group B, F4 kernel (32 occurrences).
// Source: `vector.insert_strided_slice` assembling the reduced amax.
// Consumer: `arith.bitcast` of the amax to i16 for the mantissa mask.
gpu.module @xevm_module {
gpu.func @convert_layout_slice_group_b_f4() {
  %src = "test.some_op"() : () -> vector<8x2xbf16>
  %cvt = xegpu.convert_layout %src
    <{
      input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>,
      target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>
    }> : vector<8x2xbf16>
  gpu.return
}
}

// -----
// Group C, F4 kernel (32 occurrences).
// Source: `arith.truncf` producing the f8E8M0 scale.
// Consumer: `xegpu.dpas_mx` as `scale_a`, whose `layout_a_scale` is the plain
// rank-2 `lane_layout = [8, 1]`.
gpu.module @xevm_module {
gpu.func @convert_layout_slice_group_c_f4() {
  %src = "test.some_op"() : () -> vector<8x2xf8E8M0FNU>
  %cvt = xegpu.convert_layout %src
    <{
      input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>,
      target_layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>
    }> : vector<8x2xf8E8M0FNU>
  gpu.return
}
}

// -----
// Group B, F8 kernel (64 occurrences). Same structure as the F4 case, but the
// distributed dimension of the target moves: lane_layout [16, 1, 1] with
// lane_data [2, 1, 1] rather than [8, 1, 2] / [4, 1, 1].
gpu.module @xevm_module {
gpu.func @convert_layout_slice_group_b_f8() {
  %src = "test.some_op"() : () -> vector<8x1xbf16>
  %cvt = xegpu.convert_layout %src
    <{
      input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>,
      target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>
    }> : vector<8x1xbf16>
  gpu.return
}
}

// -----
// Group C, F8 kernel (64 occurrences).
gpu.module @xevm_module {
gpu.func @convert_layout_slice_group_c_f8() {
  %src = "test.some_op"() : () -> vector<8x1xf8E8M0FNU>
  %cvt = xegpu.convert_layout %src
    <{
      input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>,
      target_layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>
    }> : vector<8x1xf8E8M0FNU>
  gpu.return
}
}
