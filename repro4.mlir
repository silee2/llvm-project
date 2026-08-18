// RUN: mlir-opt --xevm-attach-target='module=xevm_* chip=cri' \
// RUN:   --xegpu-propagate-layout='layout-kind=inst' %s

#dpas_a_scale = #xegpu.layout<inst_data = [8, 2], lane_layout = [8, 1], lane_data = [1, 1]>
#dpas_b_scale = #xegpu.layout<inst_data = [2, 16], lane_layout = [1, 16], lane_data = [1, 1]>
#a = #xegpu.layout<inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 4]>
#b = #xegpu.layout<inst_data = [64, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#c = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>

gpu.module @xevm_module {
  gpu.func @reduce_then_broadcast(%A: memref<16x1024xbf16>,
                                  %B: memref<1024x16xf4E2M1FN>,
                                  %BS: memref<32x16xf8E8M0FNU>,
                                  %C: memref<16x16xf32>) kernel {
    %c0 = arith.constant 0 : index
    %at = xegpu.create_nd_tdesc %A : memref<16x1024xbf16> -> !xegpu.tensor_desc<16x1024xbf16>
    %bt = xegpu.create_nd_tdesc %B : memref<1024x16xf4E2M1FN> -> !xegpu.tensor_desc<1024x16xf4E2M1FN>
    %bst = xegpu.create_nd_tdesc %BS : memref<32x16xf8E8M0FNU> -> !xegpu.tensor_desc<32x16xf8E8M0FNU>
    %ct = xegpu.create_nd_tdesc %C : memref<16x16xf32> -> !xegpu.tensor_desc<16x16xf32>

    %a_bf16 = xegpu.load_nd %at[%c0, %c0] : !xegpu.tensor_desc<16x1024xbf16> -> vector<16x1024xbf16>
    %b = xegpu.load_nd %bt[%c0, %c0] {layout = #b} : !xegpu.tensor_desc<1024x16xf4E2M1FN> -> vector<1024x16xf4E2M1FN>
    %scale_b = xegpu.load_nd %bst[%c0, %c0] {layout = #dpas_b_scale} : !xegpu.tensor_desc<32x16xf8E8M0FNU> -> vector<32x16xf8E8M0FNU>
    %c = xegpu.load_nd %ct[%c0, %c0] {layout = #c} : !xegpu.tensor_desc<16x16xf32> -> vector<16x16xf32>

    %neg_inf = arith.constant dense<0xFF80> : vector<16x32xbf16>
    %abs = math.absf %a_bf16 : vector<16x1024xbf16>
    %r = vector.shape_cast %abs : vector<16x1024xbf16> to vector<16x32x32xbf16>
    %amax = vector.multi_reduction <maximumf>, %r, %neg_inf [2]
        : vector<16x32x32xbf16> to vector<16x32xbf16>

    %i16 = arith.bitcast %amax : vector<16x32xbf16> to vector<16x32xi16>
    %mask = arith.constant dense<0x7F80> : vector<16x32xi16>
    %pow2_i16 = arith.andi %i16, %mask : vector<16x32xi16>
    %pow2 = arith.bitcast %pow2_i16 : vector<16x32xi16> to vector<16x32xbf16>
    %four = arith.constant dense<4.0> : vector<16x32xbf16>
    %sc_bf16 = arith.divf %pow2, %four : vector<16x32xbf16>
    %sc = arith.truncf %sc_bf16 : vector<16x32xbf16> to vector<16x32xf8E8M0FNU>

    %lead = vector.broadcast %sc
        : vector<16x32xf8E8M0FNU> to vector<32x16x32xf8E8M0FNU>
    %t = vector.transpose %lead, [1, 2, 0]
        : vector<32x16x32xf8E8M0FNU> to vector<16x32x32xf8E8M0FNU>
    %full = vector.shape_cast %t
        : vector<16x32x32xf8E8M0FNU> to vector<16x1024xf8E8M0FNU>

    %af4 = arith.scaling_truncf %a_bf16, %full
        : vector<16x1024xbf16>, vector<16x1024xf8E8M0FNU> to vector<16x1024xf4E2M1FN>

    %res = xegpu.dpas_mx %af4, %b, %c scale_a = %sc scale_b = %scale_b
        {layout_a = #a, layout_b = #b, layout_cd = #c,
         layout_a_scale = #dpas_a_scale, layout_b_scale = #dpas_b_scale}
        : (vector<16x1024xf4E2M1FN>, vector<1024x16xf4E2M1FN>, vector<16x16xf32>,
           vector<16x32xf8E8M0FNU>, vector<32x16xf8E8M0FNU>) -> vector<16x16xf32>
    xegpu.store_nd %res, %ct[%c0, %c0] {layout = #c} : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32>
    gpu.return
  }
}
