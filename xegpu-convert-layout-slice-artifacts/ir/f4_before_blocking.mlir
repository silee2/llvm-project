// -----// IR Dump Before XeGPUBlocking: xegpu-blocking //----- //
gpu.module @kernel [#xevm.target<chip = "cri">] {
  gpu.func @gemm_mxfp(%arg0: memref<256x4096xbf16>, %arg1: memref<2048x256xi8>, %arg2: memref<128x256xf8E8M0FNU>, %arg3: memref<256x256xf32>) kernel {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %c512 = arith.constant 512 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xegpu.create_nd_tdesc %arg0 : memref<256x4096xbf16> -> !xegpu.tensor_desc<16x1024xbf16, #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %3 = xegpu.create_nd_tdesc %arg1 : memref<2048x256xi8> -> !xegpu.tensor_desc<512x16xi8, #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>>
    %4 = xegpu.create_nd_tdesc %arg2 : memref<128x256xf8E8M0FNU> -> !xegpu.tensor_desc<32x16xf8E8M0FNU, #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %5 = xegpu.create_nd_tdesc %arg3 : memref<256x256xf32> -> !xegpu.tensor_desc<16x16xf32, #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %6 = gpu.subgroup_id : index
    %c2 = arith.constant 2 : index
    %7 = arith.remui %6, %c2 : index
    %8 = arith.divui %6, %c2 : index
    %9 = arith.remui %8, %c2 : index
    %c16 = arith.constant 16 : index
    %10 = arith.muli %9, %c16 : index
    %11 = arith.muli %7, %c16 : index
    %12 = arith.remui %10, %c32 : index
    %13 = arith.remui %11, %c32 : index
    %14 = arith.addi %12, %0 : index
    %15 = arith.addi %13, %1 : index
    %16 = xegpu.load_nd %5[%14, %15] <{layout = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf32>
    %17:3 = scf.for %arg4 = %c0 to %c4096 step %c1024 iter_args(%arg5 = %16, %arg6 = %c0, %arg7 = %c0) -> (vector<16x16xf32>, index, index) {
      %18 = arith.muli %7, %c1024 : index
      %19 = arith.remui %18, %c1024 : index
      %20 = arith.addi %19, %arg4 : index
      %21 = xegpu.load_nd %2[%14, %20] <{layout = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<16x1024xbf16, #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x1024xbf16>
      %22 = xegpu.convert_layout %21 <{input_layout = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>, target_layout = #xegpu.layout<inst_data = [1, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<16x1024xbf16>
      %23 = xegpu.convert_layout %21 <{input_layout = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>, target_layout = #xegpu.layout<inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 4]>}> : vector<16x1024xbf16>
      %24 = math.absf %22 {layout_result_0 = #xegpu.layout<inst_data = [1, 16], lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x1024xbf16>
      %25 = vector.shape_cast %24 {layout_result_0 = #xegpu.layout<inst_data = [1, 1, 16], lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>} : vector<16x1024xbf16> to vector<16x32x32xbf16>
      %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16], lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>} dense<0xFF80> : vector<16x32xbf16>
      %26 = vector.multi_reduction <maximumf>, %25, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16], lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>} [2] : vector<16x32x32xbf16> to vector<16x32xbf16>
      %27 = xegpu.convert_layout %26 <{input_layout = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16], lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>, target_layout = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 2], lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>}> : vector<16x32xbf16>
      %28 = arith.bitcast %27 {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 2], lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<16x32xbf16> to vector<16x32xi16>
      %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 2], lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>} dense<32640> : vector<16x32xi16>
      %29 = arith.andi %28, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 2], lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<16x32xi16>
      %30 = arith.bitcast %29 {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 2], lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<16x32xi16> to vector<16x32xbf16>
      %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 2], lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>} dense<4.000000e+00> : vector<16x32xbf16>
      %31 = arith.divf %30, %cst_1 {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 2], lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<16x32xbf16>
      %32 = arith.truncf %31 {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 2], lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<16x32xbf16> to vector<16x32xf8E8M0FNU>
      %33 = xegpu.convert_layout %32 <{input_layout = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 2], lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>, target_layout = #xegpu.layout<inst_data = [8, 2], lane_layout = [8, 1], lane_data = [1, 1]>}> : vector<16x32xf8E8M0FNU>
      %34 = vector.broadcast %32 {layout_result_0 = #xegpu.layout<inst_data = [32, 8, 2], lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>} : vector<16x32xf8E8M0FNU> to vector<32x16x32xf8E8M0FNU>
      %35 = vector.transpose %34, [1, 2, 0] {layout_result_0 = #xegpu.layout<inst_data = [8, 2, 32], lane_layout = [1, 2, 8], lane_data = [1, 1, 4]>} : vector<32x16x32xf8E8M0FNU> to vector<16x32x32xf8E8M0FNU>
      %36 = vector.shape_cast %35 {layout_result_0 = #xegpu.layout<inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 4]>} : vector<16x32x32xf8E8M0FNU> to vector<16x1024xf8E8M0FNU>
      %37 = arith.scaling_truncf %23, %36 {layout_result_0 = #xegpu.layout<inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 4]>} : vector<16x1024xbf16>, vector<16x1024xf8E8M0FNU> to vector<16x1024xf4E2M1FN>
      %38 = arith.muli %9, %c512 : index
      %39 = arith.remui %38, %c512 : index
      %40 = arith.addi %39, %arg6 : index
      %41 = xegpu.load_nd %3[%40, %15] <{layout = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>}> : !xegpu.tensor_desc<512x16xi8, #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>> -> vector<512x16xi8>
      %42 = xegpu.convert_layout %41 <{input_layout = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>, target_layout = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1], order = [1, 0]>}> : vector<512x16xi8>
      %43 = vector.bitcast %42 {layout_result_0 = #xegpu.layout<inst_data = [32, 32], lane_layout = [1, 16], lane_data = [4, 2], order = [1, 0]>} : vector<512x16xi8> to vector<512x32xf4E2M1FN>
      %res1, %res2 = vector.deinterleave %43 {layout_result_0 = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1], order = [1, 0]>, layout_result_1 = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1], order = [1, 0]>} : vector<512x32xf4E2M1FN> -> vector<512x16xf4E2M1FN>
      %44 = vector.transpose %res1, [1, 0] {layout_result_0 = #xegpu.layout<inst_data = [16, 32], lane_layout = [16, 1], lane_data = [1, 4], order = [0, 1]>} : vector<512x16xf4E2M1FN> to vector<16x512xf4E2M1FN>
      %45 = vector.transpose %res2, [1, 0] {layout_result_0 = #xegpu.layout<inst_data = [16, 32], lane_layout = [16, 1], lane_data = [1, 4], order = [0, 1]>} : vector<512x16xf4E2M1FN> to vector<16x512xf4E2M1FN>
      %46 = vector.interleave %44, %45 {layout_result_0 = #xegpu.layout<inst_data = [16, 64], lane_layout = [16, 1], lane_data = [1, 8], order = [0, 1]>} : vector<16x512xf4E2M1FN> -> vector<16x1024xf4E2M1FN>
      %47 = vector.transpose %46, [1, 0] {layout_result_0 = #xegpu.layout<inst_data = [64, 16], lane_layout = [1, 16], lane_data = [8, 1]>} : vector<16x1024xf4E2M1FN> to vector<1024x16xf4E2M1FN>
      %48 = arith.muli %9, %c32 : index
      %49 = arith.remui %48, %c32 : index
      %50 = arith.addi %49, %arg7 : index
      %51 = xegpu.load_nd %4[%50, %15] <{layout = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<32x16xf8E8M0FNU, #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<32x16xf8E8M0FNU>
      %52 = xegpu.convert_layout %51 <{input_layout = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>, target_layout = #xegpu.layout<inst_data = [2, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<32x16xf8E8M0FNU>
      %53 = xegpu.dpas_mx %37, %47, %arg5 scale_a = %33 scale_b = %52 <{layout_a = #xegpu.layout<inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 4]>, layout_a_scale = #xegpu.layout<inst_data = [8, 2], lane_layout = [8, 1], lane_data = [1, 1]>, layout_b = #xegpu.layout<inst_data = [64, 16], lane_layout = [1, 16], lane_data = [8, 1]>, layout_b_scale = #xegpu.layout<inst_data = [2, 16], lane_layout = [1, 16], lane_data = [1, 1]>, layout_cd = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : (vector<16x1024xf4E2M1FN>, vector<1024x16xf4E2M1FN>, vector<16x16xf32>, vector<16x32xf8E8M0FNU>, vector<32x16xf8E8M0FNU>) -> vector<16x16xf32>
      %54 = arith.addi %arg6, %c512 : index
      %55 = arith.addi %arg7, %c32 : index
      scf.yield %53, %54, %55 : vector<16x16xf32>, index, index
    } {layout_operand_3 = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
    xegpu.store_nd %17#0, %5[%14, %15] <{layout = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32, #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

/tmp/quantizeA_F4.mlir:67:23: error: failed to legalize operation 'xegpu.convert_layout' that was explicitly marked illegal: %22159 = "xegpu.convert_layout"(%22158) <{input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>, target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>}> : (vector<8x2xbf16>) -> vector<8x2xbf16>
        %a_amax_i16 = arith.bitcast %a_amax : vector<32x32xbf16> to vector<32x32xi16>
                      ^
/tmp/quantizeA_F4.mlir:67:23: note: see current operation: %22159 = "xegpu.convert_layout"(%22158) <{input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>, target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2], lane_data = [4, 1, 1], order = [0, 2, 1]>, dims = [0]>}> : (vector<8x2xbf16>) -> vector<8x2xbf16>
/tmp/quantizeA_F4.mlir:63:19: error: failed to legalize operation 'vector.multi_reduction' that was explicitly marked illegal: %9939 = "vector.multi_reduction"(%9938, %9) <{kind = #vector.kind<maximumf>, reduction_dims = array<i64: 2>}> : (vector<1x1x16xbf16>, vector<1x1xbf16>) -> vector<1x1xbf16>
        %a_amax = vector.multi_reduction <maximumf>, %a_abs_r, %a_neg_inf [2]
                  ^
/tmp/quantizeA_F4.mlir:63:19: note: see current operation: %9939 = "vector.multi_reduction"(%9938, %9) <{kind = #vector.kind<maximumf>, reduction_dims = array<i64: 2>}> : (vector<1x1x16xbf16>, vector<1x1xbf16>) -> vector<1x1xbf16>
