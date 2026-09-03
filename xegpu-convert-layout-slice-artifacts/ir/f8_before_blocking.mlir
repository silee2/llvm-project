// -----// IR Dump Before XeGPUBlocking: xegpu-blocking //----- //
gpu.module @kernel [#xevm.target<chip = "cri">] {
  gpu.func @gemm_mxfp(%arg0: memref<256x4096xbf16>, %arg1: memref<4096x256xf8E5M2>, %arg2: memref<128x256xf8E8M0FNU>, %arg3: memref<256x256xf32>) kernel {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = xegpu.create_nd_tdesc %arg0 : memref<256x4096xbf16> -> !xegpu.tensor_desc<16x1024xbf16, #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %3 = xegpu.create_nd_tdesc %arg1 : memref<4096x256xf8E5M2> -> !xegpu.tensor_desc<1024x16xf8E5M2, #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>>
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
    %17:2 = scf.for %arg4 = %c0 to %c4096 step %c1024 iter_args(%arg5 = %16, %arg6 = %c0) -> (vector<16x16xf32>, index) {
      %18 = arith.muli %7, %c1024 : index
      %19 = arith.remui %18, %c1024 : index
      %20 = arith.addi %19, %arg4 : index
      %21 = xegpu.load_nd %2[%14, %20] <{layout = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<16x1024xbf16, #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x1024xbf16>
      %22 = xegpu.convert_layout %21 <{input_layout = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>, target_layout = #xegpu.layout<inst_data = [1, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<16x1024xbf16>
      %23 = xegpu.convert_layout %21 <{input_layout = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>, target_layout = #xegpu.layout<inst_data = [8, 32], lane_layout = [1, 16], lane_data = [1, 2]>}> : vector<16x1024xbf16>
      %24 = math.absf %22 {layout_result_0 = #xegpu.layout<inst_data = [1, 16], lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x1024xbf16>
      %25 = vector.shape_cast %24 {layout_result_0 = #xegpu.layout<inst_data = [1, 1, 16], lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>} : vector<16x1024xbf16> to vector<16x32x32xbf16>
      %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16], lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>} dense<0xFF80> : vector<16x32xbf16>
      %26 = vector.multi_reduction <maximumf>, %25, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16], lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>} [2] : vector<16x32x32xbf16> to vector<16x32xbf16>
      %27 = xegpu.convert_layout %26 <{input_layout = #xegpu.slice<#xegpu.layout<inst_data = [1, 1, 16], lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>, dims = [2]>, target_layout = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 1], lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>}> : vector<16x32xbf16>
      %28 = arith.bitcast %27 {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 1], lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<16x32xbf16> to vector<16x32xi16>
      %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 1], lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>} dense<32640> : vector<16x32xi16>
      %29 = arith.andi %28, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 1], lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<16x32xi16>
      %30 = arith.bitcast %29 {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 1], lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<16x32xi16> to vector<16x32xbf16>
      %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 1], lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>} dense<3.276800e+04> : vector<16x32xbf16>
      %31 = arith.divf %30, %cst_1 {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 1], lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<16x32xbf16>
      %32 = arith.truncf %31 {layout_result_0 = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 1], lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>} : vector<16x32xbf16> to vector<16x32xf8E8M0FNU>
      %33 = xegpu.convert_layout %32 <{input_layout = #xegpu.slice<#xegpu.layout<inst_data = [32, 8, 1], lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>, target_layout = #xegpu.layout<inst_data = [8, 1], lane_layout = [8, 1], lane_data = [1, 1]>}> : vector<16x32xf8E8M0FNU>
      %34 = vector.broadcast %32 {layout_result_0 = #xegpu.layout<inst_data = [32, 8, 1], lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>} : vector<16x32xf8E8M0FNU> to vector<32x16x32xf8E8M0FNU>
      %35 = vector.transpose %34, [1, 2, 0] {layout_result_0 = #xegpu.layout<inst_data = [8, 1, 32], lane_layout = [1, 1, 16], lane_data = [1, 1, 2]>} : vector<32x16x32xf8E8M0FNU> to vector<16x32x32xf8E8M0FNU>
      %36 = vector.shape_cast %35 {layout_result_0 = #xegpu.layout<inst_data = [8, 32], lane_layout = [1, 16], lane_data = [1, 2]>} : vector<16x32x32xf8E8M0FNU> to vector<16x1024xf8E8M0FNU>
      %37 = arith.scaling_truncf %23, %36 {layout_result_0 = #xegpu.layout<inst_data = [8, 32], lane_layout = [1, 16], lane_data = [1, 2]>} : vector<16x1024xbf16>, vector<16x1024xf8E8M0FNU> to vector<16x1024xf8E5M2>
      %38 = arith.muli %9, %c1024 : index
      %39 = arith.remui %38, %c1024 : index
      %40 = arith.addi %39, %arg4 : index
      %41 = xegpu.load_nd %3[%40, %15] <{layout = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>}> : !xegpu.tensor_desc<1024x16xf8E5M2, #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>> -> vector<1024x16xf8E5M2>
      %42 = arith.muli %9, %c32 : index
      %43 = arith.remui %42, %c32 : index
      %44 = arith.addi %43, %arg6 : index
      %45 = xegpu.load_nd %4[%44, %15] <{layout = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<32x16xf8E8M0FNU, #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<32x16xf8E8M0FNU>
      %46 = xegpu.convert_layout %45 <{input_layout = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>, target_layout = #xegpu.layout<inst_data = [1, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<32x16xf8E8M0FNU>
      %47 = xegpu.dpas_mx %37, %41, %arg5 scale_a = %33 scale_b = %46 <{layout_a = #xegpu.layout<inst_data = [8, 32], lane_layout = [1, 16], lane_data = [1, 2]>, layout_a_scale = #xegpu.layout<inst_data = [8, 1], lane_layout = [8, 1], lane_data = [1, 1]>, layout_b = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>, layout_b_scale = #xegpu.layout<inst_data = [1, 16], lane_layout = [1, 16], lane_data = [1, 1]>, layout_cd = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : (vector<16x1024xf8E5M2>, vector<1024x16xf8E5M2>, vector<16x16xf32>, vector<16x32xf8E8M0FNU>, vector<32x16xf8E8M0FNU>) -> vector<16x16xf32>
      %48 = arith.addi %arg6, %c32 : index
      scf.yield %47, %48 : vector<16x16xf32>, index
    } {layout_operand_3 = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
    xegpu.store_nd %17#0, %5[%14, %15] <{layout = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32, #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F8.mlir:107:26: error: failed to legalize operation 'xegpu.convert_layout' that was explicitly marked illegal: %24687 = "xegpu.convert_layout"(%24560) <{input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>, target_layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>}> : (vector<8x1xf8E8M0FNU>) -> vector<8x1xf8E8M0FNU>
        %new_c_partial = xegpu.dpas_mx %a, %b, %c_partial scale_a = %a_scale scale_b = %scale_b
                         ^
mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F8.mlir:107:26: note: see current operation: %24687 = "xegpu.convert_layout"(%24560) <{input_layout = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1, 1], lane_data = [2, 1, 1], order = [0, 2, 1]>, dims = [0]>, target_layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>}> : (vector<8x1xf8E8M0FNU>) -> vector<8x1xf8E8M0FNU>
mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F8.mlir:69:19: error: failed to legalize operation 'vector.multi_reduction' that was explicitly marked illegal: %9971 = "vector.multi_reduction"(%9970, %9) <{kind = #vector.kind<maximumf>, reduction_dims = array<i64: 2>}> : (vector<1x1x16xbf16>, vector<1x1xbf16>) -> vector<1x1xbf16>
        %a_amax = vector.multi_reduction <maximumf>, %a_abs_r, %a_neg_inf [2]
                  ^
mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F8.mlir:69:19: note: see current operation: %9971 = "vector.multi_reduction"(%9970, %9) <{kind = #vector.kind<maximumf>, reduction_dims = array<i64: 2>}> : (vector<1x1x16xbf16>, vector<1x1xbf16>) -> vector<1x1xbf16>
