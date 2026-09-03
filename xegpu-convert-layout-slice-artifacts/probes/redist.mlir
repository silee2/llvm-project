module @m attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @redistribute(%in: memref<128xf8E8M0FNU>, %out: memref<128xf8E8M0FNU>) kernel {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c8i = arith.constant 8 : i32
      %c16i = arith.constant 16 : i32
      %lane = gpu.lane_id
      %off = arith.muli %lane, %c8 : index
      %src = vector.load %in[%off] : memref<128xf8E8M0FNU>, vector<8xf8E8M0FNU>
      %row = arith.andi %lane, %c7 : index
      %own = vector.extract %src[%row] : f8E8M0FNU from vector<8xf8E8M0FNU>
      %partner, %valid = gpu.shuffle xor %own, %c8i, %c16i : f8E8M0FNU
      %res = vector.from_elements %own, %partner : vector<2xf8E8M0FNU>
      %ooff = arith.muli %lane, %c2 : index
      vector.store %res, %out[%ooff] : memref<128xf8E8M0FNU>, vector<2xf8E8M0FNU>
      gpu.return
    }
  }
}
