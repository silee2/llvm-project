func.func @famB(%ms: memref<32xf8E8M0FNU>, %ma: memref<32xbf16>, %mo: memref<32xf4E2M1FN>) {
  %ci = arith.constant 0 : index
  %scale = vector.load %ms[%ci] : memref<32xf8E8M0FNU>, vector<32xf8E8M0FNU>
  %a = vector.load %ma[%ci] : memref<32xbf16>, vector<32xbf16>
  %c127 = arith.constant dense<127> : vector<32xi8>
  %c23  = arith.constant dense<23> : vector<32xi32>
  %cnan = arith.constant dense<2143289344> : vector<32xi32>
  %0 = arith.bitcast %scale : vector<32xf8E8M0FNU> to vector<32xi8>
  %1 = arith.cmpi eq, %0, %c127 : vector<32xi8>
  %2 = arith.extui %0 : vector<32xi8> to vector<32xi32>
  %3 = arith.shli %2, %c23 : vector<32xi32>
  %4 = arith.select %1, %cnan, %3 : vector<32xi1>, vector<32xi32>
  %5 = arith.bitcast %4 : vector<32xi32> to vector<32xf32>
  %6 = arith.truncf %5 : vector<32xf32> to vector<32xbf16>
  %7 = arith.divf %a, %6 : vector<32xbf16>
  %8 = arith.truncf %7 : vector<32xbf16> to vector<32xf4E2M1FN>
  vector.store %8, %mo[%ci] : memref<32xf4E2M1FN>, vector<32xf4E2M1FN>
  return
}
