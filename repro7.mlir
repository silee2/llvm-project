func.func @c16(%a: vector<16xbf16>, %b: vector<1xbf16>) -> vector<16xbf16> {
  %r = vector.shuffle %a, %b [0, 1, 2, 16, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xbf16>, vector<1xbf16>
  return %r : vector<16xbf16>
}
func.func @v1smaller(%a: vector<2xf32>, %b: vector<4xf32>) -> vector<4xf32> {
  %r = vector.shuffle %a, %b [0, 2, 1, 3] : vector<2xf32>, vector<4xf32>
  return %r : vector<4xf32>
}
func.func @f8(%a: vector<32xf8E8M0FNU>, %b: vector<8xf8E8M0FNU>) -> vector<32xf8E8M0FNU> {
  %r = vector.shuffle %a, %b [32, 1, 2, 3, 33, 5, 6, 7, 34, 9, 10, 11, 35, 13, 14, 15, 36, 17, 18, 19, 37, 21, 22, 23, 38, 25, 26, 27, 39, 29, 30, 31] : vector<32xf8E8M0FNU>, vector<8xf8E8M0FNU>
  return %r : vector<32xf8E8M0FNU>
}
