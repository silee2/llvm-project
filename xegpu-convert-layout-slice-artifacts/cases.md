# Slice-attributed `xegpu.convert_layout` cases that fail `xegpu-sg-to-lane-distribute`

Source kernels:
- `mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F4.mlir`
- `mlir/test/Integration/Dialect/XeGPU/WG/simple_mxfp_gemm_quantizeA_F8.mlir`

Instructions below are verbatim from the IR immediately before `xegpu-sg-to-lane-distribute`
(`--mlir-print-ir-before=xegpu-sg-to-lane-distribute`). Subgroup size is 16.

In the generated sequences, `%src` is the distributed operand, i.e. `adaptor.getSource()`,
and the result must have the distributed `target_layout` type.

---

## Case C/F8

64 occurrences. `f8_before_sg2lane.clean.mlir:6500`

### Instruction

```mlir
%6422 = xegpu.convert_layout %6358
  <{input_layout  = #xegpu.slice<#xegpu.layout<lane_layout = [16, 1, 1],
                                               lane_data   = [2, 1, 1],
                                               order       = [0, 2, 1]>, dims = [0]>,
    target_layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>}>
  : vector<8x1xf8E8M0FNU>
```

Producer: `%6358 = arith.truncf %6294 : vector<8x1xbf16> to vector<8x1xf8E8M0FNU>`

Consumer: `xegpu.dpas_mx %6678, %6745, %arg5 scale_a = %6422 scale_b = %6812 ...`
with `layout_a_scale = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>`

### Layouts

| | `input_layout` | `target_layout` |
|---|---|---|
| form | `#xegpu.slice` | `#xegpu.layout` |
| effective `lane_layout` | `[1, 1]` | `[8, 1]` |
| effective `lane_data` | `[1, 1]` | `[1, 1]` |
| effective `order` | `[1, 0]` | `[1, 0]` |
| distributed type | `vector<8x1xf8E8M0FNU>` | `vector<1x1xf8E8M0FNU>` |

`getDistributedDimLaneStride` on `input_layout` fails: no non-sliced dim has extent > 1.

### Sequence

```mlir
%lane = gpu.lane_id
%c8   = arith.constant 8 : index
%row  = arith.remui %lane, %c8 : index
%elem = vector.extract %src[%row, 0] : f8E8M0FNU from vector<8x1xf8E8M0FNU>
%res  = vector.from_elements %elem : vector<1x1xf8E8M0FNU>
```

`8` in `%c8` is `targetLaneLayout[0]`, which must equal `shape[0]`.

---

## Case B/F4

32 occurrences. `f4_before_sg2lane.clean.mlir:5588`

### Instruction

```mlir
%5510 = xegpu.convert_layout %5509
  <{input_layout  = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16],
                                               lane_data   = [1, 1, 1]>, dims = [2]>,
    target_layout = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2],
                                               lane_data   = [4, 1, 1],
                                               order       = [0, 2, 1]>, dims = [0]>}>
  : vector<8x2xbf16>
```

Producer: `%5509 = vector.insert_strided_slice %4921, %5508 offsets = [7, 1], strides = [1, 1] : vector<1x1xbf16> into vector<8x2xbf16>`

Consumer: `%6038 = arith.bitcast %5510 : vector<8x2xbf16> to vector<8x2xi16>`

### Layouts

| | `input_layout` | `target_layout` |
|---|---|---|
| form | `#xegpu.slice` | `#xegpu.slice` |
| effective `lane_layout` | `[1, 1]` | `[1, 2]` |
| effective `lane_data` | `[1, 1]` | `[1, 1]` |
| effective `order` | `[1, 0]` | `[1, 0]` |
| distributed type | `vector<8x2xbf16>` | `vector<8x1xbf16>` |

`getDistributedDimLaneStride` on `target_layout` returns `8`: parent dim 2 has extent 2, and
under `order = [0, 2, 1]` dim 0 is fastest with extent 8, so dim 2 has lane stride 8.

### Sequence

```mlir
%flat        = vector.shape_cast %src : vector<8x2xbf16> to vector<16xbf16>
%even, %odd  = vector.deinterleave %flat : vector<16xbf16> -> vector<8xbf16>
%lane        = gpu.lane_id
%c8          = arith.constant 8 : index
%c0          = arith.constant 0 : index
%half        = arith.divui %lane, %c8 : index
%isFirst     = arith.cmpi eq, %half, %c0 : index
%sel         = arith.select %isFirst, %even, %odd : vector<8xbf16>
%res         = vector.shape_cast %sel : vector<8xbf16> to vector<8x1xbf16>
```

`%even` is column 0, `%odd` is column 1, because `vector<8x2>` is row-major so a column is a
stride-2 subset of the flattened value.

`8` in `%c8` is `distributedDimLaneStride` from `target_layout`. The `8` in `vector<8xbf16>`
and `vector<8x1xbf16>` is `shape[0]`. These two are unrelated and coincide only in this kernel.

---

## Case C/F4

32 occurrences. `f4_before_sg2lane.clean.mlir:6276`

### Instruction

```mlir
%6198 = xegpu.convert_layout %6166
  <{input_layout  = #xegpu.slice<#xegpu.layout<lane_layout = [8, 1, 2],
                                               lane_data   = [4, 1, 1],
                                               order       = [0, 2, 1]>, dims = [0]>,
    target_layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>}>
  : vector<8x2xf8E8M0FNU>
```

Producer: `%6166 = arith.truncf %6134 : vector<8x2xbf16> to vector<8x2xf8E8M0FNU>`

Consumer: `xegpu.dpas_mx %6326, %6472, %arg5 scale_a = %6198 scale_b = %6492 ...`
with `layout_a_scale = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>`

### Layouts

| | `input_layout` | `target_layout` |
|---|---|---|
| form | `#xegpu.slice` | `#xegpu.layout` |
| effective `lane_layout` | `[1, 2]` | `[8, 1]` |
| effective `lane_data` | `[1, 1]` | `[1, 1]` |
| effective `order` | `[1, 0]` | `[1, 0]` |
| distributed type | `vector<8x1xf8E8M0FNU>` | `vector<1x2xf8E8M0FNU>` |

`getDistributedDimLaneStride` on `input_layout` returns `8`, same derivation as B/F4's target.

Lane `l` holds column `l / 8`, all 8 rows. Target lane `i` needs `[i, 0]` and `[i, 1]`.

### Sequence

```mlir
%lane    = gpu.lane_id
%c8      = arith.constant 8 : index
%row     = arith.remui %lane, %c8 : index
%own     = vector.extract %src[%row, 0] : f8E8M0FNU from vector<8x1xf8E8M0FNU>
%c8_i32  = arith.constant 8 : i32
%c16_i32 = arith.constant 16 : i32
%partner, %valid = gpu.shuffle xor %own, %c8_i32, %c16_i32 : f8E8M0FNU
%res     = vector.from_elements %own, %partner : vector<1x2xf8E8M0FNU>
```

`lane_id ^ 8` flips only bit 3, so the partner holds the same row in the other column. For
lanes 0..7 `%own` is column 0 and `%partner` is column 1. Lanes 8..15 get them reversed, which
is unused because `target_layout` `lane_layout = [8, 1]` occupies only 8 lanes.

`%c8` (index) is `targetLaneLayout[0]`, which must equal `shape[0]`.
`%c8_i32` is `distributedDimLaneStride` from `input_layout`. Unrelated quantities.
`%c16_i32` is the subgroup size, required by `gpu.shuffle`.

---

## Summary

| | input effective `lane_layout` | target effective `lane_layout` | distributed input | distributed target | sequence |
|---|---|---|---|---|---|
| C/F8 | `[1, 1]` | `[8, 1]` | `vector<8x1>` | `vector<1x1>` | 1 extract |
| B/F4 | `[1, 1]` | `[1, 2]` | `vector<8x2>` | `vector<8x1>` | 1 deinterleave + 1 select |
| C/F4 | `[1, 2]` | `[8, 1]` | `vector<8x1>` | `vector<1x2>` | 1 extract + 1 shuffle |

The three input/target effective `lane_layout` pairs are disjoint.

## Verification status

Verbatim from tool output:
- all three `xegpu.convert_layout` instructions, their producers and consumers
- all three fail `xegpu-sg-to-lane-distribute` today
- `vector.deinterleave` lowers to two `llvm.shufflevector` with constant masks
  (`ConvertVectorToLLVM.cpp:1897`), rank-1 only

Measured with probe runs through `--test-xegpu-sg-to-lane-distribute`:
- distributed types for C/F8 target, C/F4 target, B/F4 input, B/F4 target

Compiled end-to-end to `gpu.binary` for `chip=cri`:
- the C/F4 op triple `vector.extract` (dynamic index) + `gpu.shuffle xor` + `vector.from_elements`
  on `f8E8M0FNU`, producing `llvm.extractelement`, `@_Z21sub_group_shuffle_xorcj`, and two
  `llvm.insertelement`. Tested with a rank-1 `vector<8xf8E8M0FNU>` source, not the rank-2
  `vector<8x1xf8E8M0FNU>` shown above.

Not verified:
- the exact spelling of all three sequences, in particular whether `vector.extract` with mixed
  dynamic and static indices is preferred over a `vector.shape_cast` to rank 1 first, and whether
  `vector.deinterleave` needs the explicit `shape_cast` or whether `xegpu-vector-linearize`
  handles it later in the pipeline
- that the sequences produce correct results at runtime; none has been executed
