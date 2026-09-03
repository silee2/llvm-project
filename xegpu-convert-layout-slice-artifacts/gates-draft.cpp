#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

static bool isRank2UnitLaneDataRowMajor(xegpu::DistributeLayoutAttr layout) {
  if (layout.getRank() != 2)
    return false;
  SmallVector<int64_t> laneData = layout.getEffectiveLaneDataAsInt();
  SmallVector<int64_t> order = layout.getEffectiveOrderAsInt();
  return laneData == SmallVector<int64_t>{1, 1} &&
         order == SmallVector<int64_t>{1, 0};
}

static FailureOr<int64_t> getDistributedDimLaneStride(xegpu::SliceAttr slice) {
  xegpu::SliceAttr flattened = slice.flatten();
  auto parent = dyn_cast<xegpu::LayoutAttr>(flattened.getParent());
  if (!parent)
    return failure();

  SmallVector<int64_t> parentLaneLayout = parent.getEffectiveLaneLayoutAsInt();
  SmallVector<int64_t> parentOrder = parent.getEffectiveOrderAsInt();
  if (parentLaneLayout.size() != parentOrder.size())
    return failure();

  llvm::SmallDenseSet<int64_t> slicedDims(
      flattened.getDims().asArrayRef().begin(),
      flattened.getDims().asArrayRef().end());

  std::optional<int64_t> distributedDim;
  for (int64_t dim = 0, rank = parentLaneLayout.size(); dim < rank; ++dim) {
    if (slicedDims.contains(dim) || parentLaneLayout[dim] == 1)
      continue;
    if (distributedDim)
      return failure();
    distributedDim = dim;
  }
  if (!distributedDim)
    return failure();

  int64_t stride = 1;
  for (int64_t dim : parentOrder) {
    if (dim == *distributedDim)
      return stride;
    stride *= parentLaneLayout[dim];
  }
  return failure();
}

static LogicalResult matchExtractOnly(xegpu::ConvertLayoutOp op,
                                      ConversionPatternRewriter &rewriter,
                                      int64_t subgroupSize,
                                      int64_t &targetLaneCount) {
  auto inputLayout = op.getEffectiveInputLayout();
  auto targetLayout = op.getTargetLayoutAttr();

  if (!isa<xegpu::SliceAttr>(inputLayout))
    return rewriter.notifyMatchFailure(op, "input_layout must be #xegpu.slice");
  if (!isa<xegpu::LayoutAttr>(targetLayout))
    return rewriter.notifyMatchFailure(op,
                                       "target_layout must be #xegpu.layout");
  if (!isRank2UnitLaneDataRowMajor(inputLayout) ||
      !isRank2UnitLaneDataRowMajor(targetLayout))
    return rewriter.notifyMatchFailure(
        op, "both layouts must be rank 2 with effective lane_data [1, 1] and "
            "effective order [1, 0]");

  auto vectorType = cast<VectorType>(op.getType());
  FailureOr<VectorType> distributedInput =
      xegpu::getDistributedVectorType(vectorType, inputLayout);
  FailureOr<VectorType> distributedTarget =
      xegpu::getDistributedVectorType(vectorType, targetLayout);
  if (failed(distributedInput) || failed(distributedTarget))
    return rewriter.notifyMatchFailure(
        op, "value type must be distributable by both layouts");
  if (*distributedInput != vectorType)
    return rewriter.notifyMatchFailure(
        op, "distributed input_layout type must equal the value type");
  if (distributedTarget->getShape() != ArrayRef<int64_t>{1, 1})
    return rewriter.notifyMatchFailure(
        op, "distributed target_layout type must be vector<1x1>");

  SmallVector<int64_t> inputLaneLayout =
      inputLayout.getEffectiveLaneLayoutAsInt();
  SmallVector<int64_t> targetLaneLayout =
      targetLayout.getEffectiveLaneLayoutAsInt();
  if (targetLaneLayout[0] > subgroupSize)
    return rewriter.notifyMatchFailure(
        op, "target_layout effective lane_layout[0] must not exceed the "
            "subgroup size");
  if (inputLaneLayout != SmallVector<int64_t>{1, 1})
    return rewriter.notifyMatchFailure(
        op, "input_layout effective lane_layout must be [1, 1]");
  if (targetLaneLayout[0] <= 1 || targetLaneLayout[1] != 1)
    return rewriter.notifyMatchFailure(
        op, "target_layout effective lane_layout must be [n, 1] with n > 1");

  targetLaneCount = targetLaneLayout[0];
  return success();
}

static LogicalResult
matchDeinterleaveAndSelect(xegpu::ConvertLayoutOp op,
                           ConversionPatternRewriter &rewriter,
                           int64_t subgroupSize,
                           int64_t &distributedDimLaneStride) {
  auto inputLayout = op.getEffectiveInputLayout();
  auto targetLayout = op.getTargetLayoutAttr();

  if (!isa<xegpu::SliceAttr>(inputLayout))
    return rewriter.notifyMatchFailure(op, "input_layout must be #xegpu.slice");
  auto targetSlice = dyn_cast<xegpu::SliceAttr>(targetLayout);
  if (!targetSlice)
    return rewriter.notifyMatchFailure(op,
                                       "target_layout must be #xegpu.slice");
  if (!isRank2UnitLaneDataRowMajor(inputLayout) ||
      !isRank2UnitLaneDataRowMajor(targetLayout))
    return rewriter.notifyMatchFailure(
        op, "both layouts must be rank 2 with effective lane_data [1, 1] and "
            "effective order [1, 0]");

  auto vectorType = cast<VectorType>(op.getType());
  FailureOr<VectorType> distributedInput =
      xegpu::getDistributedVectorType(vectorType, inputLayout);
  FailureOr<VectorType> distributedTarget =
      xegpu::getDistributedVectorType(vectorType, targetLayout);
  if (failed(distributedInput) || failed(distributedTarget))
    return rewriter.notifyMatchFailure(
        op, "value type must be distributable by both layouts");
  if (*distributedInput != vectorType)
    return rewriter.notifyMatchFailure(
        op, "distributed input_layout type must equal the value type");
  if (distributedTarget->getShape() !=
      ArrayRef<int64_t>{vectorType.getDimSize(0), 1})
    return rewriter.notifyMatchFailure(
        op, "distributed target_layout type must be vector<shape[0]x1>");

  FailureOr<int64_t> stride = getDistributedDimLaneStride(targetSlice);
  if (failed(stride))
    return rewriter.notifyMatchFailure(
        op, "target_layout parent must have exactly one non-sliced dimension "
            "with effective lane_layout extent greater than one");

  SmallVector<int64_t> inputLaneLayout =
      inputLayout.getEffectiveLaneLayoutAsInt();
  SmallVector<int64_t> targetLaneLayout =
      targetLayout.getEffectiveLaneLayoutAsInt();
  if (*stride * targetLaneLayout[1] != subgroupSize)
    return rewriter.notifyMatchFailure(
        op, "target_layout distributed dimension lane stride times its extent "
            "must equal the subgroup size");
  if (inputLaneLayout != SmallVector<int64_t>{1, 1})
    return rewriter.notifyMatchFailure(
        op, "input_layout effective lane_layout must be [1, 1]");
  if (targetLaneLayout != SmallVector<int64_t>{1, 2})
    return rewriter.notifyMatchFailure(
        op, "target_layout effective lane_layout must be [1, 2]");

  distributedDimLaneStride = *stride;
  return success();
}

static LogicalResult
matchExtractAndShuffleXor(xegpu::ConvertLayoutOp op,
                          ConversionPatternRewriter &rewriter,
                          int64_t subgroupSize, int64_t &targetLaneCount,
                          int64_t &distributedDimLaneStride) {
  auto inputLayout = op.getEffectiveInputLayout();
  auto targetLayout = op.getTargetLayoutAttr();

  auto inputSlice = dyn_cast<xegpu::SliceAttr>(inputLayout);
  if (!inputSlice)
    return rewriter.notifyMatchFailure(op, "input_layout must be #xegpu.slice");
  if (!isa<xegpu::LayoutAttr>(targetLayout))
    return rewriter.notifyMatchFailure(op,
                                       "target_layout must be #xegpu.layout");
  if (!isRank2UnitLaneDataRowMajor(inputLayout) ||
      !isRank2UnitLaneDataRowMajor(targetLayout))
    return rewriter.notifyMatchFailure(
        op, "both layouts must be rank 2 with effective lane_data [1, 1] and "
            "effective order [1, 0]");

  auto vectorType = cast<VectorType>(op.getType());
  Type elementType = vectorType.getElementType();
  if (!elementType.isIntOrFloat() ||
      !llvm::is_contained({8u, 16u, 32u, 64u},
                          elementType.getIntOrFloatBitWidth()))
    return rewriter.notifyMatchFailure(
        op, "element type must be an int or float of bit width 8, 16, 32 or 64 "
            "to be carried by gpu.shuffle");

  FailureOr<VectorType> distributedInput =
      xegpu::getDistributedVectorType(vectorType, inputLayout);
  FailureOr<VectorType> distributedTarget =
      xegpu::getDistributedVectorType(vectorType, targetLayout);
  if (failed(distributedInput) || failed(distributedTarget))
    return rewriter.notifyMatchFailure(
        op, "value type must be distributable by both layouts");
  if (distributedInput->getShape() !=
      ArrayRef<int64_t>{vectorType.getDimSize(0), 1})
    return rewriter.notifyMatchFailure(
        op, "distributed input_layout type must be vector<shape[0]x1>");
  if (distributedTarget->getShape() != ArrayRef<int64_t>{1, 2})
    return rewriter.notifyMatchFailure(
        op, "distributed target_layout type must be vector<1x2>");

  FailureOr<int64_t> stride = getDistributedDimLaneStride(inputSlice);
  if (failed(stride))
    return rewriter.notifyMatchFailure(
        op, "input_layout parent must have exactly one non-sliced dimension "
            "with effective lane_layout extent greater than one");

  SmallVector<int64_t> inputLaneLayout =
      inputLayout.getEffectiveLaneLayoutAsInt();
  SmallVector<int64_t> targetLaneLayout =
      targetLayout.getEffectiveLaneLayoutAsInt();
  if (*stride * inputLaneLayout[1] != subgroupSize)
    return rewriter.notifyMatchFailure(
        op, "input_layout distributed dimension lane stride times its extent "
            "must equal the subgroup size");
  if (targetLaneLayout[0] > *stride)
    return rewriter.notifyMatchFailure(
        op, "target_layout effective lane_layout[0] must not exceed the "
            "input_layout distributed dimension lane stride");
  if (inputLaneLayout != SmallVector<int64_t>{1, 2})
    return rewriter.notifyMatchFailure(
        op, "input_layout effective lane_layout must be [1, 2]");
  if (targetLaneLayout[0] <= 1 || targetLaneLayout[1] != 1)
    return rewriter.notifyMatchFailure(
        op, "target_layout effective lane_layout must be [n, 1] with n > 1");

  targetLaneCount = targetLaneLayout[0];
  distributedDimLaneStride = *stride;
  return success();
}
