; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK,UNTYPED
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK,UNTYPED
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ sed -e 's/i32/i64/g' -e '/async_work_group_copy/s/, ptr null)/, i64 1, ptr null)/; /async_work_group_copy/s/, i64, ptr)/, i64, i64, ptr)/' -e 's/_Z21async_work_group_copy/_Z29async_work_group_strided_copy/g; s/ij9ocl_event/imm9ocl_event/g; s/S_j9ocl_event/S_mm9ocl_event/g; s/ijj9ocl_event/imm9ocl_event/g; s/S_jj9ocl_event/S_mm9ocl_event/g' %s | llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
;
; Builtin declarations do not provide ordinary function parameter types. Null
; and undef copy operands must still get their pointee types from the mangling,
; rather than falling back to byte-sized elements. Each address space's null
; and undef is shared by the int and uint4 overloads: adapt individual uses,
; without changing the shared constant's type for all calls.
;
; Keep the existing i32 count/stride ABI in the ordinary-copy codegen checks.
; Typed Physical64 validation needs i64 counts and strides. Use the equivalent
; strided overload with an explicit unit stride there, since ordinary copies
; currently synthesize an i32 unit stride even with an i64 count.
;
; CHECK-DAG: OpName %[[#Test:]] "test"
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Event:]] = OpTypeEvent
; TYPED-DAG: %[[#Byte:]] = OpTypeInt 8 0
; TYPED-DAG: %[[#Vec:]] = OpTypeVector %[[#Int]] 4
; TYPED-DAG: %[[#GlobalBytePtr:]] = OpTypePointer CrossWorkgroup %[[#Byte]]
; TYPED-DAG: %[[#LocalBytePtr:]] = OpTypePointer Workgroup %[[#Byte]]
; TYPED-DAG: %[[#GlobalIntPtr:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; TYPED-DAG: %[[#LocalIntPtr:]] = OpTypePointer Workgroup %[[#Int]]
; TYPED-DAG: %[[#GlobalVecPtr:]] = OpTypePointer CrossWorkgroup %[[#Vec]]
; TYPED-DAG: %[[#LocalVecPtr:]] = OpTypePointer Workgroup %[[#Vec]]
; UNTYPED-DAG: OpCapability UntypedPointersKHR
; UNTYPED-DAG: OpExtension "SPV_KHR_untyped_pointers"
; UNTYPED-DAG: %[[#GlobalBytePtr:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; UNTYPED-DAG: %[[#LocalBytePtr:]] = OpTypeUntypedPointerKHR Workgroup
; UNTYPED-DAG: %[[#IntBytes:]] = OpConstant %[[#Int]] 4
; UNTYPED-DAG: %[[#VecBytes:]] = OpConstant %[[#Int]] 16
; CHECK-DAG: %[[#GlobalNull:]] = OpConstantNull %[[#GlobalBytePtr]]
; CHECK-DAG: %[[#LocalNull:]] = OpConstantNull %[[#LocalBytePtr]]
; CHECK-DAG: %[[#GlobalUndef:]] = OpUndef %[[#GlobalBytePtr]]
; CHECK-DAG: %[[#LocalUndef:]] = OpUndef %[[#LocalBytePtr]]
; CHECK-DAG: %[[#Scope:]] = OpConstant %[[#Int]] 2
; CHECK-DAG: %[[#Count:]] = OpConstant %[[#Int]] 7
; CHECK-DAG: %[[#One:]] = OpConstant %[[#Int]] 1
; CHECK-DAG: %[[#Stride:]] = OpConstant %[[#Int]] 3
; CHECK-DAG: %[[#NullEvent:]] = OpConstantNull %[[#Event]]
;
; CHECK: %[[#Test]] = OpFunction
;
; TYPED-DAG: %[[#GlobalIntNull:]] = OpBitcast %[[#GlobalIntPtr]] %[[#GlobalNull]]
; TYPED-DAG: %[[#LocalIntNull:]] = OpBitcast %[[#LocalIntPtr]] %[[#LocalNull]]
; TYPED: OpGroupAsyncCopy %[[#Event]] %[[#Scope]] %[[#GlobalIntNull]] %[[#LocalIntNull]] %[[#Count]] %[[#One]] %[[#NullEvent]]
; UNTYPED: OpUntypedGroupAsyncCopyKHR %[[#Event]] %[[#Scope]] %[[#GlobalNull]] %[[#LocalNull]] %[[#IntBytes]] %[[#Count]] %[[#One]] %[[#NullEvent]]
;
; TYPED-DAG: %[[#GlobalVecNull:]] = OpBitcast %[[#GlobalVecPtr]] %[[#GlobalNull]]
; TYPED-DAG: %[[#LocalVecNull:]] = OpBitcast %[[#LocalVecPtr]] %[[#LocalNull]]
; TYPED: OpGroupAsyncCopy %[[#Event]] %[[#Scope]] %[[#GlobalVecNull]] %[[#LocalVecNull]] %[[#Count]] %[[#One]] %[[#NullEvent]]
; UNTYPED: OpUntypedGroupAsyncCopyKHR %[[#Event]] %[[#Scope]] %[[#GlobalNull]] %[[#LocalNull]] %[[#VecBytes]] %[[#Count]] %[[#One]] %[[#NullEvent]]
;
; TYPED-DAG: %[[#StridedLocalIntNull:]] = OpBitcast %[[#LocalIntPtr]] %[[#LocalNull]]
; TYPED-DAG: %[[#StridedGlobalIntNull:]] = OpBitcast %[[#GlobalIntPtr]] %[[#GlobalNull]]
; TYPED: OpGroupAsyncCopy %[[#Event]] %[[#Scope]] %[[#StridedLocalIntNull]] %[[#StridedGlobalIntNull]] %[[#Count]] %[[#Stride]] %[[#NullEvent]]
; UNTYPED: OpUntypedGroupAsyncCopyKHR %[[#Event]] %[[#Scope]] %[[#LocalNull]] %[[#GlobalNull]] %[[#IntBytes]] %[[#Count]] %[[#Stride]] %[[#NullEvent]]
;
; TYPED-DAG: %[[#StridedLocalVecNull:]] = OpBitcast %[[#LocalVecPtr]] %[[#LocalNull]]
; TYPED-DAG: %[[#StridedGlobalVecNull:]] = OpBitcast %[[#GlobalVecPtr]] %[[#GlobalNull]]
; TYPED: OpGroupAsyncCopy %[[#Event]] %[[#Scope]] %[[#StridedLocalVecNull]] %[[#StridedGlobalVecNull]] %[[#Count]] %[[#Stride]] %[[#NullEvent]]
; UNTYPED: OpUntypedGroupAsyncCopyKHR %[[#Event]] %[[#Scope]] %[[#LocalNull]] %[[#GlobalNull]] %[[#VecBytes]] %[[#Count]] %[[#Stride]] %[[#NullEvent]]
;
; TYPED-DAG: %[[#GlobalIntUndef:]] = OpBitcast %[[#GlobalIntPtr]] %[[#GlobalUndef]]
; TYPED-DAG: %[[#LocalIntUndef:]] = OpBitcast %[[#LocalIntPtr]] %[[#LocalUndef]]
; TYPED: OpGroupAsyncCopy %[[#Event]] %[[#Scope]] %[[#GlobalIntUndef]] %[[#LocalIntUndef]] %[[#Count]] %[[#One]] %[[#NullEvent]]
; UNTYPED: OpUntypedGroupAsyncCopyKHR %[[#Event]] %[[#Scope]] %[[#GlobalUndef]] %[[#LocalUndef]] %[[#IntBytes]] %[[#Count]] %[[#One]] %[[#NullEvent]]
;
; TYPED-DAG: %[[#GlobalVecUndef:]] = OpBitcast %[[#GlobalVecPtr]] %[[#GlobalUndef]]
; TYPED-DAG: %[[#LocalVecUndef:]] = OpBitcast %[[#LocalVecPtr]] %[[#LocalUndef]]
; TYPED: OpGroupAsyncCopy %[[#Event]] %[[#Scope]] %[[#GlobalVecUndef]] %[[#LocalVecUndef]] %[[#Count]] %[[#One]] %[[#NullEvent]]
; UNTYPED: OpUntypedGroupAsyncCopyKHR %[[#Event]] %[[#Scope]] %[[#GlobalUndef]] %[[#LocalUndef]] %[[#VecBytes]] %[[#Count]] %[[#One]] %[[#NullEvent]]
;
; TYPED-DAG: %[[#StridedLocalIntUndef:]] = OpBitcast %[[#LocalIntPtr]] %[[#LocalUndef]]
; TYPED-DAG: %[[#StridedGlobalIntUndef:]] = OpBitcast %[[#GlobalIntPtr]] %[[#GlobalUndef]]
; TYPED: OpGroupAsyncCopy %[[#Event]] %[[#Scope]] %[[#StridedLocalIntUndef]] %[[#StridedGlobalIntUndef]] %[[#Count]] %[[#Stride]] %[[#NullEvent]]
; UNTYPED: OpUntypedGroupAsyncCopyKHR %[[#Event]] %[[#Scope]] %[[#LocalUndef]] %[[#GlobalUndef]] %[[#IntBytes]] %[[#Count]] %[[#Stride]] %[[#NullEvent]]
;
; TYPED-DAG: %[[#StridedLocalVecUndef:]] = OpBitcast %[[#LocalVecPtr]] %[[#LocalUndef]]
; TYPED-DAG: %[[#StridedGlobalVecUndef:]] = OpBitcast %[[#GlobalVecPtr]] %[[#GlobalUndef]]
; TYPED: OpGroupAsyncCopy %[[#Event]] %[[#Scope]] %[[#StridedLocalVecUndef]] %[[#StridedGlobalVecUndef]] %[[#Count]] %[[#Stride]] %[[#NullEvent]]
; UNTYPED: OpUntypedGroupAsyncCopyKHR %[[#Event]] %[[#Scope]] %[[#LocalUndef]] %[[#GlobalUndef]] %[[#VecBytes]] %[[#Count]] %[[#Stride]] %[[#NullEvent]]
; CHECK: OpFunctionEnd

define spir_kernel void @test() {
  %int.copy = call spir_func ptr @_Z21async_work_group_copyPU3AS1iPKU3AS3ij9ocl_event(ptr addrspace(1) null, ptr addrspace(3) null, i32 7, ptr null)
  %vec.copy = call spir_func ptr @_Z21async_work_group_copyPU3AS1Dv4_jPKU3AS3S_j9ocl_event(ptr addrspace(1) null, ptr addrspace(3) null, i32 7, ptr null)
  %int.strided = call spir_func ptr @_Z29async_work_group_strided_copyPU3AS3iPU3AS1Kijj9ocl_event(ptr addrspace(3) null, ptr addrspace(1) null, i32 7, i32 3, ptr null)
  %vec.strided = call spir_func ptr @_Z29async_work_group_strided_copyPU3AS3Dv4_jPKU3AS1S_jj9ocl_event(ptr addrspace(3) null, ptr addrspace(1) null, i32 7, i32 3, ptr null)
  %int.copy.undef = call spir_func ptr @_Z21async_work_group_copyPU3AS1iPKU3AS3ij9ocl_event(ptr addrspace(1) undef, ptr addrspace(3) undef, i32 7, ptr null)
  %vec.copy.undef = call spir_func ptr @_Z21async_work_group_copyPU3AS1Dv4_jPKU3AS3S_j9ocl_event(ptr addrspace(1) undef, ptr addrspace(3) undef, i32 7, ptr null)
  %int.strided.undef = call spir_func ptr @_Z29async_work_group_strided_copyPU3AS3iPU3AS1Kijj9ocl_event(ptr addrspace(3) undef, ptr addrspace(1) undef, i32 7, i32 3, ptr null)
  %vec.strided.undef = call spir_func ptr @_Z29async_work_group_strided_copyPU3AS3Dv4_jPKU3AS1S_jj9ocl_event(ptr addrspace(3) undef, ptr addrspace(1) undef, i32 7, i32 3, ptr null)
  ret void
}

declare spir_func ptr @_Z21async_work_group_copyPU3AS1iPKU3AS3ij9ocl_event(ptr addrspace(1), ptr addrspace(3), i32, ptr)
declare spir_func ptr @_Z21async_work_group_copyPU3AS1Dv4_jPKU3AS3S_j9ocl_event(ptr addrspace(1), ptr addrspace(3), i32, ptr)
declare spir_func ptr @_Z29async_work_group_strided_copyPU3AS3iPU3AS1Kijj9ocl_event(ptr addrspace(3), ptr addrspace(1), i32, i32, ptr)
declare spir_func ptr @_Z29async_work_group_strided_copyPU3AS3Dv4_jPKU3AS1S_jj9ocl_event(ptr addrspace(3), ptr addrspace(1), i32, i32, ptr)
