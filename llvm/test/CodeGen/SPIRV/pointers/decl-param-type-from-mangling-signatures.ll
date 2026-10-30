; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Pointee types come from the parsed parameter types, so templates, nested
; signatures and substitutions in the mangled name do not prevent inference
; for the other parameters. A class named like an OpenCL type keeps the byte
; pointer fallback, also through template arguments and substitutions.

; CHECK-DAG: OpName %[[#TmplInt:]] "_Z4takeIiEvPT_"
; CHECK-DAG: OpName %[[#TmplHalf:]] "_Z4takeI4halfEvPT_"
; CHECK-DAG: OpName %[[#FnPtr:]] "_Z4takePFiiEPi"
; CHECK-DAG: OpName %[[#TmplArg:]] "_Z4takeP4PairIifEPf"
; CHECK-DAG: OpName %[[#NsSubst:]] "_ZN2ns4takeEPiS0_"
; CHECK-DAG: OpName %[[#VecSubst:]] "_Z4takePDv4_jPS_"
; CHECK-DAG: OpName %[[#ClassSubst:]] "_Z4takeP4halfS0_"

; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#V4Int:]] = OpTypeVector %[[#Int]] 4
; CHECK-DAG: %[[#CharPtr:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-DAG: %[[#IntPtr:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#FloatPtr:]] = OpTypePointer CrossWorkgroup %[[#Float]]
; CHECK-DAG: %[[#V4IntPtr:]] = OpTypePointer CrossWorkgroup %[[#V4Int]]
; CHECK-DAG: %[[#IntFn:]] = OpTypeFunction %[[#Void]] %[[#IntPtr]]{{$}}
; CHECK-DAG: %[[#CharFn:]] = OpTypeFunction %[[#Void]] %[[#CharPtr]]{{$}}
; CHECK-DAG: %[[#FnPtrFn:]] = OpTypeFunction %[[#Void]] %[[#CharPtr]] %[[#IntPtr]]{{$}}
; CHECK-DAG: %[[#TmplArgFn:]] = OpTypeFunction %[[#Void]] %[[#CharPtr]] %[[#FloatPtr]]{{$}}
; CHECK-DAG: %[[#IntIntFn:]] = OpTypeFunction %[[#Void]] %[[#IntPtr]] %[[#IntPtr]]{{$}}
; CHECK-DAG: %[[#VecVecFn:]] = OpTypeFunction %[[#Void]] %[[#V4IntPtr]] %[[#V4IntPtr]]{{$}}
; CHECK-DAG: %[[#CharCharFn:]] = OpTypeFunction %[[#Void]] %[[#CharPtr]] %[[#CharPtr]]{{$}}

; CHECK: %[[#TmplInt]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK: %[[#TmplHalf]] = OpFunction %[[#Void]] None %[[#CharFn]]
; CHECK: %[[#FnPtr]] = OpFunction %[[#Void]] None %[[#FnPtrFn]]
; CHECK: %[[#TmplArg]] = OpFunction %[[#Void]] None %[[#TmplArgFn]]
; CHECK: %[[#NsSubst]] = OpFunction %[[#Void]] None %[[#IntIntFn]]
; CHECK: %[[#VecSubst]] = OpFunction %[[#Void]] None %[[#VecVecFn]]
; CHECK: %[[#ClassSubst]] = OpFunction %[[#Void]] None %[[#CharCharFn]]

; void take<int>(int *)
declare spir_func void @_Z4takeIiEvPT_(ptr addrspace(1))
; void take<half>(half *), where half is a class
declare spir_func void @_Z4takeI4halfEvPT_(ptr addrspace(1))
; take(int (*)(int), int *)
declare spir_func void @_Z4takePFiiEPi(ptr addrspace(1), ptr addrspace(1))
; take(Pair<int, float> *, float *)
declare spir_func void @_Z4takeP4PairIifEPf(ptr addrspace(1), ptr addrspace(1))
; ns::take(int *, int *)
declare spir_func void @_ZN2ns4takeEPiS0_(ptr addrspace(1), ptr addrspace(1))
; take(uint4 *, uint4 *)
declare spir_func void @_Z4takePDv4_jPS_(ptr addrspace(1), ptr addrspace(1))
; take(half *, half *), where half is a class
declare spir_func void @_Z4takeP4halfS0_(ptr addrspace(1), ptr addrspace(1))

define spir_kernel void @test(ptr addrspace(1) %p) {
  call spir_func void @_Z4takeIiEvPT_(ptr addrspace(1) %p)
  call spir_func void @_Z4takeI4halfEvPT_(ptr addrspace(1) %p)
  call spir_func void @_Z4takePFiiEPi(ptr addrspace(1) %p, ptr addrspace(1) %p)
  call spir_func void @_Z4takeP4PairIifEPf(ptr addrspace(1) %p, ptr addrspace(1) %p)
  call spir_func void @_ZN2ns4takeEPiS0_(ptr addrspace(1) %p, ptr addrspace(1) %p)
  call spir_func void @_Z4takePDv4_jPS_(ptr addrspace(1) %p, ptr addrspace(1) %p)
  call spir_func void @_Z4takeP4halfS0_(ptr addrspace(1) %p, ptr addrspace(1) %p)
  ret void
}
