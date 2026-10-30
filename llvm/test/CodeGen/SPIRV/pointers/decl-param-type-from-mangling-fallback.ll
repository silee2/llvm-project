; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Do not infer scalar pointees from class names or map source parameters onto
; implicit this/sret arguments. The template comma also makes the naive split's
; count equal the IR argument count, despite this being a member function.
; long and unsigned long retain the fallback, as their width can follow the
; host. Vendor qualifiers other than address spaces also retain it.
; CHECK-DAG: OpName %[[#UInt:]] "_Z4takeP4uint"
; CHECK-DAG: OpName %[[#IntConst:]] "_Z4takeP8intconst"
; CHECK-DAG: OpName %[[#Atomic:]] "_Z4takeP10atomic_int"
; CHECK-DAG: OpName %[[#HalfClass:]] "_Z4takeP4half"
; CHECK-DAG: OpName %[[#Member:]] "_ZN1S4takeEPiPf"
; CHECK-DAG: OpName %[[#Template:]] "_ZN1S4pairEP4PairIifEPi"
; CHECK-DAG: OpName %[[#Long:]] "_Z4takePl"
; CHECK-DAG: OpName %[[#ULong:]] "_Z4takePm"
; CHECK-DAG: OpName %[[#ConstMember:]] "_ZNK1S4takeEPi"
; CHECK-DAG: OpName %[[#AtomicQual:]] "_Z4takePU7_Atomici"
; CHECK-DAG: OpName %[[#SRet:]] "_Z4takePiPf"
; CHECK-DAG: OpName %[[#ByVal:]] "_Z5byvalPi"
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#Int]]
; CHECK-DAG: %[[#CharPtr:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-DAG: %[[#StructPtr:]] = OpTypePointer CrossWorkgroup %[[#Struct]]
; CHECK-DAG: %[[#FallbackTy:]] = OpTypeFunction %[[#Void]] %[[#CharPtr]]
; CHECK-DAG: %[[#MemberTy:]] = OpTypeFunction %[[#Void]] %[[#CharPtr]] %[[#CharPtr]] %[[#CharPtr]]
; CHECK-DAG: %[[#ConstMemberTy:]] = OpTypeFunction %[[#Void]] %[[#CharPtr]] %[[#CharPtr]]{{$}}
; CHECK-DAG: %[[#SRetTy:]] = OpTypeFunction %[[#Void]] %[[#StructPtr]] %[[#CharPtr]] %[[#CharPtr]]
; CHECK-DAG: %[[#ByValTy:]] = OpTypeFunction %[[#Void]] %[[#StructPtr]]

; CHECK: %[[#UInt]] = OpFunction %[[#Void]] None %[[#FallbackTy]]
; CHECK: %[[#IntConst]] = OpFunction %[[#Void]] None %[[#FallbackTy]]
; CHECK: %[[#Atomic]] = OpFunction %[[#Void]] None %[[#FallbackTy]]
; CHECK: %[[#HalfClass]] = OpFunction %[[#Void]] None %[[#FallbackTy]]
; CHECK: %[[#Member]] = OpFunction %[[#Void]] None %[[#MemberTy]]
; CHECK: %[[#Template]] = OpFunction %[[#Void]] None %[[#MemberTy]]
; CHECK: %[[#Long]] = OpFunction %[[#Void]] None %[[#FallbackTy]]
; CHECK: %[[#ULong]] = OpFunction %[[#Void]] None %[[#FallbackTy]]
; CHECK: %[[#ConstMember]] = OpFunction %[[#Void]] None %[[#ConstMemberTy]]
; CHECK: %[[#AtomicQual]] = OpFunction %[[#Void]] None %[[#FallbackTy]]
; CHECK: %[[#SRet]] = OpFunction %[[#Void]] None %[[#SRetTy]]
; Explicit pointee type attributes take precedence over the mangled name.
; CHECK: %[[#ByVal]] = OpFunction %[[#Void]] None %[[#ByValTy]]

%S = type { i32 }
declare spir_func void @_Z4takeP4uint(ptr addrspace(1))
declare spir_func void @_Z4takeP8intconst(ptr addrspace(1))
declare spir_func void @_Z4takeP10atomic_int(ptr addrspace(1))
declare spir_func void @_Z4takeP4half(ptr addrspace(1))
declare spir_func void @_ZN1S4takeEPiPf(ptr addrspace(1), ptr addrspace(1), ptr addrspace(1))
declare spir_func void @_ZN1S4pairEP4PairIifEPi(ptr addrspace(1), ptr addrspace(1), ptr addrspace(1))
declare spir_func void @_Z4takePl(ptr addrspace(1))
declare spir_func void @_Z4takePm(ptr addrspace(1))
declare spir_func void @_ZNK1S4takeEPi(ptr addrspace(1), ptr addrspace(1))
declare spir_func void @_Z4takePU7_Atomici(ptr addrspace(1))
declare spir_func void @_Z4takePiPf(ptr addrspace(1) sret(%S), ptr addrspace(1), ptr addrspace(1))
declare spir_func void @_Z5byvalPi(ptr addrspace(1) byval(%S))

define spir_kernel void @test(ptr addrspace(1) %p) {
  call spir_func void @_Z4takeP4uint(ptr addrspace(1) %p)
  call spir_func void @_Z4takeP8intconst(ptr addrspace(1) %p)
  call spir_func void @_Z4takeP10atomic_int(ptr addrspace(1) %p)
  call spir_func void @_Z4takeP4half(ptr addrspace(1) %p)
  call spir_func void @_ZN1S4takeEPiPf(ptr addrspace(1) %p, ptr addrspace(1) %p, ptr addrspace(1) %p)
  call spir_func void @_ZN1S4pairEP4PairIifEPi(ptr addrspace(1) %p, ptr addrspace(1) %p, ptr addrspace(1) %p)
  call spir_func void @_Z4takePl(ptr addrspace(1) %p)
  call spir_func void @_Z4takePm(ptr addrspace(1) %p)
  call spir_func void @_ZNK1S4takeEPi(ptr addrspace(1) %p, ptr addrspace(1) %p)
  call spir_func void @_Z4takePU7_Atomici(ptr addrspace(1) %p)
  call spir_func void @_Z4takePiPf(ptr addrspace(1) sret(%S) %p, ptr addrspace(1) %p, ptr addrspace(1) %p)
  call spir_func void @_Z5byvalPi(ptr addrspace(1) byval(%S) %p)
  ret void
}
