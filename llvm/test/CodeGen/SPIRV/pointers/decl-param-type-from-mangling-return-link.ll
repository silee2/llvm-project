; RUN: split-file %s %t
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %t/caller.ll -o - | FileCheck %s --check-prefix=CALLER
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %t/caller.ll -o - | FileCheck %s --check-prefix=CALLER
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %t/callee.ll -o - | FileCheck %s --check-prefix=CALLEE
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %t/callee.ll -o - | FileCheck %s --check-prefix=CALLEE
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %t/caller.ll -filetype=obj -o %t/caller32.spv %}
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %t/callee.ll -filetype=obj -o %t/callee32.spv %}
; RUN: %if spirv-tools %{ spirv-val %t/caller32.spv %}
; RUN: %if spirv-tools %{ spirv-val %t/callee32.spv %}
; RUN: %if spirv-tools %{ spirv-link %t/caller32.spv %t/callee32.spv -o %t/linked32.spv %}
; RUN: %if spirv-tools %{ spirv-val %t/linked32.spv %}
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %t/caller.ll -filetype=obj -o %t/caller64.spv %}
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %t/callee.ll -filetype=obj -o %t/callee64.spv %}
; RUN: %if spirv-tools %{ spirv-val %t/caller64.spv %}
; RUN: %if spirv-tools %{ spirv-val %t/callee64.spv %}
; RUN: %if spirv-tools %{ spirv-link %t/caller64.spv %t/callee64.spv -o %t/linked64.spv %}
; RUN: %if spirv-tools %{ spirv-val %t/linked64.spv %}

; The caller cannot see the producer's definition. The accepted int* formal of
; foo must weakly infer get_ptr's generic int-pointer return type, so both
; imported signatures match the independently compiled definitions at link time.
; The calls before and after the hinted call must use that same return type.
;
; CALLER-DAG: OpName %[[#Get:]] "get_ptr"
; CALLER-DAG: OpName %[[#Foo:]] "_Z3fooPi"
; CALLER-DAG: OpName %[[#Test:]] "test"
; CALLER-DAG: OpDecorate %[[#Get]] LinkageAttributes "get_ptr" Import
; CALLER-DAG: OpDecorate %[[#Foo]] LinkageAttributes "_Z3fooPi" Import
; CALLER-DAG: %[[#Void:]] = OpTypeVoid
; CALLER-DAG: %[[#Int:]] = OpTypeInt 32 0
; CALLER-DAG: %[[#Ptr:]] = OpTypePointer Generic %[[#Int]]
; CALLER-DAG: %[[#GetTy:]] = OpTypeFunction %[[#Ptr]]{{$}}
; CALLER-DAG: %[[#FooTy:]] = OpTypeFunction %[[#Void]] %[[#Ptr]]{{$}}
; CALLER: %[[#Get]] = OpFunction %[[#Ptr]] None %[[#GetTy]]
; CALLER: %[[#Foo]] = OpFunction %[[#Void]] None %[[#FooTy]]
; CALLER: OpFunctionParameter %[[#Ptr]]
; CALLER: %[[#Test]] = OpFunction
; CALLER: OpFunctionCall %[[#Ptr]] %[[#Get]]{{$}}
; CALLER: %[[#P:]] = OpFunctionCall %[[#Ptr]] %[[#Get]]{{$}}
; CALLER-NOT: OpBitcast
; CALLER: OpFunctionCall %[[#Void]] %[[#Foo]] %[[#P]]{{$}}
; CALLER: OpFunctionCall %[[#Ptr]] %[[#Get]]{{$}}
; CALLER: OpFunctionEnd
;
; CALLEE-DAG: OpName %[[#Get:]] "get_ptr"
; CALLEE-DAG: OpName %[[#Foo:]] "_Z3fooPi"
; CALLEE-DAG: OpDecorate %[[#Get]] LinkageAttributes "get_ptr" Export
; CALLEE-DAG: OpDecorate %[[#Foo]] LinkageAttributes "_Z3fooPi" Export
; CALLEE-DAG: %[[#Void:]] = OpTypeVoid
; CALLEE-DAG: %[[#Int:]] = OpTypeInt 32 0
; CALLEE-DAG: %[[#GlobalPtr:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CALLEE-DAG: %[[#Ptr:]] = OpTypePointer Generic %[[#Int]]
; CALLEE-DAG: %[[#GetTy:]] = OpTypeFunction %[[#Ptr]]{{$}}
; CALLEE-DAG: %[[#FooTy:]] = OpTypeFunction %[[#Void]] %[[#Ptr]]{{$}}
; CALLEE-DAG: %[[#Storage:]] = OpVariable %[[#GlobalPtr]] CrossWorkgroup
; CALLEE: %[[#Get]] = OpFunction %[[#Ptr]] None %[[#GetTy]]
; CALLEE: %[[#P:]] = OpPtrCastToGeneric %[[#Ptr]] %[[#Storage]]{{$}}
; CALLEE: OpReturnValue %[[#P]]{{$}}
; CALLEE: OpFunctionEnd
; CALLEE: %[[#Foo]] = OpFunction %[[#Void]] None %[[#FooTy]]
; CALLEE: %[[#Arg:]] = OpFunctionParameter %[[#Ptr]]
; CALLEE: OpStore %[[#Arg]] %[[#]]
; CALLEE: OpFunctionEnd

;--- caller.ll
declare spir_func ptr addrspace(4) @get_ptr()
declare spir_func void @_Z3fooPi(ptr addrspace(4))

define spir_kernel void @test() {
  %before = call spir_func ptr addrspace(4) @get_ptr()
  %p = call spir_func ptr addrspace(4) @get_ptr()
  call spir_func void @_Z3fooPi(ptr addrspace(4) %p)
  %after = call spir_func ptr addrspace(4) @get_ptr()
  ret void
}

;--- callee.ll
@storage = addrspace(1) global i32 0, align 4

define spir_func ptr addrspace(4) @get_ptr() {
  %p = addrspacecast ptr addrspace(1) @storage to ptr addrspace(4)
  ret ptr addrspace(4) %p
}

define spir_func void @_Z3fooPi(ptr addrspace(4) %p) {
  store i32 42, ptr addrspace(4) %p, align 4
  ret void
}
