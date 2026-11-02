; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o /dev/null
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o /dev/null
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Parameter-pack expansions must conservatively disable inference for the
; entire declaration, even when the unexpanded AST parameter count matches IR.
; In particular, the empty member pack leaves two AST parameters (int* and the
; expansion) and two IR arguments (this and int*). Neither IR argument may be
; inferred as int*. The free functions also retain the byte fallback for the
; ordinary int* parameter, not just the expanded parameters. A bare T_ that
; resolves to a pack must also disable inference, even without Dp. An expansion
; nested inside a function-pointer type must prevent inference for a later
; ordinary int* parameter. An unused empty template-argument pack, however,
; must not disable inference when there is no pack in the parameter types.
;
; Top-level cv qualifiers on a template pointer parameter do not change its
; pointee. Template reference substitution may leave nested reference nodes:
; collapse them to find the scalar, but stop at an ultimate pointer. Neither
; pointers-to-pointers nor references-to-pointers may become pointers-to-int.
; Every declaration is called with null, so no load, store, allocation or
; caller parameter can supply the pointee type independently of the mangling.

; CHECK-DAG: OpName %[[#MemberEmpty:]] "_ZN1S6memberIJEEEvPiDpT_"
; CHECK-DAG: OpName %[[#FreeEmpty:]] "_Z4packIJEEvPiDpT_"
; CHECK-DAG: OpName %[[#FreeNonempty:]] "_Z4packIJPiEEvPiDpT_"
; CHECK-DAG: OpName %[[#BarePack:]] "_Z4packIJPiEEvT_Pi"
; CHECK-DAG: OpName %[[#NestedPack:]] "_Z4packIJPiEEvPFvDpT_EPi"
; CHECK-DAG: OpName %[[#UnusedPack:]] "_Z4packIJEEvPi"
; CHECK-DAG: OpName %[[#ConstPtr:]] "_Z4takeIKPiEvT_"
; CHECK-DAG: OpName %[[#VolatilePtr:]] "_Z4takeIVPiEvT_"
; CHECK-DAG: OpName %[[#CVPtr:]] "_Z4takeIVKPiEvT_"
; CHECK-DAG: OpName %[[#CVPointee:]] "_Z4takeIKPVKiEvT_"
; CHECK-DAG: OpName %[[#LRefCollapse:]] "_Z3refIRiEvOT_"
; CHECK-DAG: OpName %[[#RRefCollapse:]] "_Z3refIOiEvOT_"
; CHECK-DAG: OpName %[[#ConstRefCollapse:]] "_Z3refIRKiEvOT_"
; CHECK-DAG: OpName %[[#LRefLRefCollapse:]] "_Z3refIRiEvRT_"
; CHECK-DAG: OpName %[[#LRef:]] "_Z3refRi"
; CHECK-DAG: OpName %[[#RRef:]] "_Z3refOi"
; CHECK-DAG: OpName %[[#PtrPtr:]] "_Z4takeIPPiEvT_"
; CHECK-DAG: OpName %[[#ConstPtrPtr:]] "_Z4takeIKPPiEvT_"
; CHECK-DAG: OpName %[[#RefPtr:]] "_Z3refRPi"
; CHECK-DAG: OpName %[[#RefConstPtr:]] "_Z3refOKPi"
; CHECK-DAG: OpName %[[#LRefPtrCollapse:]] "_Z3refIRPiEvOT_"
; CHECK-DAG: OpName %[[#RRefPtrCollapse:]] "_Z3refIOPiEvOT_"

; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#Byte:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#BytePtr:]] = OpTypePointer CrossWorkgroup %[[#Byte]]
; CHECK-DAG: %[[#IntPtr:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#ByteFn:]] = OpTypeFunction %[[#Void]] %[[#BytePtr]]{{$}}
; CHECK-DAG: %[[#ByteByteFn:]] = OpTypeFunction %[[#Void]] %[[#BytePtr]] %[[#BytePtr]]{{$}}
; CHECK-DAG: %[[#IntFn:]] = OpTypeFunction %[[#Void]] %[[#IntPtr]]{{$}}

; CHECK: %[[#MemberEmpty]] = OpFunction %[[#Void]] None %[[#ByteByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#FreeEmpty]] = OpFunction %[[#Void]] None %[[#ByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#FreeNonempty]] = OpFunction %[[#Void]] None %[[#ByteByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd

; CHECK: %[[#BarePack]] = OpFunction %[[#Void]] None %[[#ByteByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#NestedPack]] = OpFunction %[[#Void]] None %[[#ByteByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#UnusedPack]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd

; CHECK: %[[#ConstPtr]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#VolatilePtr]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#CVPtr]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#CVPointee]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd

; CHECK: %[[#LRefCollapse]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#RRefCollapse]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#ConstRefCollapse]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#LRefLRefCollapse]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#LRef]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#RRef]] = OpFunction %[[#Void]] None %[[#IntFn]]
; CHECK-NEXT: OpFunctionParameter %[[#IntPtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd

; CHECK: %[[#PtrPtr]] = OpFunction %[[#Void]] None %[[#ByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#ConstPtrPtr]] = OpFunction %[[#Void]] None %[[#ByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#RefPtr]] = OpFunction %[[#Void]] None %[[#ByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#RefConstPtr]] = OpFunction %[[#Void]] None %[[#ByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#LRefPtrCollapse]] = OpFunction %[[#Void]] None %[[#ByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#RRefPtrCollapse]] = OpFunction %[[#Void]] None %[[#ByteFn]]
; CHECK-NEXT: OpFunctionParameter %[[#BytePtr]]{{$}}
; CHECK-NEXT: OpFunctionEnd

; void S::member<>(int *): the first IR argument is this.
declare spir_func void @_ZN1S6memberIJEEEvPiDpT_(ptr addrspace(1), ptr addrspace(1))
; void pack<>(int *)
declare spir_func void @_Z4packIJEEvPiDpT_(ptr addrspace(1))
; void pack<int *>(int *, int *)
declare spir_func void @_Z4packIJPiEEvPiDpT_(ptr addrspace(1), ptr addrspace(1))
; Parser-accepted bare pack reference T_ without an expansion wrapper Dp.
; Both AST and IR have two parameters; the later int* must retain fallback.
declare spir_func void @_Z4packIJPiEEvT_Pi(ptr addrspace(1), ptr addrspace(1))
; void pack<int *>(void (*)(int *), int *): expansion is inside the function type.
declare spir_func void @_Z4packIJPiEEvPFvDpT_EPi(ptr addrspace(1), ptr addrspace(1))
; void pack<>(int *): no parameter-pack expansion despite the empty pack argument.
declare spir_func void @_Z4packIJEEvPi(ptr addrspace(1))

; void take<int *const>(int *const), and volatile/cv variants.
declare spir_func void @_Z4takeIKPiEvT_(ptr addrspace(1))
declare spir_func void @_Z4takeIVPiEvT_(ptr addrspace(1))
declare spir_func void @_Z4takeIVKPiEvT_(ptr addrspace(1))
; void take<int const volatile *const>(int const volatile *const)
declare spir_func void @_Z4takeIKPVKiEvT_(ptr addrspace(1))

; void ref<int &>(int &), with the parameter encoded as T&&.
declare spir_func void @_Z3refIRiEvOT_(ptr addrspace(1))
; void ref<int &&>(int &&), with the parameter encoded as T&&.
declare spir_func void @_Z3refIOiEvOT_(ptr addrspace(1))
declare spir_func void @_Z3refIRKiEvOT_(ptr addrspace(1))
; void ref<int &>(int &), with the parameter encoded as T&.
declare spir_func void @_Z3refIRiEvRT_(ptr addrspace(1))
declare spir_func void @_Z3refRi(ptr addrspace(1))
declare spir_func void @_Z3refOi(ptr addrspace(1))

; Unsupported pointees must retain byte pointers, including after collapsing.
declare spir_func void @_Z4takeIPPiEvT_(ptr addrspace(1))
declare spir_func void @_Z4takeIKPPiEvT_(ptr addrspace(1))
declare spir_func void @_Z3refRPi(ptr addrspace(1))
declare spir_func void @_Z3refOKPi(ptr addrspace(1))
declare spir_func void @_Z3refIRPiEvOT_(ptr addrspace(1))
declare spir_func void @_Z3refIOPiEvOT_(ptr addrspace(1))

define spir_kernel void @test() {
  call spir_func void @_ZN1S6memberIJEEEvPiDpT_(ptr addrspace(1) null, ptr addrspace(1) null)
  call spir_func void @_Z4packIJEEvPiDpT_(ptr addrspace(1) null)
  call spir_func void @_Z4packIJPiEEvPiDpT_(ptr addrspace(1) null, ptr addrspace(1) null)
  call spir_func void @_Z4packIJPiEEvT_Pi(ptr addrspace(1) null, ptr addrspace(1) null)
  call spir_func void @_Z4packIJPiEEvPFvDpT_EPi(ptr addrspace(1) null, ptr addrspace(1) null)
  call spir_func void @_Z4packIJEEvPi(ptr addrspace(1) null)
  call spir_func void @_Z4takeIKPiEvT_(ptr addrspace(1) null)
  call spir_func void @_Z4takeIVPiEvT_(ptr addrspace(1) null)
  call spir_func void @_Z4takeIVKPiEvT_(ptr addrspace(1) null)
  call spir_func void @_Z4takeIKPVKiEvT_(ptr addrspace(1) null)
  call spir_func void @_Z3refIRiEvOT_(ptr addrspace(1) null)
  call spir_func void @_Z3refIOiEvOT_(ptr addrspace(1) null)
  call spir_func void @_Z3refIRKiEvOT_(ptr addrspace(1) null)
  call spir_func void @_Z3refIRiEvRT_(ptr addrspace(1) null)
  call spir_func void @_Z3refRi(ptr addrspace(1) null)
  call spir_func void @_Z3refOi(ptr addrspace(1) null)
  call spir_func void @_Z4takeIPPiEvT_(ptr addrspace(1) null)
  call spir_func void @_Z4takeIKPPiEvT_(ptr addrspace(1) null)
  call spir_func void @_Z3refRPi(ptr addrspace(1) null)
  call spir_func void @_Z3refOKPi(ptr addrspace(1) null)
  call spir_func void @_Z3refIRPiEvOT_(ptr addrspace(1) null)
  call spir_func void @_Z3refIOPiEvOT_(ptr addrspace(1) null)
  ret void
}
