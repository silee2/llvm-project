; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefix=UNTYPED
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -filetype=obj -o - | spirv-val %}
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -filetype=obj -o - | spirv-val %}

; An accepted mangled formal provides only a weak hint for an otherwise opaque
; external producer's result. Choose one canonical return type for all calls,
; not just the call whose result is passed to the mangled formal. Known return
; types take precedence (including a real i8 load, not an i8 fallback).
; Conflicting weak hints must fall back to i8, independently of their order.
;
; CHECK-DAG: OpName %[[#Foo:]] "_Z3fooPi"
; CHECK-DAG: OpName %[[#Bar:]] "_Z3barPf"
; CHECK-DAG: OpName %[[#Get:]] "get_ptr"
; CHECK-DAG: OpName %[[#All:]] "get_ptr_all"
; CHECK-DAG: OpName %[[#ConflictIF:]] "get_ptr_conflict_if"
; CHECK-DAG: OpName %[[#ConflictFI:]] "get_ptr_conflict_fi"
; CHECK-DAG: OpName %[[#Byte:]] "get_ptr_byte"
; CHECK-DAG: OpName %[[#Known:]] "get_ptr_float"
; CHECK-DAG: OpName %[[#Basic:]] "basic"
; CHECK-DAG: OpName %[[#Earlier:]] "unhinted_earlier"
; CHECK-DAG: OpName %[[#Hinted:]] "hinted"
; CHECK-DAG: OpName %[[#Later:]] "unhinted_later"
; CHECK-DAG: OpName %[[#KnownCaller:]] "known_definition"
; CHECK-DAG: OpName %[[#IntFirst:]] "int_first"
; CHECK-DAG: OpName %[[#FloatFirst:]] "float_first"
; CHECK-DAG: OpName %[[#ByteHint:]] "byte_hint"
; CHECK-DAG: OpName %[[#ByteLoad:]] "byte_load"
; CHECK-DAG: OpName %[[#StrongFloat:]] "get_ptr_strong_float"
; CHECK-DAG: OpName %[[#StrongFloatReverse:]] "get_ptr_strong_float_reverse"
; CHECK-DAG: OpName %[[#StrongFloatHint:]] "strong_float_hint"
; CHECK-DAG: OpName %[[#StrongFloatLoad:]] "strong_float_load"
; CHECK-DAG: OpName %[[#StrongFloatLoadFirst:]] "strong_float_load_first"
; CHECK-DAG: OpName %[[#StrongFloatHintLater:]] "strong_float_hint_later"
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#IntPtr:]] = OpTypePointer Generic %[[#Int]]
; CHECK-DAG: %[[#FloatPtr:]] = OpTypePointer Generic %[[#Float]]
; CHECK-DAG: %[[#CharPtr:]] = OpTypePointer Generic %[[#Char]]
; CHECK-DAG: %[[#IntRet:]] = OpTypeFunction %[[#IntPtr]]{{$}}
; CHECK-DAG: %[[#FloatRet:]] = OpTypeFunction %[[#FloatPtr]]{{$}}
; CHECK-DAG: %[[#CharRet:]] = OpTypeFunction %[[#CharPtr]]{{$}}
; CHECK-DAG: %[[#FooTy:]] = OpTypeFunction %[[#Void]] %[[#IntPtr]]{{$}}
; CHECK-DAG: %[[#BarTy:]] = OpTypeFunction %[[#Void]] %[[#FloatPtr]]{{$}}
; CHECK: %[[#Get]] = OpFunction %[[#IntPtr]] None %[[#IntRet]]
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#Foo]] = OpFunction %[[#Void]] None %[[#FooTy]]
; CHECK-NEXT: {{%[0-9]+}} = OpFunctionParameter %[[#IntPtr]]
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#All]] = OpFunction %[[#IntPtr]] None %[[#IntRet]]
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#ConflictIF]] = OpFunction %[[#CharPtr]] None %[[#CharRet]]
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#Bar]] = OpFunction %[[#Void]] None %[[#BarTy]]
; CHECK-NEXT: {{%[0-9]+}} = OpFunctionParameter %[[#FloatPtr]]
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#ConflictFI]] = OpFunction %[[#CharPtr]] None %[[#CharRet]]
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#Byte]] = OpFunction %[[#CharPtr]] None %[[#CharRet]]
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#StrongFloat]] = OpFunction %[[#FloatPtr]] None %[[#FloatRet]]
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#StrongFloatReverse]] = OpFunction %[[#FloatPtr]] None %[[#FloatRet]]
; CHECK-NEXT: OpFunctionEnd

declare spir_func void @_Z3fooPi(ptr addrspace(4))
declare spir_func void @_Z3barPf(ptr addrspace(4))
declare spir_func ptr addrspace(4) @get_ptr()
declare spir_func ptr addrspace(4) @get_ptr_all()
declare spir_func ptr addrspace(4) @get_ptr_conflict_if()
declare spir_func ptr addrspace(4) @get_ptr_conflict_fi()
declare spir_func ptr addrspace(4) @get_ptr_byte()
declare spir_func ptr addrspace(4) @get_ptr_strong_float()
declare spir_func ptr addrspace(4) @get_ptr_strong_float_reverse()

; The formal's int pointee must reach the producer's signature and call result.
; CHECK: %[[#Basic]] = OpFunction
; CHECK: %[[#P:]] = OpFunctionCall %[[#IntPtr]] %[[#Get]]{{$}}
; CHECK-NOT: OpBitcast
; CHECK: OpFunctionCall %[[#Void]] %[[#Foo]] %[[#P]]{{$}}
; CHECK: OpFunctionEnd
;
; UNTYPED-DAG: OpName %[[#UGet:]] "get_ptr"
; UNTYPED-DAG: OpName %[[#UFoo:]] "_Z3fooPi"
; UNTYPED-DAG: OpName %[[#UBasic:]] "basic"
; UNTYPED-DAG: OpName %[[#UByte:]] "get_ptr_byte"
; UNTYPED-DAG: OpName %[[#UByteLoad:]] "byte_load"
; UNTYPED-DAG: OpName %[[#UStrongFloat:]] "get_ptr_strong_float"
; UNTYPED-DAG: OpName %[[#UStrongFloatReverse:]] "get_ptr_strong_float_reverse"
; UNTYPED-DAG: OpName %[[#UStrongFloatHint:]] "strong_float_hint"
; UNTYPED-DAG: OpName %[[#UStrongFloatLoad:]] "strong_float_load"
; UNTYPED-DAG: OpName %[[#UStrongFloatLoadFirst:]] "strong_float_load_first"
; UNTYPED-DAG: OpName %[[#UStrongFloatHintLater:]] "strong_float_hint_later"
; UNTYPED-DAG: %[[#UVoid:]] = OpTypeVoid
; UNTYPED-DAG: %[[#UChar:]] = OpTypeInt 8 0
; UNTYPED-DAG: %[[#UFloat:]] = OpTypeFloat 32
; UNTYPED-DAG: %[[#UPtr:]] = OpTypeUntypedPointerKHR Generic
; UNTYPED-DAG: %[[#URet:]] = OpTypeFunction %[[#UPtr]]{{$}}
; UNTYPED: %[[#UGet]] = OpFunction %[[#UPtr]] None %[[#URet]]
; UNTYPED: %[[#UStrongFloat]] = OpFunction %[[#UPtr]] None %[[#URet]]
; UNTYPED-NEXT: OpFunctionEnd
; UNTYPED: %[[#UStrongFloatReverse]] = OpFunction %[[#UPtr]] None %[[#URet]]
; UNTYPED-NEXT: OpFunctionEnd
; UNTYPED: %[[#UBasic]] = OpFunction
; UNTYPED: %[[#UP:]] = OpFunctionCall %[[#UPtr]] %[[#UGet]]{{$}}
; UNTYPED-NOT: OpBitcast
; UNTYPED: OpFunctionCall %[[#UVoid]] %[[#UFoo]] %[[#UP]]{{$}}
; UNTYPED: OpFunctionEnd
define spir_kernel void @basic() {
  %p = call spir_func ptr addrspace(4) @get_ptr()
  call spir_func void @_Z3fooPi(ptr addrspace(4) %p)
  ret void
}

; Unused results have no hint. Both calls surrounding the hinted function must
; still agree with the producer's canonical int-pointer return type.
; CHECK: %[[#Earlier]] = OpFunction
; CHECK: OpFunctionCall %[[#IntPtr]] %[[#All]]{{$}}
; CHECK: OpFunctionEnd
define spir_kernel void @unhinted_earlier() {
  %p = call spir_func ptr addrspace(4) @get_ptr_all()
  ret void
}

; CHECK: %[[#Hinted]] = OpFunction
; CHECK: %[[#H:]] = OpFunctionCall %[[#IntPtr]] %[[#All]]{{$}}
; CHECK-NOT: OpBitcast
; CHECK: OpFunctionCall %[[#Void]] %[[#Foo]] %[[#H]]{{$}}
; CHECK: OpFunctionEnd
define spir_kernel void @hinted() {
  %p = call spir_func ptr addrspace(4) @get_ptr_all()
  call spir_func void @_Z3fooPi(ptr addrspace(4) %p)
  ret void
}

; CHECK: %[[#Later]] = OpFunction
; CHECK: OpFunctionCall %[[#IntPtr]] %[[#All]]{{$}}
; CHECK: OpFunctionEnd
define spir_kernel void @unhinted_later() {
  %p = call spir_func ptr addrspace(4) @get_ptr_all()
  ret void
}

; A definition later in the module supplies a strong float return type.
; CHECK: %[[#KnownCaller]] = OpFunction
; CHECK: %[[#F:]] = OpFunctionCall %[[#FloatPtr]] %[[#Known]]{{$}}
; CHECK: %[[#FI:]] = OpBitcast %[[#IntPtr]] %[[#F]]{{$}}
; CHECK: OpFunctionCall %[[#Void]] %[[#Foo]] %[[#FI]]{{$}}
; CHECK: OpFunctionEnd
define spir_kernel void @known_definition() {
  %p = call spir_func ptr addrspace(4) @get_ptr_float()
  call spir_func void @_Z3fooPi(ptr addrspace(4) %p)
  ret void
}

; CHECK: %[[#IntFirst]] = OpFunction
; CHECK: %[[#CI:]] = OpFunctionCall %[[#CharPtr]] %[[#ConflictIF]]{{$}}
; CHECK: %[[#CII:]] = OpBitcast %[[#IntPtr]] %[[#CI]]{{$}}
; CHECK: OpFunctionCall %[[#Void]] %[[#Foo]] %[[#CII]]{{$}}
; CHECK: %[[#CF:]] = OpFunctionCall %[[#CharPtr]] %[[#ConflictIF]]{{$}}
; CHECK: %[[#CFF:]] = OpBitcast %[[#FloatPtr]] %[[#CF]]{{$}}
; CHECK: OpFunctionCall %[[#Void]] %[[#Bar]] %[[#CFF]]{{$}}
; CHECK: OpFunctionEnd
define spir_kernel void @int_first() {
  %i = call spir_func ptr addrspace(4) @get_ptr_conflict_if()
  call spir_func void @_Z3fooPi(ptr addrspace(4) %i)
  %f = call spir_func ptr addrspace(4) @get_ptr_conflict_if()
  call spir_func void @_Z3barPf(ptr addrspace(4) %f)
  ret void
}

; Reverse the hint order for a separate producer, rather than letting the first
; producer's cached type mask order-dependent inference.
; CHECK: %[[#FloatFirst]] = OpFunction
; CHECK: %[[#RCF:]] = OpFunctionCall %[[#CharPtr]] %[[#ConflictFI]]{{$}}
; CHECK: %[[#RCFF:]] = OpBitcast %[[#FloatPtr]] %[[#RCF]]{{$}}
; CHECK: OpFunctionCall %[[#Void]] %[[#Bar]] %[[#RCFF]]{{$}}
; CHECK: %[[#RCI:]] = OpFunctionCall %[[#CharPtr]] %[[#ConflictFI]]{{$}}
; CHECK: %[[#RCII:]] = OpBitcast %[[#IntPtr]] %[[#RCI]]{{$}}
; CHECK: OpFunctionCall %[[#Void]] %[[#Foo]] %[[#RCII]]{{$}}
; CHECK: OpFunctionEnd
define spir_kernel void @float_first() {
  %f = call spir_func ptr addrspace(4) @get_ptr_conflict_fi()
  call spir_func void @_Z3barPf(ptr addrspace(4) %f)
  %i = call spir_func ptr addrspace(4) @get_ptr_conflict_fi()
  call spir_func void @_Z3fooPi(ptr addrspace(4) %i)
  ret void
}

; A strong result on another call wins even when its load is in a later function.
; The byte type is intentional: do not mistake known i8 for the opaque fallback.
; CHECK: %[[#ByteHint]] = OpFunction
; CHECK: %[[#B:]] = OpFunctionCall %[[#CharPtr]] %[[#Byte]]{{$}}
; CHECK: %[[#BI:]] = OpBitcast %[[#IntPtr]] %[[#B]]{{$}}
; CHECK: OpFunctionCall %[[#Void]] %[[#Foo]] %[[#BI]]{{$}}
; CHECK: OpFunctionEnd
define spir_kernel void @byte_hint() {
  %p = call spir_func ptr addrspace(4) @get_ptr_byte()
  call spir_func void @_Z3fooPi(ptr addrspace(4) %p)
  ret void
}

; CHECK: %[[#ByteLoad]] = OpFunction
; CHECK: %[[#BL:]] = OpFunctionCall %[[#CharPtr]] %[[#Byte]]{{$}}
; CHECK-NOT: OpBitcast
; CHECK: OpLoad %[[#Char]] %[[#BL]]
; CHECK: OpFunctionEnd
; UNTYPED: %[[#UByteLoad]] = OpFunction
; UNTYPED: %[[#UBL:]] = OpFunctionCall %[[#UPtr]] %[[#UByte]]{{$}}
; UNTYPED-NOT: OpBitcast
; UNTYPED: OpLoad %[[#UChar]] %[[#UBL]]
; UNTYPED: OpFunctionEnd
define spir_kernel void @byte_load() {
  %p = call spir_func ptr addrspace(4) @get_ptr_byte()
  %v = load volatile i8, ptr addrspace(4) %p, align 1
  ret void
}

; Unlike a known definition, an external producer gets its strong float type
; from memory evidence on another call. The later load must override the earlier
; weak int hint even if adapting the load initially requires a pointer cast.
; CHECK: %[[#StrongFloatHint]] = OpFunction
; CHECK-NOT: OpBitcast
; CHECK: %[[#SFH:]] = OpFunctionCall %[[#FloatPtr]] %[[#StrongFloat]]{{$}}
; CHECK-NEXT: %[[#SFHI:]] = OpBitcast %[[#IntPtr]] %[[#SFH]]{{$}}
; CHECK-NEXT: {{%[0-9]+}} = OpFunctionCall %[[#Void]] %[[#Foo]] %[[#SFHI]]{{$}}
; CHECK-NOT: OpBitcast
; CHECK: OpFunctionEnd
; UNTYPED: %[[#UStrongFloatHint]] = OpFunction
; UNTYPED-NOT: OpBitcast
; UNTYPED: %[[#USFH:]] = OpFunctionCall %[[#UPtr]] %[[#UStrongFloat]]{{$}}
; UNTYPED-NEXT: {{%[0-9]+}} = OpFunctionCall %[[#UVoid]] %[[#UFoo]] %[[#USFH]]{{$}}
; UNTYPED-NOT: OpBitcast
; UNTYPED: OpFunctionEnd
define spir_kernel void @strong_float_hint() {
  %p = call spir_func ptr addrspace(4) @get_ptr_strong_float()
  call spir_func void @_Z3fooPi(ptr addrspace(4) %p)
  ret void
}

; CHECK: %[[#StrongFloatLoad]] = OpFunction
; CHECK-NOT: OpBitcast
; CHECK: %[[#SFL:]] = OpFunctionCall %[[#FloatPtr]] %[[#StrongFloat]]{{$}}
; CHECK-NEXT: {{%[0-9]+}} = OpLoad %[[#Float]] %[[#SFL]] Volatile|Aligned 4{{$}}
; CHECK-NOT: OpBitcast
; CHECK: OpFunctionEnd
; UNTYPED: %[[#UStrongFloatLoad]] = OpFunction
; UNTYPED-NOT: OpBitcast
; UNTYPED: %[[#USFL:]] = OpFunctionCall %[[#UPtr]] %[[#UStrongFloat]]{{$}}
; UNTYPED-NEXT: {{%[0-9]+}} = OpLoad %[[#UFloat]] %[[#USFL]] Volatile|Aligned 4{{$}}
; UNTYPED-NOT: OpBitcast
; UNTYPED: OpFunctionEnd
define spir_kernel void @strong_float_load() {
  %p = call spir_func ptr addrspace(4) @get_ptr_strong_float()
  %v = load volatile float, ptr addrspace(4) %p, align 4
  ret void
}

; Reverse the function order for a distinct producer to avoid cached inference
; from the first producer masking order-dependent handling of memory evidence.
; CHECK: %[[#StrongFloatLoadFirst]] = OpFunction
; CHECK-NOT: OpBitcast
; CHECK: %[[#SFRL:]] = OpFunctionCall %[[#FloatPtr]] %[[#StrongFloatReverse]]{{$}}
; CHECK-NEXT: {{%[0-9]+}} = OpLoad %[[#Float]] %[[#SFRL]] Volatile|Aligned 4{{$}}
; CHECK-NOT: OpBitcast
; CHECK: OpFunctionEnd
; UNTYPED: %[[#UStrongFloatLoadFirst]] = OpFunction
; UNTYPED-NOT: OpBitcast
; UNTYPED: %[[#USFRL:]] = OpFunctionCall %[[#UPtr]] %[[#UStrongFloatReverse]]{{$}}
; UNTYPED-NEXT: {{%[0-9]+}} = OpLoad %[[#UFloat]] %[[#USFRL]] Volatile|Aligned 4{{$}}
; UNTYPED-NOT: OpBitcast
; UNTYPED: OpFunctionEnd
define spir_kernel void @strong_float_load_first() {
  %p = call spir_func ptr addrspace(4) @get_ptr_strong_float_reverse()
  %v = load volatile float, ptr addrspace(4) %p, align 4
  ret void
}

; CHECK: %[[#StrongFloatHintLater]] = OpFunction
; CHECK-NOT: OpBitcast
; CHECK: %[[#SFRH:]] = OpFunctionCall %[[#FloatPtr]] %[[#StrongFloatReverse]]{{$}}
; CHECK-NEXT: %[[#SFRHI:]] = OpBitcast %[[#IntPtr]] %[[#SFRH]]{{$}}
; CHECK-NEXT: {{%[0-9]+}} = OpFunctionCall %[[#Void]] %[[#Foo]] %[[#SFRHI]]{{$}}
; CHECK-NOT: OpBitcast
; CHECK: OpFunctionEnd
; UNTYPED: %[[#UStrongFloatHintLater]] = OpFunction
; UNTYPED-NOT: OpBitcast
; UNTYPED: %[[#USFRH:]] = OpFunctionCall %[[#UPtr]] %[[#UStrongFloatReverse]]{{$}}
; UNTYPED-NEXT: {{%[0-9]+}} = OpFunctionCall %[[#UVoid]] %[[#UFoo]] %[[#USFRH]]{{$}}
; UNTYPED-NOT: OpBitcast
; UNTYPED: OpFunctionEnd
define spir_kernel void @strong_float_hint_later() {
  %p = call spir_func ptr addrspace(4) @get_ptr_strong_float_reverse()
  call spir_func void @_Z3fooPi(ptr addrspace(4) %p)
  ret void
}

@float_storage = addrspace(1) global float 1.0, align 4

; CHECK: %[[#Known]] = OpFunction %[[#FloatPtr]] None %[[#FloatRet]]
; CHECK: %[[#FG:]] = OpPtrCastToGeneric %[[#FloatPtr]] %[[#]]
; CHECK: OpReturnValue %[[#FG]]{{$}}
; CHECK: OpFunctionEnd
define spir_func ptr addrspace(4) @get_ptr_float() {
  %p = addrspacecast ptr addrspace(1) @float_storage to ptr addrspace(4)
  %v = load volatile float, ptr addrspace(4) %p, align 4
  ret ptr addrspace(4) %p
}
