# Plan: MLIR runtime wrapper backed by liboffload

## Goal and scope

Add an opt-in MLIR execution-engine runtime library, provisionally named
`mlir_offload_runtime`, implementing the `mgpu*` C interface using LLVM's
`libLLVMOffload` and generated `offload/OffloadAPI.h`.

The first release must cover the functionality exported by
`mlir/lib/ExecutionEngine/LevelZeroRuntimeWrappers.cpp`, including native AOT
modules, supported JIT modules (including SPIR-V on Level Zero), named kernel
lookup, launches, device/shared memory, copies, fills, streams, events, and
default-device selection. It must preserve
valid generated host-code behavior, not reproduce bugs in the existing wrapper.
Level Zero dynamic local memory is an explicit initial exclusion; minimum API
coverage does not imply unrestricted feature parity.

The wrapper is backend-neutral from the outset and targets the CUDA, AMDGPU, and
Level Zero devices exposed by liboffload. The existing Level Zero wrapper defines
the minimum API feature set, not a backend restriction or a Level Zero-first goal.
Validate common functionality on each backend; record backend-specific gaps and
prerequisites explicitly. Broader CUDA/ROCm wrapper functionality is outside the
initial scope. Host-device execution is a separate follow-up, not an automatic
fallback for GPU execution.

Do not replace or change the default behavior of existing CUDA, ROCm, SYCL, or
Level Zero wrappers. Do not depend on Intel SYCL, upstream libsycl, private
libsycl ABI, or direct vendor-runtime calls.

This is a source-based implementation plan against commit `7374c58a131c`.
No proposed implementation or GPU execution has been validated yet.

## Architecture

```text
MLIR GPU lowering and binary emission
  |-- host code: mgpu* calls, opaque handles, launch arguments
  `-- device image: backend-native AOT, SPIR-V, or LLVM bitcode
          |
          v
mlir_offload_runtime
  |-- device selection and context ownership
  |-- stream/event/module/kernel adapter objects
  |-- binary adaptation and argument-ABI adaptation
  `-- offload error handling
          |
          v
libLLVMOffload public C API
          |
          v
LLVM backend plugin compiled into libLLVMOffload
          |
          v
External vendor runtime/driver and hardware
```

The adapter's dependencies are liboffload and, if it creates binary containers,
LLVMObject/LLVMSupport. Vendor runtimes/loaders are selected by liboffload; MLIR
must not independently find or link CUDA, HSA, or Level Zero libraries for this
wrapper.

## Source anchors

- Existing parity baseline: `mlir/lib/ExecutionEngine/LevelZeroRuntimeWrappers.cpp`.
- Runtime target definitions: `mlir/lib/ExecutionEngine/CMakeLists.txt`.
- Runner options: `mlir/CMakeLists.txt`.
- Binary embedding and final launch-call emission:
  `mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp`.
- Operand legalization, memory operations, and async lowering:
  `mlir/lib/Conversion/GPUCommon/GPUToLLVMConversion.cpp`.
- Public offload API definitions: `offload/liboffload/API/*.td`.
- API implementation: `offload/liboffload/src/OffloadImpl.cpp`.
- CUDA/AMDGPU program loading and JIT postprocessing:
  `offload/plugins-nextgen/cuda/src/rtl.cpp` and
  `offload/plugins-nextgen/amdgpu/src/rtl.cpp`.
- Common LLVM-bitcode JIT: `offload/plugins-nextgen/common/src/JIT.cpp` and
  `PluginInterface.cpp` in the same directory.
- Level Zero program loading: `offload/plugins-nextgen/level_zero/src/L0Program.cpp`.
- Level Zero launch implementation: `offload/plugins-nextgen/level_zero/src/L0Kernel.cpp`
  and `L0Queue.cpp` in the same directory.
- Runtime build separation: `llvm/runtimes/CMakeLists.txt`.

## Feature-parity inventory

Preserve the existing ABI widths and calling conventions; do not substitute C++
implementation handle types into generated declarations.

| Existing entry point | Proposed implementation | Important adaptation |
| --- | --- | --- |
| `mgpuSetDefaultDevice` | Enumerate devices and choose the requested index | Define index scope and enforce resource-lifetime restrictions |
| `mgpuStreamCreate` | `olCreateQueue` | Create adapter-owned stream containing context/device/queue |
| `mgpuStreamSynchronize` | `olSyncQueue` | Preserve null-stream no-op behavior |
| `mgpuStreamDestroy` | Sync, then `olDestroyQueue` | Existing wrapper synchronizes on destruction |
| `mgpuStreamWaitEvent` | `olWaitEvents` | Enqueue dependency rather than reproduce blocking/reset behavior |
| `mgpuEventCreate` | Allocate empty adapter event | Offload event creation requires a queue and records immediately |
| `mgpuEventRecord` | `olCreateEvent` on the given stream | Safely replace previous event snapshot |
| `mgpuEventSynchronize` | `olSyncEvent` | Define unrecorded-event and repeated-wait behavior |
| `mgpuEventDestroy` | Release/defer `olDestroyEvent` | Backend event must survive pending signals/waits |
| `mgpuMemAlloc` | `olMemAllocAligned`, alignment 64 | DEVICE versus MANAGED from `isShared` |
| `mgpuMemFree` | Sync stream, then `olMemFree` | Track original allocation context; null pointer is harmless |
| `mgpuMemcpy` | Queue-based generic `olMemcpy` path | Clarify null/null endpoint contract; initial scope is host/selected-device copies |
| `mgpuMemset16` | `olMemFill` with 2-byte pattern | Count is elements, not bytes; check multiplication overflow |
| `mgpuMemset32` | `olMemFill` with 4-byte pattern | Preserve stream ordering and pattern value |
| `mgpuModuleLoad` | Validate/adapt image for selected backend, then `olCreateProgram` | Native formats are backend-specific; Level Zero needs an OffloadBinary container |
| `mgpuModuleLoadJIT` | Validate/adapt JIT input, then `olCreateProgram` | Distinguish LLVM bitcode, SPIR-V, and backend-specific inputs; honor embedding lengths |
| `mgpuModuleGetFunction` | `olGetSymbol`, kind KERNEL | Return adapter kernel handle; symbol owned by program |
| `mgpuLaunchKernel` | `olLaunchKernel` | Argument sizes and dynamic local memory require explicit decisions |
| `mgpuModuleUnload` | Complete relevant work, then `olDestroyProgram` | Invalidate kernel handles and release adapter state |

Host registration, profiling, cooperative/cluster launch, and other APIs exported
only by CUDA/ROCm wrappers are follow-up work, not prerequisites for this scope.

## Design decisions and blockers

### 1. Device selection and context consistency

Initialize liboffload once with a thread-safe process-level owner. Never call
`olShutDown` while adapter resources remain alive; account for other liboffload
users in the process rather than assuming exclusive ownership.

Enumerate once with `olIterateDevices` and cache eligible GPU devices across all
enabled backends, preserving callback order. Define `mgpuSetDefaultDevice(index)`
against that cached list and default to index zero. This makes indices consistent
within the wrapper's lifetime; the API does not guarantee stable ordering across
process runs, driver changes, or hardware changes. Track backend/vendor/name/UID
for actionable diagnostics. Device selection is policy, not a missing offload API.
Do not silently fall back to a CPU or switch backends when an image is incompatible
with the selected GPU.

Implement the existing `gpu.set_default_device` contract through
`mgpuSetDefaultDevice`: a zero-based index into the system's GPUs, with thread-local
selection permitted by the dialect. The operation is defined in
`mlir/include/mlir/Dialect/GPU/IR/GPUOps.td`; thread-local selection is allowed,
not mandated. Choose thread-local selection to match existing wrapper behavior;
no IR contract change or new device-selection API is needed.

Create a context lazily for the selected device and make resource ownership
explicit. Streams, modules, kernels, allocations, and events retain their context
and device; reject incompatible resource combinations. Initially reject switching
while that thread has live resources. Stable cross-process indices, UID-based
selection, and automatic image-driven selection are outside the initial scope.
This is bounded adapter policy/ownership work, not a missing liboffload capability.
The exact first-driver/subdevice behavior of the old wrapper needs an enumeration
test; liboffload enumeration should not be assumed identical to `zeDeviceGet`.

Use `olIsValidBinary` to validate input for the selected device. An explicit
image-driven selection policy can use `olIterateCompatibleDevices`, but should
happen before allocations and streams fix the execution context, not silently
during module loading. Test indexing with multiple devices and mixed backends.

### 2. Backend-specific AOT and JIT formats

The Level Zero plugin currently accepts raw binary SPIR-V. Native code is accepted
inside a single-entry LLVM OffloadBinary with `IMG_Object` and an appropriate
`spirv64` target triple. Neither LLVM bitcode nor SPIR-V is a native GPU binary.

Pass compiler-produced images to `olCreateProgram` for the selected device,
performing only required container/embedding adaptation. Delegate compatibility
checking and format handling to liboffload wherever supported; do not implement
another vendor loader in MLIR. Backend-dependent binary formats are distinct from
the common `olLaunchKernel` interface and its argument-marshalling contract.
Keep container construction in the wrapper initially; compiler-side packaging is
not required for the initial scope.

- `mgpuModuleLoad`: forward compatible native CUDA/AMDGPU images through the public
  offload API. For raw Level Zero native bytes, create the required container with
  validated target metadata. Reuse LLVMObject serialization; do not implement a
  second binary-container parser/writer. Verify accepted formats through the
  offload path, not merely through vendor API documentation.
- `mgpuModuleLoadJIT`: support target-specific LLVM bitcode through the common JIT
  path and raw SPIR-V for Level Zero. Audit legacy CUDA PTX input against the
  offload loader; do not assume it is accepted or silently reinterpret it.
  Existing Assembly-format embedding appends one NUL byte. Define exact lengths
  per format: check nonzero size and trailing NUL before removing a known extra
  byte; check SPIR-V word alignment when applicable. Do not blindly trim arbitrary
  bitcode/native input or underflow size.
- Container input can be an explicitly documented additional accepted format.
- Keep module bytes alive through the call; the offload API promises to copy them.
- Report module creation errors with backend, device, format, and offload details.

Compiler-side container emission is a later optimization. Device-native images
remain device-specific; packaging does not make them portable.

The existing Level Zero JIT wrapper ignores `optLevel`, whereas legacy CUDA JIT
has vendor-specific options. Define and document the new wrapper's behavior for
each JIT input; do not claim per-program optimization control that
`olCreateProgram` does not expose. Global LLVM JIT controls are not equivalent.

### 3. Kernel arguments: preserve compiler-known sizes in a launch variant

Arguments are typed in MLIR, and their final ABI sizes and count are generally
known at compile time. The current `mgpuLaunchKernel` call passes `void **params`
and `paramsCount`, discarding sizes; a separately compiled wrapper cannot recover
pointee types from `void *`. This is an information-preservation task, not runtime
type discovery or a need for separate vendor launch interfaces.

Use a new variant, provisionally `mgpuLaunchKernelWithArgSizes`, as the primary
launch path for the offload wrapper. Retain the existing launch parameters and
add `const size_t *argSizes`, parallel to `params`. An opt-in GPU
translation/offloading-handler option emits this call and both arrays, forwarding
them to `olLaunchKernel`. Existing wrappers and their default `mgpuLaunchKernel`
ABI remain unchanged.

Compute sizes from the final lowered device argument types and ABI/data layout,
including promoted/flattened memref operands and pointer address spaces. Check
host storage representation against device representation; don't merely use
source operand sizes or assume all host/device layouts agree. Test mixed widths,
aggregates if supported, zero-argument kernels, and bare-pointer versus descriptor
lowering. Reject representations the adapter cannot marshal.

A legacy pointer-sized `mgpuLaunchKernel` compatibility entry point is optional,
not a prerequisite or primary path. If included, explicitly document its narrow
argument convention; do not silently use pointer size for arbitrary arguments.
Keep `extra` unused unless a separate contract is agreed; it is not an implicit
argument-size channel.

### 4. Dynamic local/shared memory: initial Level Zero limitation

Level Zero dynamic local memory is explicitly unsupported in the initial wrapper
scope and is not a release blocker. A nonzero request on Level Zero must produce
a clear unsupported-feature diagnostic, never be silently ignored. This is a
backend-specific limitation, not a restriction on the backend-neutral design.

The existing MLIR Level Zero wrapper attempts to set the final local-memory
argument via `zeKernelSetArgumentValue(index, requestedBytes, nullptr)`, but this
path has not been validated end to end in this investigation. Generic host launch
emission does not automatically append that slot. Do not treat the presence of
wrapper code as proof that the existing lowering/runtime combination works.
The current offload Level Zero backend explicitly rejects nonzero dynamic memory.
No change to that backend behavior is required for the initial milestone.

Forward the common byte count through `DynSharedMemory` where supported, after
validating negative input and integer conversions. CUDA already forwards dynamic
bytes in its launch configuration; AMDGPU adds them to the HSA group segment size.
Validate those paths against backend-appropriate kernels. Do not add a Level
Zero-style trailing argument to CUDA/AMDGPU launches. Static work-group memory
is distinct from the unsupported dynamic-local allocation path.

Future Level Zero work should first establish a working device/host signature and
local-argument convention with an end-to-end reproducer. Level Zero supports
runtime-sized local allocation, including a size_t value supplied for an SLM
argument through `zeCommandListAppendLaunchKernelWithArguments` in the latest
specification. Evaluate the available driver API versions and the explicit-setting
fallback rather than assume the pointer-list path cannot support it. Agree an
explicit offload representation, argument counting, and memory-limit validation
before adding support. This investigation and backend/API changes are deferred.

### 5. Stream/event ordering and deferred reclamation

Use one offload queue per logical stream. Preserve copy/fill/launch order through
the queue; reproducing the old wrapper's separate compute/copy lists is unnecessary.
The backend can own scheduling decisions.

An adapter event starts empty. Recording materializes an `olCreateEvent` snapshot
of that stream. Waiting uses `olWaitEvents`; synchronization uses `olSyncEvent`.
Define unrecorded waits, re-recording, and completed-event behavior explicitly.
Do not reproduce the baseline's event reset on synchronization or its ambiguous
signal-on-both-command-lists behavior.

MLIR async tokens already express the dependencies; no additional compiler
consumer-count or conditional-control-flow analysis is required. Lowering can
emit record -> enqueue wait -> destroy event immediately. The required runtime
contract is that destruction invalidates the host handle, not dependencies already
submitted. Retain actual queued uses, not hypothetical consumers in untaken branches.

Verify this contract in liboffload first and prefer backend fixes over duplicating
a dependency tracker in the MLIR adapter. CUDA delegates queued-event semantics
to the driver; the AMDGPU backend retains dependent HSA signals with use counts
and releases them through stream completion actions.

Level Zero has two distinct event models. The regular-event API requires the
caller to ensure the device no longer references an event before destruction.
The latest programming guide explicitly allows counter-based events to be reused
or destroyed without waiting; queued waits retain their earlier counter point.
The offload backend uses an experimental counter-based pool descriptor when
available and skips reset for such events. Verify that extension's semantics and
capability selection rather than assume every pooled event has a lifetime bug.
The regular-event fallback's immediate recycling/reset remains a source-based
risk to reproduce and fix, not a demonstrated hardware failure.

References:
- [Level Zero zeEventDestroy](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/event.html#zeeventdestroy).
- [Counter-based events programming guide](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#counter-based-events).

If needed for regular events, defer reclamation until producer/consumer queued
uses complete. Producer signaling alone does not prove all consumers finished
reading the event. A conservative synchronized-reclamation implementation is
acceptable initially if documented and measured; avoid synchronizing future work
while holding adapter locks. Stream destruction must retire dependencies without
invalidating explicit event handles that remain live.

Test repeated recording, source-stream destruction, destroy-after-wait, and forced
pool reuse. Restrict incompatible context/device event waits with diagnostics
until their semantics are verified. A missing backend event implementation must
not be silently treated as sufficient for required asynchronous ordering.

### 6. Memory ownership and errors

Initial copy scope is ordinary host memory and device/shared memory associated
with the calling thread's selected device/context: H->D, D->H, and same-device
D->D. Device selection is implicit for subsequent GPU operations; cross-device
copies and importing external GPU allocations are outside the initial scope.
This is a chosen initial execution model, not a claim that thread-local default
selection alone makes all multi-device programs impossible.

Use the stream's non-null offload queue for generic copies. The current
`olMemcpy(queue, dst, nullptr, src, nullptr, size)` implementation delegates to
that queue device's `dataMemcpy` operation, allowing backend address inference.
However, the public documentation currently describes null endpoint devices as
host memory. Clarify/document this supported queue-based behavior (or add an
explicit generic-copy API) and test it on CUDA, AMDGPU, and Level Zero before
relying on it. Never use the same null/null endpoint combination with a null
queue for device pointers: that path executes host `std::memcpy`.

Do not require an allocation range map solely to classify copy endpoints if the
generic queue-based contract is established. Retain minimal context/ownership
bookkeeping needed for allocation lifetime, free, and device-switch restrictions;
use offload allocation queries where appropriate. Copies involving interior
pointers and shared allocations need tests, not a new MLIR pointer representation.
If a backend cannot honor generic copies, resolve that API/backend gap explicitly
rather than silently misclassify endpoints.

Managed allocation is required for `isShared=true`; do not substitute device-only
or ordinary host memory. Unsupported managed memory gets an explicit error.

Check all `ol_result_t` values against `OL_SUCCESS`; report code/details and the
failing operation. Exceptions must not escape exported C entry points. Use the
existing fail-fast wrapper convention initially. Never report asynchronous work
as successful merely because a handle was constructed.

### 7. Resource lifetime: conservative functional implementation

Prioritize correctness and functionality over performance in the initial
implementation. Use existing offload destruction APIs with simple adapter
ownership bookkeeping; no new compiler analysis or performance-oriented resource
tracking is required.

- `mgpuMemFree(ptr, stream)`: synchronize the supplied stream before freeing in
  the allocation's owning context. The caller must order uses on other streams;
  synchronizing this stream alone does not protect unrelated outstanding uses.
- `mgpuStreamDestroy(stream)`: synchronize, then destroy its offload queue.
- `mgpuModuleUnload(module)`: synchronize live queues in the owning context before
  destroying the program. Invalidate adapter kernel handles, which refer to
  symbols owned by that program.
- Device switching: initially reject it while the calling thread has live
  resources, rather than support simultaneous resources on multiple defaults.
- Context destruction: release it only after its queues, programs, allocations,
  and events have been safely released, including deferred event state.

Context-wide synchronization on module unload and conservative event reclamation
are acceptable initial costs. Optimize only after the functional tests pass;
avoid building a complex per-kernel/per-allocation use tracker for this milestone.

## Build and test integration

Proposed new files:

- `mlir/lib/ExecutionEngine/OffloadRuntimeWrappers.cpp`.
- `mlir/cmake/modules/FindOffloadRuntime.cmake`, if no usable exported package exists.
- Wrapper unit tests under the existing MLIR execution-engine unit-test hierarchy.
- `mlir/test/Integration/GPU/Offload/` for gated hardware integration tests.

Add `MLIR_ENABLE_OFFLOAD_RUNNER` (OFF by default), the `mlir_offload_runtime` shared
library, `%mlir_offload_runtime` lit substitution, and a separate test feature.
Keep existing runner gates and tests intact.

Use target-based dependency wiring and an `if(TARGET ...)` availability gate.
Prefer the real `LLVMOffload` library target when visible; link it with
`target_link_libraries(mlir_offload_runtime PRIVATE LLVMOffload)` to consume its
library dependency and public include interface. Ensure generated `OffloadAPI.h`
is ready before wrapper compilation, adding a generation dependency as needed.
Target availability is a build integration condition, not a hardware/driver probe.

Core LLVM targets are already defined before MLIR is configured. However, the
current `llvm/CMakeLists.txt` adds `lib`, then `tools` (which adds MLIR through
`llvm/tools/CMakeLists.txt`), then `runtimes`. Thus a gate inside MLIR does not yet
see runtime bridge targets created later. Establish an imported bridge before
MLIR's gate or finalize dependency wiring after runtime setup; do not reorder the
whole runtimes build blindly, since it depends on toolchain targets. `if(TARGET
...)` only sees targets already defined and visible at that point. A missing
target may mean configuration order or scope, not that offload was disabled.
Imported library targets and interface targets are standard dependency mechanisms,
not blockers: the wrapper consumes the target interface whether the underlying
library is prebuilt or produced by the overall build. Attach build-order
prerequisites only in the latter case.
Keep wrapper enablement opt-in; when explicitly requested and no usable dependency
target can be established, emit a clear diagnostic rather than silently disable it.

With `LLVM_ENABLE_RUNTIMES=offload`, runtimes normally use a separate CMake build;
its `LLVMOffload` library target is not automatically visible in MLIR's parent
configuration. Bridge this boundary with an imported library target exposing the
correct library location and public includes, plus explicit external-build and
generated-header dependencies. A runtime utility target is suitable for ordering,
but is not itself a linkable library. Check for dependency cycles before wiring
MLIR compilation/linking to the runtime build.

Also support an installed matching runtime through an imported target. Prefer an
existing package export if available; otherwise use narrowly scoped discovery
with an explicit root option and required header/library checks. Support platform
library names and per-target installation directories without hardcoded version
suffixes. A separate manual/two-stage build is a fallback workflow, not the only
intended integration.

This is bounded target/interface wiring across a build boundary, not a fundamental
runtime blocker. Validate clean parallel builds, generated-header ordering,
incremental rebuilds, and install-tree RPATH/library deployment. Linking records
binary dependencies; it does not by itself deploy the libraries. Do not reuse
MLIR's `FindLevelZeroRuntime.cmake` or independently discover vendor libraries.
Use a separate experimental build rather than modifying the existing MLIR cache.

## Implementation phases

### Phase 0: prove backend capabilities and settle ABI

- Write small direct-offload probes on CUDA, AMDGPU, and Level Zero for native
  image load, named lookup, launch, managed/device allocation, queue copy/fill,
  and event dependencies; gate execution on available hardware.
- Probe raw SPIR-V and packaged native loading on Level Zero, and target-specific
  LLVM-bitcode JIT on each backend with its required target/tool dependencies.
- Reproduce backend-specific gaps, including Level Zero nonzero local-memory
  rejection and event lifetime behavior.
- Record a precise compatibility argument convention and sized-launch design.
- Determine the generated header/package export/install layout and build workflow.

Deliverable: passing core probes and explicit remaining blockers, not a claim of
full parity. Backend fixes can proceed independently of MLIR adapter implementation.

### Phase 1: core adapter and build discovery

Implement initialization, thread-local selection, context/resource ownership,
streams, minimal allocation ownership bookkeeping, queue-based copies/fills, error
handling, module adaptation,
named lookup, and sized-argument launch, together with opt-in MLIR emission of the
new launch variant. Add unit tests with a mocked
public offload API; do not require GPU hardware for ownership/marshalling tests.

Deliverable: a backend-neutral adapter with synchronous native AOT and supported
JIT executions on available CUDA, AMDGPU, and Level Zero devices, without direct
vendor-runtime/SYCL dependencies. Record untested backend paths explicitly.

### Phase 2: asynchronous behavior and lifecycle

Implement event objects, stream waits, recording/re-recording, safe deferred
reclamation, and cleanup/device-switch rules. Exercise actual sequences produced
by `GPUToLLVMConversion.cpp`, including immediate event destruction.

Deliverable: asynchronous copy -> launch -> copy and cross-stream dependency tests
without host-side data races or premature backend event reuse.

### Phase 3: complete scoped functionality

Close capability gaps required by the agreed initial scope and complete
sized-argument lowering coverage, preserving existing wrappers' default ABI.
Validate geometry, argument count, sizes, range overflow, and resource lifetime.
Keep Level Zero dynamic-local-memory support deferred; test clear rejection of
nonzero requests instead. Validate dynamic shared memory on CUDA/AMDGPU where
supported without vendor-specific assumptions in the common adapter.

Deliverable: all baseline entry points implemented with tested supported semantics
and explicitly documented limitations. Do not claim unrestricted feature parity:
Level Zero dynamic local memory is an accepted initial exclusion.

### Phase 4: package, document, and expand

Validate installed MLIR plus installed liboffload, document the two-stage build,
device selection, image formats, argument ABI, and runtime driver prerequisites.
Consider same-build support only after the independent build is reliable.

Complete the CUDA/AMDGPU/Level Zero validation matrix and document capability
coverage, including LLVM-bitcode JIT. Hardware availability may stage execution
validation, but backend neutrality is not deferred to this phase. Do not
reinterpret legacy JIT input silently: distinguish binary formats, validate
lengths, and define whether a terminator belongs to the embedding contract.

## Validation matrix

### Hardware-independent checks

- Mock offload tests for every entry point, parameter forwarding, ownership, errors,
  module format adaptation, allocation interior pointers, and overflow handling.
- Verify the exported `mgpu*` ABI and exact argument widths.
- Extend `mlir/test/Target/LLVMIR/gpu.mlir` for binary/JIT embedding and optional
  sized-launch declarations/calls.
- Extend GPUCommon launch/async/wait conversion tests without changing defaults.
- Test OFF configuration, missing dependency diagnostics, and separate installed
  runtime discovery. Wrapper-only builds must not require a GPU or Level Zero SDK.
- Check shared-library dependencies: libLLVMOffload is linked; no new direct SYCL
  or Level Zero loader dependency is introduced by the wrapper.

### Hardware checks across CUDA, AMDGPU, and Level Zero

Run the common tests independently for each backend using backend-appropriate
images and capability gates. Missing hardware is an explicit validation gap, not
a successful test. Keep Level Zero-specific tests for SPIR-V/native packaging and
its local-argument convention; do not treat these as the only integration coverage.

- Port relevant existing CUDA, ROCm, and Level Zero integration scenarios to the
  new wrapper gate, without modifying their original runtime selections.
- Backend-native AOT load and supported JIT load, each with named lookup and
  observable output; include raw SPIR-V on Level Zero.
- Cached enumeration indices, mixed-backend device selection, incompatible-image
  rejection, and device/context consistency.
- Device and shared allocation; host/device/device copies; 16-/32-bit fills.
- Sized mixed scalar/pointer arguments; pointer-sized compatibility tests only if
  the optional legacy launch entry point is included.
- Zero-argument launch, 1D/2D/3D geometry, and repeated launches.
- Dynamic shared memory on CUDA/AMDGPU where supported; clear unsupported-feature
  failure for nonzero Level Zero dynamic-local-memory requests.
- Two streams with record/wait, event re-record, immediate destruction after wait,
  repeated pool reuse, synchronization, and stream destruction with work pending.
- Invalid device index, unknown kernel, incompatible image, malformed JIT length,
  illegal geometry, context mismatch, and unsupported feature diagnostics.
- Module unload/free after queued work; repeated load/unload and device switching.

Existing test anchors:

- `mlir/test/Integration/GPU/LevelZero/gpu-memcpy-addf32-to-spirv.mlir`.
- `mlir/test/Conversion/GPUCommon/lower-launch-func-to-gpu-runtime-calls.mlir`.
- `mlir/test/Conversion/GPUCommon/lower-async-to-gpu-runtime-calls.mlir`.
- `mlir/test/Conversion/GPUCommon/lower-wait-to-gpu-runtime-calls.mlir`.
- `mlir/test/Dialect/GPU/module-to-binary-spirv.mlir`.
- `offload/unittests/OffloadAPI/kernel/olLaunchKernel.cpp` (local-memory cases
  currently skip Level Zero).
- Offload queue/event unit tests, especially `olWaitEvents` and `olDestroyEvent`.

### LLVM-bitcode JIT checks

`olCreateProgram` detects raw LLVM bitcode and invokes the common JIT before
backend loading. This is separate from passing SPIR-V directly. Triple matching
is a candidate check, not proof that all device features/libraries are available.

| Selected backend | Bitcode architecture | Runtime compilation | Additional requirements |
| --- | --- | --- | --- |
| Level Zero | `spirv64` | LLVM -> SPIR-V -> driver native compilation | SPIRV enabled in LLVM_TARGETS_TO_BUILD; compatible driver |
| CUDA | `nvptx64` | LLVM -> PTX -> cubin | NVPTX enabled; `ptxas` on PATH |
| AMDGPU | `amdgcn` | LLVM -> object -> HSA code object | AMDGPU enabled; `lld` on PATH |

Use device-lowered, version-compatible LLVM bitcode with correct target triples,
calling conventions, address spaces, and resolved device-library requirements.
Textual LLVM IR is not detected by the ordinary bitcode entry path. Validate Intel
bitcode JIT end to end before advertising it; current source wiring alone is not
an execution result. `optLevel` has no per-program offload parameter; global JIT
controls are not a safe substitute for independent concurrent program options.

## Acceptance criteria

The initial functional milestone (baseline API coverage with the documented
Level Zero dynamic-local-memory exclusion) is complete only when:

1. Every Level Zero wrapper entry point in the inventory is available with defined
   ABI and valid-input semantics.
2. Backend-appropriate native AOT and supported JIT images load, resolve by name,
   and execute on CUDA, AMDGPU, and Level Zero; Level Zero includes SPIR-V JIT.
   Untested hardware paths and backend capability gaps are explicitly recorded.
3. Sized launch arguments are handled correctly. Level Zero dynamic local memory
   is documented as unsupported and nonzero requests fail clearly; supported
   CUDA/AMDGPU dynamic shared-memory paths are tested.
4. Async dependencies and event lifetimes survive generated MLIR cleanup sequences.
5. Build-tree and install-tree workflows work without external SYCL or independent
   Level Zero discovery in MLIR.
6. Existing wrapper defaults and tests remain unchanged.
7. Unsupported inputs/backends fail clearly rather than silently changing execution.

The Level Zero wrapper supplies the minimum API inventory only. CUDA, AMDGPU, and
Level Zero are architectural targets of the new wrapper from the beginning;
backend support and JIT formats must not be advertised as validated before their
tests pass. Additional vendor-wrapper APIs and host execution are later scope.
