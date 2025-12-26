# rocm-truth spec

This document defines the verification semantics and output fields produced by
rocm-truth.

## Definitions

execution_verified

True only when a GPU-computed value is read back on the CPU via a forced data
dependency. If this value is returned, GPU execution occurred.

rocm_stack_verified

True only when ROCm userland tools are detectable (rocminfo/rocm-smi/hipconfig).
This is expected to be false on Windows.

## Verification method

rocm-truth runs a deterministic GPU matrix multiply and then reads a single
scalar with `.item()`. This enforces a device synchronization and a GPU -> CPU
data transfer. The scalar cannot exist unless the GPU kernel executed.

## Status values

PASS

The GPU workload completed and execution_verified is true.

FAIL

The workload was attempted but execution could not be verified, or an error
occurred during execution.

INVALID_ENVIRONMENT

The test could not be executed due to missing prerequisites (no GPU device,
driver/runtime unavailable, or torch not configured for GPU).

## Required fields

Receipts include:

- version
- receipt_hash
- status
- verification (execution_verified, rocm_stack_verified, method)
- test_result
- env
