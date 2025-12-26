# rocm-truth

Machine-verifiable GPU execution receipts for ROCm environments.

rocm-truth runs a deterministic GPU workload, forces a GPU -> CPU data
dependency, and emits a JSON receipt that captures the exact environment and
result.

## Quick start

```powershell
python rocm-truth.py
```

Exit codes:

- 0 on PASS
- 1 on FAIL or INVALID_ENVIRONMENT

Receipts are written to the working directory as:

```
rocm-truth-<hash>.json
```

## Verification fields

- `execution_verified`: true only when a GPU-computed value is read back on the CPU
- `rocm_stack_verified`: true only when ROCm tools are detectable (Linux only)

## Spec

- `SPEC.md`

## Example receipts (synthetic)

- `docs/example_receipt_pass.json`
- `docs/example_receipt_fail.json`

## Contributing

Keep changes minimal and verification-only. If a change cannot be proven, it
must be marked explicitly. Issues and small PRs are welcome.
