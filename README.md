[![CI](https://github.com/juxstin1/rocm-truth/actions/workflows/ci.yml/badge.svg)](https://github.com/juxstin1/rocm-truth/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/juxstin1/rocm-truth)](LICENSE)

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

## Regulatory & Compliance

Attach receipts to prove real GPU execution occurred.

**Government & Policy**
- Export controls — verify what hardware actually ran a workload
- Carbon reporting — prove actual GPU usage, not estimates
- Compute subsidies — prove funded hardware is being utilized
- Data sovereignty — verify compute jurisdiction (pair with location attestation)

**Commercial**
- Cloud billing — receipt proves you got what you paid for
- SLA verification — prove the GPU tier promised was delivered
- Audit trails — every inference request gets a hardware receipt

**AI/ML**
- Model training claims — prove which GPU trained a model
- Inference verification — attach receipts to LLM API responses

## Verification fields

- `execution_verified`: true only when a GPU-computed value is read back on the CPU
- `rocm_stack_verified`: true only when ROCm tools are detectable (Linux only)

## What this repo is not

- Not a benchmark or performance suite.
- Not an installer or ROCm setup guide.
- Not a toolkit or workflow system.

## Spec

- `SPEC.md`

## Example receipts (synthetic)

- `docs/example_receipt_pass.json`
- `docs/example_receipt_fail.json`

## Contributing

Keep changes minimal and verification-only. If a change cannot be proven, it
must be marked explicitly. Issues and small PRs are welcome.
