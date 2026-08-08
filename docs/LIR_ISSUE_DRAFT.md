# LiR issue report — Colab dependency compatibility

A report has been prepared for the upstream NFI LiR project based on the DeepGuard Colab experience.

## Observed problem

Installing LiR in a current Google Colab environment caused dependency/version conflicts involving the preinstalled CUDA/RAPIDS stack. In particular, the environment reported:

`cudf-cu12 26.2.1 requires numba<0.62,>=0.60, but numba 0.65.1 was installed.`

The conflict is not necessarily a LiR algorithmic defect; it is an environment reproducibility/packaging problem for a GPU-oriented Colab workflow.

## Request to upstream

Please consider documenting and/or providing a tested Colab environment (requirements/lockfile or container) that avoids conflicts with the current Colab CUDA/RAPIDS packages, and clarify the supported Python versions and dependency pins for the current release.

DeepGuard has temporarily removed the external `lir` runtime dependency from its Colab path and uses a small, explicitly tested internal LR module so that detector research is not blocked by environment resolution. This is not intended as a replacement for LiR; it is a reproducibility workaround.

## Reproduction context

- Google Colab
- GPU runtime
- DeepfakeBench/DF40 research workflow
- LiR installed alongside the Colab CUDA/RAPIDS environment
- Dependency resolver reported the numba/cuDF/cuML incompatibility above

## Desired outcome

A minimal, reproducible Colab installation recipe for LiR, ideally with tested Python/CUDA/package versions and no manual package downgrades that can destabilize the runtime.
