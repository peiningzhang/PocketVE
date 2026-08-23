# External Resources

This supplement package includes one reproducibility checkpoint (`354000.pt`) and does not bundle full datasets.

## Expected External Inputs

- Benchmark/test datasets required by scripts under `code/` and `scripts/`
- Optional external tools for evaluation (for example `genbench3d`)

## How to Use

1. Download or prepare required datasets locally.
2. (Optional) Install and prepare a local `genbench3d` environment.
3. Set local paths with environment variables:
   - `DATA_ROOT`
   - `CKPT_ROOT`
4. Update `code/configs/*.yml` where needed.
5. If `genbench3d` is not available, run sampling with SDF export and perform downstream validation from grouped SDF files.

## Licensing and Attribution

Please follow the licenses and citation requirements of all third-party datasets, checkpoints, and evaluation tools you use.
