# PocketVE Supplement Material

This package contains the core implementation and selected reproducible sample outputs for anonymous supplementary review.

## Directory Layout

- `code/`: core implementation (`scripts`, `models`, `datasets`, `utils`, `configs`)
- `samples/`: selected sample outputs and model artifacts for reproducibility checks
- `scripts/`: runnable entry scripts for training/sampling/evaluation
- `THIRD_PARTY.md`: external resources and dependency notes

## Environment

Recommended:

- Linux
- Python 3.10+
- PyTorch with CUDA support (if GPU is available)

Install dependencies using your project environment manager. If you have an existing environment from the main project, you can reuse it.

## Path Conventions

Use environment variables instead of absolute machine-specific paths:

- `PROJECT_ROOT`: root of this supplement package
- `DATA_ROOT`: local path to datasets
- `CKPT_ROOT`: local path to pretrained checkpoints

Example:

```bash
export PROJECT_ROOT=/path/to/PocketVE
export DATA_ROOT=/path/to/data
export CKPT_ROOT=/path/to/pretrained_models
```

## Minimal Reproducibility Workflow

1. Prepare dataset paths under your local environment.
2. Use the included sampling config and checkpoint:
   - Config: `samples/model_artifacts/training_cfg_muon_release/sampling.yml`
   - Checkpoint: `samples/model_artifacts/training_cfg_muon_release/checkpoints/354000.pt`
3. Run the standard sampling command:

```bash
python code/scripts/quick_evaluate.py \
  --config samples/model_artifacts/training_cfg_muon_release/sampling.yml \
  --checkpoint samples/model_artifacts/training_cfg_muon_release/checkpoints/354000.pt \
  --num_proteins 100 \
  --num_ligands_per_protein 10 \
  --docking_mode vina_score \
  --export_grouped_sdf \
  --run_genbench \
  --gb3d_dir /path/to/genbench3d \
  --genbench_python /path/to/genbench3d/python \
  --genbench_do_conf_analysis \
  --genbench_no_vina \
  --time_scheduler gen_arcsin \
  --cfg_scale 5.0
```

4. If `genbench3d` is unavailable, keep `--export_grouped_sdf` and validate using generated SDF outputs.

## Included Samples

- `samples/quick_evaluate_release/` includes:
  - `sdfs_grouped/` (generated SDF files for theoretical downstream validation)
  - `eval_results/` (evaluation outputs)
  - `sampling_timing.json`
- `samples/model_artifacts/training_cfg_muon_release/` includes:
  - `sampling.yml`
  - `checkpoints/354000.pt`

Large intermediate tensors and shard outputs are not included by default.

## Notes on Anonymization

- Machine-specific absolute paths, usernames, and hostnames have been replaced with generic placeholders.
- Public method/paper references are retained for scientific reproducibility.
