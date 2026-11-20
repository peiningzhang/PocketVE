# PocketVE

**PocketVE: Stable and Property-Guided Structure-Based Drug Design with Variance-Exploding Diffusion**

Peining Zhang and Jinbo Bi · NeurIPS 2026

[Paper](https://arxiv.org/abs/2609.08101) · [Pretrained weights](samples/model_artifacts/training_cfg_muon_release/checkpoints/last.pt) · [Benchmark and training](REPRODUCIBILITY.md)

PocketVE generates 3D ligands conditioned on a protein pocket. It combines variance-exploding coordinate diffusion, property guidance at sampling time, and adaptive protein-pocket perturbation during training.

## Installation

From the repository root, create the Linux environment based on the TAGMol dependency stack. It uses Python 3.8, PyTorch 1.13.1, CUDA 11.6, and PyG 2.2.0.

```bash
conda env create -f environment.yml
conda activate pocketve
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
```

## Pretrained weights

[`last.pt`](samples/model_artifacts/training_cfg_muon_release/checkpoints/last.pt) is the model checkpoint used for the main paper results. Use it with the accompanying [sampling configuration](samples/model_artifacts/training_cfg_muon_release/sampling.yml).

SHA-256 `8b3213357eceef849e0b2e730a14f7651ddafdaf579d1ca5df8a4da278b1ddd7`

## Sample one pocket

From the repository root, set `--pocket_pdb` to a PDB cropped to the binding pocket. This example samples ten ligands and exports the reconstructed molecules without requiring a benchmark dataset. Use `--device cpu` for a CPU run.

```bash
model_dir=samples/model_artifacts/training_cfg_muon_release

python code/scripts/quick_evaluate.py \
  --config "$model_dir/sampling.yml" \
  --checkpoint "$model_dir/checkpoints/last.pt" \
  --pocket_pdb /path/to/pocket.pdb \
  --num_proteins 1 \
  --num_ligands_per_protein 10 \
  --num_steps 100 \
  --time_scheduler gen_arcsin \
  --cfg_scale 5.0 \
  --docking_mode none \
  --export_grouped_sdf
```

The run writes sampling results under `tmp_eval/quick_eval_*/` and reconstructed molecules under its `sdfs_grouped/` subdirectory. See the [included sample outputs](samples/quick_evaluate_release/sdfs_grouped) for examples. To sample from the CrossDocked2020 test split, follow [Data setup](REPRODUCIBILITY.md#data-setup) and omit `--pocket_pdb`.

## Evaluation

- [`quick_evaluate.py`](code/scripts/quick_evaluate.py) runs sampling, evaluation, and grouped SDF export. See the [main benchmark command](REPRODUCIBILITY.md#main-benchmark) for the paper setting.
- [`evaluate_diffusion.py`](code/scripts/evaluate_diffusion.py) evaluates existing sampling outputs for Vina scores, QED, SA, and diversity.
- [`run_genbench_eval.py`](scripts/run_genbench_eval.py) evaluates grouped SDFs with a separate GenBench3D installation. See the [GenBench3D setup and command](REPRODUCIBILITY.md#genbench3d-geometry-metrics).

The paper describes the PoseCheck and pocket-permutation protocols in [Appendix N](https://arxiv.org/html/2609.08101v3#A14).

## Citation

```bibtex
@inproceedings{zhang2026pocketve,
  title     = {PocketVE: Stable and Property-Guided Structure-Based Drug Design with Variance-Exploding Diffusion},
  author    = {Zhang, Peining and Bi, Jinbo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2609.08101}
}
```

## License and acknowledgments

See [LICENSE](LICENSE) and the [third-party notices](THIRD_PARTY.md) for code licensing and attribution. The implementation builds on [TAGMol](https://github.com/MoleculeAI/TAGMol) and [TargetDiff](https://github.com/guanjq/targetdiff).
