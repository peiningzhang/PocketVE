# PocketVE

**PocketVE: Stable and Property-Guided Structure-Based Drug Design with Variance-Exploding Diffusion**

Peining Zhang and Jinbo Bi · NeurIPS 2026

[Paper](https://arxiv.org/abs/2609.08101) · [Pretrained weights](samples/model_artifacts/training_cfg_muon_release/checkpoints/last.pt) · [Reproduction guide](REPRODUCIBILITY.md)

PocketVE generates 3D ligands conditioned on a protein pocket. It combines variance-exploding coordinate diffusion, property guidance at sampling time, and adaptive protein-pocket perturbation during training.

## Environment Setup

From the repository root, create the Linux environment based on the TAGMol dependency stack. It uses Python 3.8, PyTorch 1.13.1, CUDA 11.6, and PyG 2.2.0.

```bash
conda env create -f environment.yml
conda activate pocketve
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
```

## Data and Checkpoints

Training and benchmark evaluation use the processed CrossDocked2020 10 Å pockets and the matching train/test split. Docking and GenBench3D evaluation also use the test-set receptors and reference ligands. See [data setup](REPRODUCIBILITY.md#data-setup) for the required files and paths.

[`last.pt`](samples/model_artifacts/training_cfg_muon_release/checkpoints/last.pt) is the model checkpoint used for the main paper results. Use it with the accompanying [sampling configuration](samples/model_artifacts/training_cfg_muon_release/sampling.yml).

SHA-256 `8b3213357eceef849e0b2e730a14f7651ddafdaf579d1ca5df8a4da278b1ddd7`

## Training

After preparing the data, train from scratch with the [Muon training config](code/configs/training_cfg_muon.yml) and the included [condition-bin specification](samples/model_artifacts/training_cfg_muon_release/condition_bins.yml).

```bash
python code/scripts/train_diffusion.py code/configs/training_cfg_muon.yml --no_wandb
```

Logs and checkpoints are saved under `logs_diffusion/` by default. See the [training notes](REPRODUCIBILITY.md#training) for using a different split.

## Sampling

### Sample one pocket

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

The run writes sampling results under `tmp_eval/quick_eval_*/` and reconstructed molecules under its `sdfs_grouped/` subdirectory. See the [included sample outputs](samples/quick_evaluate_release/sdfs_grouped) for examples.

### Sample the CrossDocked2020 test set

After [data setup](REPRODUCIBILITY.md#data-setup), omit `--pocket_pdb` to sample from the test split. Use `--start_protein_idx` and `--num_proteins` to select a range of pockets. The [main benchmark command](REPRODUCIBILITY.md#main-benchmark) uses 100 pockets, 100 ligands per pocket, and CFG scale 5.

## Evaluation

### Evaluate generated molecules

[`quick_evaluate.py`](code/scripts/quick_evaluate.py) runs sampling, evaluation, and grouped SDF export in one command. To evaluate an existing directory of `result_*.pt` files, use [`evaluate_diffusion.py`](code/scripts/evaluate_diffusion.py).

```bash
python code/scripts/evaluate_diffusion.py /path/to/sampling_results \
  --docking_mode vina_dock \
  --protein_root data/test_set
```

This reports Vina scores, QED, SA, and diversity. Use `--docking_mode none` to skip docking.

### Evaluate 3D geometry

[`run_genbench_eval.py`](scripts/run_genbench_eval.py) evaluates grouped SDFs with a separate GenBench3D installation. See the [GenBench3D setup and command](REPRODUCIBILITY.md#genbench3d-geometry-metrics).

The paper describes the PoseCheck and pocket-permutation protocols in [Appendix N](https://arxiv.org/html/2609.08101v3#A14).

## Results

Selected CrossDocked2020 results from [Table 1 of the paper](https://arxiv.org/html/2609.08101v3#S4).

| Method | Vina Score median ↓ | Vina Dock median ↓ | Valid₃ᴅ (%) ↑ | Strain ↓ | Clash-free (%) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| TargetDiff | −6.30 | −7.91 | 78.4 | 306.0 | 86.41 |
| TAGMol (guided) | −7.77 | −8.69 | 58.6 | 457.4 | 93.21 |
| **PocketVE (CFG scale 5)** | **−7.89** | **−8.89** | **80.6** | **127.9** | **94.02** |

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

## Acknowledgements

The implementation builds on [TAGMol](https://github.com/MoleculeAI/TAGMol) and [TargetDiff](https://github.com/guanjq/targetdiff). The 3D benchmark uses [GenBench3D](https://github.com/bbaillif/genbench3d).

## License

See [LICENSE](LICENSE) and the [third-party notices](THIRD_PARTY.md) for code licensing and attribution.
