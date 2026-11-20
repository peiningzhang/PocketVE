# Reproducing PocketVE

Use the environment and `PYTHONPATH` from the [environment setup](README.md#environment-setup). Run the commands from the repository root.

## Data setup

The released configurations expect the 10 Å pocket version of CrossDocked2020 and a split whose indices match the property-annotated LMDB. The [CrossDocked2020 documentation](https://github.com/gnina/models/tree/master/data/CrossDocked2020) and [TAGMol data instructions](https://github.com/MoleculeAI/TAGMol#data-and-checkpoints) describe the upstream resources.

| Path | Purpose |
| --- | --- |
| `data/crossdocked_v1.1_rmsd1.0_pocket10_processed_dock_guide_final.lmdb` | Preprocessed receptor-pocket and ligand pairs with property annotations. The loader derives this filename from `data.path` in the YAML. |
| `data/crossdocked_pocket10_pose_split_dock_guide.pt` | Matching train/test index split. Its test split contains the 100 benchmark pockets. |
| `data/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl` | Annotated index used when building the LMDB from raw pocket files. An existing processed LMDB is sufficient for sampling. |
| `data/test_set/` | Per-pocket reference ligands and receptor PDBs used for docking and GenBench3D. |

The repository includes the checkpoint, condition-bin specification, and selected generated samples. Obtain the matching LMDB and split separately. A split from a differently ordered LMDB can silently select different pockets. To use another layout, update `data.path`, `data.split`, and `data.test_path` in the config you pass on the command line.

The reference split has SHA-256 `124077745a21d9bc95a69c1dce50856d70508213f3575bb17d10e3a36af3fae8`.

## Main benchmark

The main paper setting uses the released checkpoint, 100 test pockets, 100 samples per pocket, 100 denoising steps, the generalized-arcsine schedule, and constant CFG at scale 5. Use `vina_dock` to collect Vina Score, Vina Min, and Vina Dock.

```bash
model_dir=samples/model_artifacts/training_cfg_muon_release

python code/scripts/quick_evaluate.py \
  --config "$model_dir/sampling.yml" \
  --checkpoint "$model_dir/checkpoints/last.pt" \
  --num_proteins 100 \
  --num_ligands_per_protein 100 \
  --batch_size 10 \
  --num_steps 100 \
  --time_scheduler gen_arcsin \
  --cfg_scale 5.0 \
  --docking_mode vina_dock \
  --protein_root data/test_set \
  --export_grouped_sdf \
  --sdf_grouped_dir outputs/pocketve/sdfs_grouped
```

See [Section 4 and the appendices](https://arxiv.org/html/2609.08101v3) for the full comparison and aggregation protocol.

### GenBench3D geometry metrics

Install [GenBench3D](https://github.com/bbaillif/genbench3d) in its own environment and configure its `config/default.yaml` for the LigBoundConf reference and local paths. Set `GENBENCH_DIR` to the checkout and `GENBENCH_PYTHON` to that environment's Python executable.

```bash
python scripts/run_genbench_eval.py \
  --gen_dir outputs/pocketve/sdfs_grouped \
  --test_set_dir data/test_set \
  --gb3d_dir "$GENBENCH_DIR" \
  --genbench_python "$GENBENCH_PYTHON" \
  --do_conf_analysis
```

GenBench3D writes per-pocket JSON and an aggregate file under `sdfs_grouped/genbench_results/`. The optional `--no_vina` flag on the wrapper requires a GenBench3D checkout with the corresponding flag in `sb_benchmark_mols.py`. With the upstream runner, use its documented Vina setup.

## Training

The [Muon training config](code/configs/training_cfg_muon.yml) uses the property-annotated training split. Its condition-bin path points to the published [bin specification](samples/model_artifacts/training_cfg_muon_release/condition_bins.yml). To use a different training split, generate a matching specification with `code/scripts/data_preparation/build_condition_bins.py` and update `data.transform.condition_bins_path`.

```bash
python code/scripts/train_diffusion.py code/configs/training_cfg_muon.yml --no_wandb
```

Training logs and checkpoints are written under `logs_diffusion/` by default.
