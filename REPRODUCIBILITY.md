# Reproducing PocketVE

Use the environment and `PYTHONPATH` from the [environment setup](README.md#environment-setup). Run the commands from the repository root.

## Data setup

For sampling and evaluation with `last.pt`, download the standard CrossDocked2020 pocket10 LMDB, its paired split, and the test reference structures from the [LandFlow data mirror](https://huggingface.co/datasets/NYCU-MLLab/Physics-Inspired-Landscape-Learning-and-Flow-Matching-data/tree/8394343d0a38690d83d7dfb850790d1896763c2c/data). The 100 test entries match PocketVE's original benchmark in order and model inputs. The receptor PDBs and reference ligand SDFs also match byte for byte.

```bash
mkdir -p data
(
  set -e
  cd data
  source_url=https://huggingface.co/datasets/NYCU-MLLab/Physics-Inspired-Landscape-Learning-and-Flow-Matching-data/resolve/8394343d0a38690d83d7dfb850790d1896763c2c/data
  curl -fLO "$source_url/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
  curl -fLO "$source_url/crossdocked_pocket10_pose_split.pt"
  curl -fLO "$source_url/test_set.tar.gz"
  sha256sum -c <<'EOF'
90e3399529420f2ecb74bc47744f10291e27fc20a325545180d7f9ba52fec68f  crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
fc2e32512985e1f01f70b95ca58c0f5abf1ab0a6a7985cabcfedd969ac730bc4  crossdocked_pocket10_pose_split.pt
c93289d4b1522f168872d53cd927bf3982e721a93cd5978bdb574a845ae3d71b  test_set.tar.gz
EOF
  tar -xzf test_set.tar.gz --wildcards 'test_set/*/*_rec.pdb' 'test_set/*/*.sdf'
)
```

Generate a sampling config that selects the standard LMDB loader and its matching split. The rest of the released sampling configuration is carried forward.

```bash
python - <<'PY'
from pathlib import Path
import yaml

source = Path("samples/model_artifacts/training_cfg_muon_release/sampling.yml")
config = yaml.safe_load(source.read_text())
config["data"]["name"] = "pl"
config["data"]["split"] = "data/crossdocked_pocket10_pose_split.pt"
Path("data/sampling_benchmark.yml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

Keep the LMDB and split paired because the split indices follow the LMDB's key order. See the [CrossDocked2020 data license](https://bits.csb.pitt.edu/files/crossdock2020/LICENSE).

## Main benchmark

The main paper setting uses the released checkpoint, 100 test pockets, 100 samples per pocket, 100 denoising steps, the generalized-arcsine schedule, and constant CFG at scale 5. Use `vina_dock` to collect Vina Score, Vina Min, and Vina Dock.

```bash
model_dir=samples/model_artifacts/training_cfg_muon_release

python code/scripts/quick_evaluate.py \
  --config data/sampling_benchmark.yml \
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

Training from scratch requires Vina, QED, and SA labels. The [Muon training config](code/configs/training_cfg_muon.yml) expects the following paired files under `data/`:

- `crossdocked_v1.1_rmsd1.0_pocket10_processed_dock_guide_final.lmdb`
- `crossdocked_pocket10_pose_split_dock_guide.pt`

The public benchmark download above supports sampling and evaluation with the pretrained checkpoint. The property-annotated training LMDB and its split are separate assets and do not currently have a public download link. The training config points to the included [bin specification](samples/model_artifacts/training_cfg_muon_release/condition_bins.yml). To use a different training split, generate a matching specification with `code/scripts/data_preparation/build_condition_bins.py` and update `data.transform.condition_bins_path`.

```bash
python code/scripts/train_diffusion.py code/configs/training_cfg_muon.yml --no_wandb
```

Training logs and checkpoints are written under `logs_diffusion/` by default.
