#!/bin/bash

set -eo pipefail

# The release sampling config uses data/... paths, so run from the prepared
# release workspace that contains the matching data and processed LMDB.
cd /tmp/pocketve_release_sweep.57M398
export PYTHONPATH=/shared/healthinfolab/phz24002/pocketVE-CM/PocketVE/code${PYTHONPATH:+:$PYTHONPATH}

# GPU35 exposes CUDA 11.6 through Environment Modules.
module load cuda/11.6
source /home/phz24002/anaconda3/etc/profile.d/conda.sh
conda activate targetdiff

CODE_ROOT=/shared/healthinfolab/phz24002/pocketVE-CM/PocketVE
PYTHON=/home/phz24002/anaconda3/envs/targetdiff/bin/python
CONFIG=$CODE_ROOT/samples/model_artifacts/training_cfg_muon_release/sampling.yml
CHECKPOINT=$CODE_ROOT/samples/model_artifacts/training_cfg_muon_release/checkpoints/354000.pt
PROTEIN_ROOT=/tmp/pocketve_release_sweep.57M398/data/test_set
BASE_OUT=$CODE_ROOT/evaluations/baseline_354k_gen_arcsin_b2p2_p0p57_cfg05_updated

echo "START baseline_354k gen_arcsin beta=2.2 p=0.57 cfg=0,5 checkpoint=$CHECKPOINT"
echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

for STEPS in 10 20 30 50 100; do
  for CFG in 0 5; do
    OUT_ROOT=${BASE_OUT}/steps${STEPS}_cfg${CFG}
    echo "START steps=${STEPS} cfg=${CFG} out=${OUT_ROOT}"
    "$PYTHON" "$CODE_ROOT/code/scripts/quick_evaluate.py" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --device cuda \
      --num_proteins 10 \
      --start_protein_idx 0 \
      --num_ligands_per_protein 10 \
      --batch_size 10 \
      --num_steps "$STEPS" \
      --cfg_scale "$CFG" \
      --cfg_strategy always \
      --time_scheduler gen_arcsin \
      --docking_mode vina_score \
      --protein_root "$PROTEIN_ROOT" \
      --tmp_root "$OUT_ROOT" \
      --exhaustiveness 16 \
      --sigma_pocket 0.0 \
      --veda_noise_injection_rate 0.4 \
      --veda_noise_injection_high_threshold 3.0 \
      --veda_noise_injection_low_threshold 0.0 \
      --export_grouped_sdf
    echo "DONE steps=${STEPS} cfg=${CFG}"
  done
done

echo "DONE baseline_354k gen_arcsin beta=2.2 p=0.57 cfg=0,5"
