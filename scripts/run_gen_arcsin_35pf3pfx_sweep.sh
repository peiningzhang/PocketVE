#!/bin/bash
#SBATCH --job-name=pocketve_genarc_sweep
#SBATCH --partition=priority-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=28-00:00:00
#SBATCH --exclude=gtx[02-03,05,15,07-21]
#SBATCH --constraint=a100|l40
#SBATCH --output=/shared/healthinfolab/phz24002/pocketVE-CM/PocketVE/job_logs/pocketve_gen_arcsin_35pf3pfx_%j.out
#SBATCH --error=/shared/healthinfolab/phz24002/pocketVE-CM/PocketVE/job_logs/pocketve_gen_arcsin_35pf3pfx_%j.err

set -eo pipefail

cd /shared/healthinfolab/phz24002/pocketVE-CM/PocketVE
export PYTHONPATH=/shared/healthinfolab/phz24002/pocketVE-CM/PocketVE/code${PYTHONPATH:+:$PYTHONPATH}
# gpu29 exposes CUDA 11.6 (not 11.7) through Environment Modules.
module load cuda/11.6
source /home/phz24002/anaconda3/etc/profile.d/conda.sh
conda activate targetdiff

PYTHON=/home/phz24002/anaconda3/envs/targetdiff/bin/python
CONFIG=code/configs/training_cfg_muon_sct.yml
CHECKPOINT=/shared/healthinfolab/phz24002/pocketVE-CM/PocketVE/logs_diffusion/training_cfg_muon_sct_2026_09_01__03_07_49_sct_coord0.1_atom0.01_gpu29/checkpoints/last.pt
PROTEIN_ROOT=/shared/healthinfolab/phz24002/TAGMol/data/test_set
BASE_OUT=/shared/healthinfolab/phz24002/pocketVE-CM/PocketVE/evaluations/35pf3pfx_gen_arcsin_steps_cfg_vina

echo "START gen_arcsin beta=2.2 p=0.57 checkpoint=$CHECKPOINT"
echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

for STEPS in 10 20 30 50 100; do
  for CFG in 0 5; do
    OUT_ROOT=${BASE_OUT}/steps${STEPS}_cfg${CFG}
    echo "START steps=${STEPS} cfg=${CFG} out=${OUT_ROOT}"
    "$PYTHON" code/scripts/quick_evaluate.py \
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

echo "DONE gen_arcsin beta=2.2 p=0.57"
