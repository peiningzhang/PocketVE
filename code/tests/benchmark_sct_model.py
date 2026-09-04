"""Small synthetic benchmark for the full SCT loss path."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.molopt_score_model import ScorePosNet3D
from utils.misc import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda')
    model = ScorePosNet3D(
        config.model, protein_atom_feature_dim=27, ligand_atom_feature_dim=13
    ).to(device).train()

    graphs, protein_nodes, ligand_nodes = 2, 300, 30
    batch_protein = torch.arange(graphs, device=device).repeat_interleave(protein_nodes)
    batch_ligand = torch.arange(graphs, device=device).repeat_interleave(ligand_nodes)
    inputs = dict(
        protein_pos=torch.randn(graphs * protein_nodes, 3, device=device),
        protein_v=torch.randn(graphs * protein_nodes, 27, device=device),
        batch_protein=batch_protein,
        ligand_pos=torch.randn(graphs * ligand_nodes, 3, device=device),
        ligand_v=torch.randint(13, (graphs * ligand_nodes,), device=device),
        batch_ligand=batch_ligand,
        vina_bin=torch.full((graphs,), 4, device=device),
        qed_bin=torch.full((graphs,), 4, device=device),
        sa_bin=torch.full((graphs,), 4, device=device),
        pocket_noise_mode=config.train.pocket_noise_mode,
        pos_noise_std=float(config.train.pos_noise_std),
        pocket_noise_sigma_coeff=float(config.train.pocket_noise_sigma_coeff),
        pocket_noise_max=float(config.train.pocket_noise_max),
        perturb_condition_only=bool(config.model.perturb_condition_only),
        global_step=2000,
    )

    elapsed = []
    for step in range(args.repeats + 1):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.get_diffusion_loss(**inputs)
        result['loss'].backward()
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        if step:
            elapsed.append(duration)
        if not torch.isfinite(result['loss']):
            raise RuntimeError('Non-finite benchmark loss')

    print(f'mean_step_seconds={sum(elapsed) / len(elapsed):.6f}')
    print(f'iterations={elapsed}')
    print(f'peak_memory_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}')


if __name__ == '__main__':
    main()
