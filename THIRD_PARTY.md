# Third-party resources

PocketVE builds on [TAGMol](https://github.com/MoleculeAI/TAGMol) and [TargetDiff](https://github.com/guanjq/targetdiff). See the [paper](https://arxiv.org/abs/2609.08101) for citations.

## Bundled code and scoring data

| Files | Source and notice |
| --- | --- |
| [`code/utils/reconstruct.py`](code/utils/reconstruct.py) | The file header identifies [liGAN's fitting code](https://github.com/mattragoza/liGAN/blob/master/fitting.py) and [GNU GPL v2.0](https://github.com/mattragoza/liGAN/blob/master/LICENSE). Follow the upstream terms for this component. |
| [`sascorer.py`](code/utils/evaluation/sascorer.py) and [`fpscores.pkl.gz`](code/utils/evaluation/fpscores.pkl.gz) | [RDKit's SA Score contribution](https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score) by Novartis Institutes for BioMedical Research. The BSD-style notice is reproduced in `sascorer.py`. |
| [`code/utils/warmup.py`](code/utils/warmup.py) | MIT-licensed code by Ildoo Kim. See its file header. |

## Benchmark data and external tools

The full CrossDocked2020 dataset and preparation artifacts are external inputs. See [the benchmark setup](REPRODUCIBILITY.md#data-setup) for the required files, and the upstream projects for license and citation terms. Vina and GenBench3D are external dependencies used for benchmarking.
