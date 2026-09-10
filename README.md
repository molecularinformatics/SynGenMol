# SynGenMol

SynGenMol is a goal-directed framework for synthesizable molecular generation. It assembles products from building blocks and reaction SMARTS templates, masks chemically incompatible choices during decoding, and optimizes a compact autoregressive policy with PPO against a task-specific molecular oracle.

This release contains two runnable PED-GeoDiff demonstrations that share one code path, one reaction-template asset, and one anonymized building-block pool. Both ship their published results so the numbers in the documentation can be checked without a GPU.

**Unconstrained generation** is the baseline workflow: 2,000 building blocks with free choice of the first building block. Its published run is in [`outputs/syngenmol_demo_unconstraint/results/`](outputs/syngenmol_demo_unconstraint/results/).

**Warhead-protected generation** adds a first-token constraint that forces every route to open with a Boc-protected primary amine, so each product carries exactly one maskable warhead site. Its published run is in [`outputs/syngenmol_demo_warhead_constraint/results/`](outputs/syngenmol_demo_warhead_constraint/results/).

Both are documented in [`docs/demonstrations.md`](docs/demonstrations.md).

The constrained demonstration is the same three commands as the unconstrained one plus one configuration key, `initial_building_block_constraint`, which intersects the reaction-compatible first-position tokens with a named anchor set during both PPO rollouts and generation, so the constraint cannot be violated by sampling.

In both demonstrations the policy learns to sample higher-scoring products. Each figure below plots, over the 500 PPO updates of the published run, the mean PED-GeoDiff score of the 64-trajectory rollout batch and the highest score in that batch; the two share axis limits and can be read against each other.

![PPO score curve for the unconstrained demonstration: batch mean and batch maximum PED-GeoDiff score over 500 PPO updates](https://github.com/molecularinformatics/SynGenMol/raw/main/docs/figures/ppo_curve_unconstrained.png)

![PPO score curve for the warhead-protected demonstration: batch mean and batch maximum PED-GeoDiff score over 500 PPO updates](https://github.com/molecularinformatics/SynGenMol/raw/main/docs/figures/ppo_curve_warhead_constrained.png)

Vendor catalogs, trained checkpoints, full PPO metric histories, prepared search spaces, and third-party model assets are not included. Prepared search spaces regenerate deterministically from the shipped inputs, so published token IDs match on a re-run.

## Quick start

Run from the repository root on a GPU-enabled machine. The supplied configuration files hold every setting the example needs; nothing has to be tuned.

### 1. Install

```bash
conda create -n syngenmol python=3.10 -y
conda activate syngenmol
pip install -e ".[training,ped-geodiff]"
```

### 2. Install PED and point to a GeoDiff checkpoint

The `ped_geodiff.py` adapter is adapted from the public [PED repository](https://github.com/molecularinformatics/PED/tree/main). PED, GeoDiff, and their model assets stay user-managed.

```bash
export PED_ROOT=/path/to/PED
git clone https://github.com/molecularinformatics/PED.git "$PED_ROOT"

# Install PED's GeoDiff dependencies and obtain a checkpoint by following the
# current PED instructions, then expose both locations to SynGenMol.
export PYTHONPATH="$PED_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SYNGENMOL_GEODIFF_CHECKPOINT=/absolute/path/to/drugs_default.pt
test -f "$SYNGENMOL_GEODIFF_CHECKPOINT"
```

PED commonly places the checkpoint at `$PED_ROOT/geodiff/log/model/checkpoints/drugs_default.pt`; use the path your installation actually provides.

### 3. Prepare the design space

```bash
syngenmol prepare-search-space \
  --input data/example/syngenmol_demo_unconstraint_building_blocks.csv \
  --id-column name \
  --smiles-column smiles \
  --reaction-templates data/reactions/syngenmol_reactions_v1.smi \
  --output-dir outputs/syngenmol_demo_unconstraint/search_space
```

### 4. Run value warm-up and PPO

```bash
syngenmol train --config configs/syngenmol_demo_unconstraint_train.yaml
```

10 terminal-value MSE warm-up epochs, then 500 PPO updates at 64 trajectories each. The included warm-up table is already sampled and GeoDiff-scored, so no initial trajectory sampling is needed.

### 5. Generate and score candidates

```bash
syngenmol generate --config configs/syngenmol_demo_unconstraint_generate.yaml
```

A re-run writes its ranked output to `outputs/syngenmol_demo_unconstraint/generated_molecules_geodiff.csv`, leaving the published `outputs/syngenmol_demo_unconstraint/results/` untouched for comparison.

Input details, the full settings list, and the reference PPO learning curves are in [`docs/demonstrations.md`](docs/demonstrations.md).

### 6. Optional: run the constrained variant

The warhead-protected demonstration is laid out identically: its inputs sit beside the unconstrained ones in `data/example/`, its two configurations in `configs/`, its published run in `outputs/syngenmol_demo_warhead_constraint/results/`. It reuses steps 1–2, then substitutes its own three commands, which are listed in [`docs/demonstrations.md`](docs/demonstrations.md) along with its 2,200-block pool and 678-member anchor set.

## Policy size

Both demonstrations run a deliberately small policy so a full run is cheap to reproduce. The reported results use a larger one.

| | Demonstrations | Reported results |
|---|---|---|
| `model.n_embd` | 64 | 512 |
| `model.n_layer` | 2 | 8 |
| `model.n_head` | 2 | 8 |
| `ppo.batch_size` | 64 | 512 |
| `ppo.learning_rate` | `1.0e-4` | `1.0e-5` |
| `ppo.max_reactions` | 1 | 1 or 2 |

At the demonstration size a complete warm-up plus 500 PPO updates takes about 37 minutes on one A100. Scaling up is a change to the `model` and `ppo` blocks of the training configuration and nothing else; the code path, reaction handling, and constraint mechanism are identical. Larger policies need proportionally more oracle calls per update, which is what sets the wall-clock cost.

## Included files

```text
# Shared assets
data/reactions/syngenmol_reactions_v1.smi                               180 ordered reaction SMARTS records
docs/figures/                                                           the two published PPO score curves and the script that draws them
docs/demonstrations.md                                                  inputs, settings, workflow, and reference results for both demonstrations
docs/restore_warhead.py                                                 post-generation Boc removal and crotonamide warhead installation

# Unconstrained demonstration
data/example/syngenmol_demo_unconstraint_building_blocks.csv            2,000 anonymized building blocks
data/example/syngenmol_demo_unconstraint_warmup_trajectories.csv        4,000 fixed, scored warm-up trajectories
data/example/syngenmol_demo_unconstraint_metadata.json                  construction record and checksums
configs/syngenmol_demo_unconstraint_train.yaml                          warm-up and PPO configuration
configs/syngenmol_demo_unconstraint_generate.yaml                       candidate-generation configuration
outputs/syngenmol_demo_unconstraint/results/                            published run: candidates, score curve, summary

# Warhead-constrained demonstration
data/example/syngenmol_demo_warhead_constraint_building_blocks.csv      2,200 anonymized building blocks
data/example/syngenmol_demo_warhead_constraint_warmup_trajectories.csv  4,000 fixed, scored constrained warm-up trajectories
data/example/syngenmol_demo_warhead_constraint_anchor_names.txt         678 permitted first-position anchors
data/example/syngenmol_demo_warhead_constraint_anchors.csv              the same anchors with SMILES and provenance
data/example/syngenmol_demo_warhead_constraint_metadata.json            construction record and checksums
configs/syngenmol_demo_warhead_constraint_train.yaml                    constrained warm-up and PPO configuration
configs/syngenmol_demo_warhead_constraint_generate.yaml                 constrained candidate-generation configuration
outputs/syngenmol_demo_warhead_constraint/results/                      published run: candidates, score curve, summary
```

The reaction templates are taken from the reaction set of Gao et al., [Generative AI for navigating synthesizable chemical space](https://doi.org/10.1073/pnas.2415665122) (SynFormer), which mixes one-, two-, and three-component reactions; SynGenMol uses its 90 bi-molecular reactions. Because a route is generated autoregressively, a template is usable only when the building block picked first matches its *first* reactant pattern, so every reaction is recorded in both reactant orders, `R1.R2>>product` and `R2.R1>>product`, letting either partner open the route. That duplication, not a larger reaction set, is why the file holds 180 ordered records for 90 reactions. [`data/reactions/REACTION_TEMPLATES.md`](data/reactions/REACTION_TEMPLATES.md) has the details.

## Notes for other projects

SynGenMol trains a new compact policy for each building-block library, reaction set, constraint set, and scoring objective. No large-scale route pretraining is required, but a task-specific search space and warm-up table must be prepared before PPO.

### Prepare a building-block catalog

SynGenMol does not redistribute Enamine or other commercial catalogs. Request access from the [Enamine Building Blocks Catalog](https://enamine.net/building-blocks/building-blocks-catalog), download the authorized file outside this repository, and comply with its license. US Stock is a practical starting point; another licensed Enamine catalog or a project-curated subset works equally well.

```bash
# Keep vendor data outside the source checkout and out of version control.
export SYNGENMOL_DATA_DIR="$HOME/syngenmol_data"
mkdir -p "$SYNGENMOL_DATA_DIR/enamine"
```

CSV and TSV input needs two columns, a stable unique identifier and SMILES:

```text
name,smiles
project-bb-000001,CC(=O)N...
```

```bash
syngenmol prepare-search-space \
  --input "$SYNGENMOL_DATA_DIR/enamine/project_building_blocks.csv" \
  --id-column name \
  --smiles-column smiles \
  --reaction-templates data/reactions/syngenmol_reactions_v1.smi \
  --output-dir outputs/my_project/search_space
```

SDF and compressed SDF catalogs work directly: drop `--smiles-column`, set `--id-column` to the SDF property holding the catalog identifier, and RDKit derives the SMILES during preprocessing.

Curate the building blocks and reaction SMARTS before training when project knowledge rules out chemistry that should not be explored; the supplied template file is a starting asset, not a mandatory design space. Do not commit licensed catalogs, supplier identifiers, or prepared search-space outputs to a public repository unless the terms permit redistribution.

### Oracle options

| Oracle | Upstream resource | SynGenMol support | Installation and user-supplied assets |
|---|---|---|---|
| ECFP Tanimoto similarity | Built into RDKit and SynGenMol | Included as `tanimoto` | No external model needed. Supply a reference SMILES and fingerprint settings in the oracle configuration. |
| PED-GeoDiff | [PED](https://github.com/molecularinformatics/PED/tree/main) | Optional `ped_geodiff` adapter | `pip install -e ".[training,ped-geodiff]"`, then install PED/GeoDiff and obtain a checkpoint independently, as in [Quick start](#2-install-ped-and-point-to-a-geodiff-checkpoint). |
| PED-MolFormer | [PED](https://github.com/molecularinformatics/PED/tree/main), [MoLFormer model card](https://huggingface.co/ibm/MoLFormer-XL-both-10pct) | Optional `ped_molformer` adapter | `pip install -e ".[training,ped-molformer]"`, then download or cache the selected MoLFormer model independently. |
| Roshambo2 | [Roshambo2](https://github.com/molecularinformatics/roshambo2) | Not bundled, not exposed by the configuration CLI | Install from upstream and implement a project-specific `MolecularOracle` adapter through the Python API. |
| Other external tools | User-selected software | Generic Python interface | Implement `MolecularOracle.score_smiles`, returning one finite higher-is-better score per input SMILES, in order. |

The configuration CLI recognizes only `tanimoto`, `ped_geodiff`, and `ped_molformer`. Anything else is used through the Python API or a locally maintained extension to `build_oracle`:

```python
from collections.abc import Sequence
from syngenmol.oracles import MolecularOracle


class ProjectOracle(MolecularOracle):
    def score_smiles(self, smiles: Sequence[str]) -> list[float]:
        # Call the independently installed scoring tool here.
        # Return one finite, higher-is-better value per input SMILES.
        return [0.0 for _ in smiles]
```

The GeoDiff example uses `embedding_mode: "3D"`, Euclidean embedding distance, and the sigmoid transformation shown in `configs/syngenmol_demo_unconstraint_train.yaml`. PED-MolFormer loads a Transformers-compatible model by `model_id` (default [`ibm/MoLFormer-XL-both-10pct`](https://huggingface.co/ibm/MoLFormer-XL-both-10pct)) and caches it under `SYNGENMOL_MOLFORMER_CACHE`; run once with network access to populate that cache, or set `offline: true` once the files are local.

Test any external oracle on a small SMILES table before launching PPO. Its per-molecule latency sets the wall-clock cost of optimization, because every PPO update scores freshly generated trajectories.

## Further reading

| Document | Contents |
|---|---|
| [`docs/demonstrations.md`](docs/demonstrations.md) | Both demonstrations: inputs, settings, workflow, reference results |
| [`data/reactions/REACTION_TEMPLATES.md`](data/reactions/REACTION_TEMPLATES.md) | Reaction-template provenance and the reactant-order duplication |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Third-party scope and adapted-module provenance |

## License and citation

SynGenMol is released under the MIT License; see [`LICENSE`](LICENSE). Third-party components are not relicensed and keep their own terms, which are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). [`CITATION.cff`](CITATION.cff) holds the software citation and will be updated with the method paper once it is published.
