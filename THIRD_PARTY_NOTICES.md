# Third-party notices

## PED-derived oracle adapters

The following SynGenMol modules are adapted integration layers based on the public **PED** repository maintained by Molecular Informatics:

- `src/syngenmol/oracles/ped_geodiff.py`
- `src/syngenmol/oracles/ped_molformer.py`

Upstream source:

```text
https://github.com/molecularinformatics/PED/tree/main
```

SynGenMol redistributes only its adapted adapter code. It does not redistribute PED, GeoDiff, MoLFormer, pretrained weights, checkpoints, model caches, or other third-party model assets. Users must obtain those assets independently, comply with the relevant licenses and terms, and cite the PED work and underlying models as required by their selected installation.

PED is distributed under the MIT License. Its copyright notice and permission notice are reproduced below as that license requires:

```text
MIT License

Copyright (c) 2026 molecularinformatics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

GeoDiff, MoLFormer, and any model weights obtained through a PED installation carry their own separate licenses, which this notice does not cover.

## Reaction templates

`data/reactions/syngenmol_reactions_v1.smi` is derived from the reaction set published with **SynFormer**:

> Wenhao Gao, Shitong Luo, Connor W. Coley. *Generative AI for navigating synthesizable chemical space.* Proceedings of the National Academy of Sciences 122(41), e2415665122, 2025.

```text
https://doi.org/10.1073/pnas.2415665122
https://github.com/wenhao-gao/synformer
```

SynGenMol redistributes the 90 two-component reaction SMARTS of that set, each written in both reactant orders, and does not redistribute SynFormer source code, checkpoints, or building-block data. Users must cite the work above when they use these templates. See [`data/reactions/REACTION_TEMPLATES.md`](data/reactions/REACTION_TEMPLATES.md) for details.

SynFormer is distributed under the Apache License, Version 2.0 (`https://www.apache.org/licenses/LICENSE-2.0`). The reaction SMARTS redistributed here are a derived work of that repository, retained with the attribution and citation above as Section 4 of that license requires. The upstream repository provides no `NOTICE` file, so no additional notice text is required.

## Unbundled external scoring tools

[Roshambo2](https://github.com/molecularinformatics/roshambo2) is an external option that users may install independently. SynGenMol does not redistribute Roshambo2 source code, dependencies, model assets, or a Roshambo2-specific adapter. A user who selects it must follow its upstream installation instructions and maintain a project-specific `MolecularOracle` integration.

## User-provided external scorers

SynGenMol exposes a generic `MolecularOracle` interface. Users may independently integrate an external scoring tool through that interface. Such user integrations and their associated dependencies are outside the scope of this repository unless explicitly documented here.
