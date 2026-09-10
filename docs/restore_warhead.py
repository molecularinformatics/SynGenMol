"""Convert Boc-protected warhead-constrained products to their warhead form.

The warhead-constrained demonstration optimizes and scores protected
intermediates: every route opens with a Boc-protected primary amine, so each
generated product carries the warhead site masked.  This script performs the
post-generation chemistry described in the manuscript appendix -- Boc
deprotection regenerates the free primary amine, then crotonylation installs
the crotonamide warhead:

    R-NH-Boc  ->  R-NH2  ->  R-NH-C(=O)/C=C/C

The anchor-derived nitrogen is located by atom-map tracking rather than by
substructure guessing: the Boc nitrogen of the first building block is tagged,
the recorded reaction template is re-run on the two recorded building blocks,
and the tag is followed into the product.  Only that nitrogen receives the
warhead.  Any further Boc groups contributed by the second building block are
deprotected to free amines without acylation.

Routes whose template acylates the carbamate nitrogen itself leave no N-H to
deprotect, so no warhead can be installed; those rows are reported as
`deprotected_only` and keep their deprotected structure.

Usage:
    python docs/restore_warhead.py \
        --input        outputs/syngenmol_demo_warhead_constraint/results/generated_molecules_geodiff.csv \
        --building-blocks data/example/syngenmol_demo_warhead_constraint_building_blocks.csv \
        --templates    outputs/syngenmol_demo_warhead_constraint/search_space/reaction_templates.smi
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# The Boc-protected N-H of an anchor building block.
BOC_NH = Chem.MolFromSmarts("[NH;$(NC(=O)OC(C)(C)C);!$(n)]")
# Deprotection of the tracked anchor nitrogen, and of any remaining Boc group.
DEBOC_TRACKED = AllChem.ReactionFromSmarts("[NH:999]C(=O)OC(C)(C)C>>[NH2:999]")
# Acidic deprotection removes every Boc carbamate, including those on a
# nitrogen with no N-H (an N-Boc ring contributed by the partner block).
DEBOC_ANY = AllChem.ReactionFromSmarts("[N:1]C(=O)OC(C)(C)C>>[N:1]")
# Crotonylation of the tracked amine: E-crotonamide.
CROTONYLATE = AllChem.ReactionFromSmarts("[NH2:999]>>[NH:999]C(=O)/C=C/C")

TRACK = 999


def _sanitized(mol):
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _clear_maps(mol):
    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        atom.SetAtomMapNum(0)
    return editable.GetMol()


def _smiles(mol):
    try:
        return Chem.MolToSmiles(_clear_maps(mol), isomericSmiles=True)
    except Exception:
        return None


def _deprotect_all(mol):
    """Remove every remaining N-H Boc group."""
    for _ in range(5):
        products = DEBOC_ANY.RunReactants((mol,))
        if not products:
            break
        nxt = _sanitized(products[0][0])
        if nxt is None:
            break
        mol = nxt
    return mol


def _apply(reaction, mol):
    products = reaction.RunReactants((mol,))
    if not products:
        return None
    return _sanitized(products[0][0])


def _canonical(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return None if mol is None else Chem.MolToSmiles(mol, isomericSmiles=True)


def restore(product_smiles, anchor_smiles, partner_smiles, template_smarts):
    """Return (smiles, status) for one generated product.

    The recorded template is re-run on the two recorded building blocks and the
    resulting product must reproduce the recorded product SMILES exactly; only
    then is the traced nitrogen trusted.  A template may match a building block
    in more than one way, so every product set is checked.
    """
    anchor = Chem.MolFromSmiles(anchor_smiles)
    partner = Chem.MolFromSmiles(partner_smiles)
    if anchor is None or partner is None:
        return None, "unparsable_building_block"

    matches = anchor.GetSubstructMatches(BOC_NH)
    if not matches:
        return None, "anchor_not_boc_protected"

    tagged = Chem.RWMol(anchor)
    tagged.GetAtomWithIdx(matches[0][0]).SetAtomMapNum(TRACK)

    try:
        reaction = AllChem.ReactionFromSmarts(template_smarts)
        product_sets = reaction.RunReactants((tagged.GetMol(), partner))
    except Exception:
        return None, "template_failed"
    if not product_sets:
        return None, "template_no_product"

    target = _canonical(product_smiles)
    reproduced = False
    for candidate_set in product_sets:
        candidate = _sanitized(candidate_set[0])
        if candidate is None or _smiles(candidate) != target:
            continue
        reproduced = True
        if not any(atom.GetAtomMapNum() == TRACK for atom in candidate.GetAtoms()):
            # The template reacted at the carbamate nitrogen itself: the masked
            # warhead site is consumed, so no warhead can be installed here.
            continue
        deprotected = _apply(DEBOC_TRACKED, candidate)
        if deprotected is None:
            continue
        deprotected = _deprotect_all(deprotected)
        acylated = _apply(CROTONYLATE, deprotected)
        if acylated is None:
            return _smiles(deprotected), "deprotected_only"
        return _smiles(acylated), "warhead_installed"

    return None, "warhead_site_consumed" if reproduced else "product_not_reproduced"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--building-blocks", required=True, type=Path)
    parser.add_argument("--templates", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input

    smiles_by_name = (
        pd.read_csv(args.building_blocks).set_index("name")["smiles"].to_dict()
    )
    templates = [
        line.strip()
        for line in args.templates.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    frame = pd.read_csv(args.input)

    restored, statuses = [], []
    for row in frame.itertuples(index=False):
        names = str(row.name).split()
        reaction_ids = ast.literal_eval(str(row.reaction_ids))
        if len(names) != 2 or len(reaction_ids) != 1:
            restored.append(None)
            statuses.append("unsupported_route_length")
            continue
        anchor, partner = (smiles_by_name.get(n) for n in names)
        index = int(reaction_ids[0])
        if anchor is None or partner is None or not 0 <= index < len(templates):
            restored.append(None)
            statuses.append("missing_route_input")
            continue
        smiles, status = restore(row.product_smiles, anchor, partner, templates[index])
        restored.append(smiles)
        statuses.append(status)

    frame["warhead_product_smiles"] = restored
    frame["warhead_status"] = statuses
    frame.to_csv(output, index=False)

    total = len(frame)
    print(f"Wrote {output} ({total:,} rows)")
    for status, count in frame["warhead_status"].value_counts().items():
        print(f"  {status:28s} {count:6,d}  ({count / total * 100:5.1f}%)")


if __name__ == "__main__":
    main()
