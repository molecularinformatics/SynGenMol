"""Reaction-SMARTS parsing and deterministic template application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class ReactionTemplate:
    """Validated two-reactant reaction SMARTS with a stable integer identifier."""

    identifier: int
    smarts: str
    reactant_queries: tuple[Chem.Mol, Chem.Mol]
    reaction: AllChem.ChemicalReaction

    @classmethod
    def from_smarts(cls, identifier: int, smarts: str) -> "ReactionTemplate":
        smarts = smarts.strip()
        if not smarts:
            raise ValueError("Reaction SMARTS cannot be empty")
        reaction = AllChem.ReactionFromSmarts(smarts)
        if reaction is None:
            raise ValueError(f"Invalid reaction SMARTS: {smarts!r}")
        if reaction.GetNumReactantTemplates() != 2:
            raise ValueError(
                f"SynGenMol currently supports exactly two reactants; got "
                f"{reaction.GetNumReactantTemplates()} for template {identifier}."
            )
        first = reaction.GetReactantTemplate(0)
        second = reaction.GetReactantTemplate(1)
        if first is None or second is None:
            raise ValueError(f"Reaction {identifier} has a missing reactant template")
        return cls(
            identifier=int(identifier),
            smarts=smarts,
            reactant_queries=(Chem.Mol(first), Chem.Mol(second)),
            reaction=reaction,
        )

    def matches_reactant(self, molecule: Chem.Mol, position: int) -> bool:
        if position not in (0, 1):
            raise ValueError(f"Reactant position must be 0 or 1, not {position}")
        return bool(molecule.HasSubstructMatch(self.reactant_queries[position], useChirality=True))

    def run(self, first: Chem.Mol, second: Chem.Mol) -> list[Chem.Mol]:
        """Apply the template and return the first valid product.

        RDKit reports one product tuple per way the template matches the
        reactant pair, so a single template application can be regio- or
        stereochemically ambiguous.  SynGenMol resolves that ambiguity by
        retaining only the first valid canonicalized product, which keeps one
        template application equal to one deterministic assembly step.  Branching
        over several *templates* that accept the same pair is handled by the
        generator instead; see ``models.gpt.generate_trajectory_branches``.

        The return type stays a list so that callers can treat an unproductive
        template application as an empty result.
        """
        try:
            product_sets = self.reaction.RunReactants((first, second))
        except Exception:
            return []
        if not product_sets or not product_sets[0]:
            return []

        product = Chem.Mol(product_sets[0][0])
        try:
            Chem.SanitizeMol(product)
            product = Chem.RemoveHs(product)
        except Exception:
            return []
        return [product]


def load_reaction_templates(path: str | Path) -> list[ReactionTemplate]:
    """Load non-comment reaction SMARTS lines and assign identifiers by line order."""
    path = Path(path)
    templates: list[ReactionTemplate] = []
    for source_line, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        smarts = raw_line.strip()
        if not smarts or smarts.startswith("#"):
            continue
        try:
            templates.append(ReactionTemplate.from_smarts(len(templates), smarts))
        except ValueError as exc:
            raise ValueError(f"Invalid reaction template at {path}:{source_line}: {exc}") from exc
    if not templates:
        raise ValueError(f"No reaction SMARTS were found in {path}")
    return templates


def write_reaction_templates(templates: Iterable[ReactionTemplate], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(template.smarts for template in templates) + "\n", encoding="utf-8")
