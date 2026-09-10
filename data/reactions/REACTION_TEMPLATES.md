# SynGenMol Reaction Templates

`syngenmol_reactions_v1.smi` holds the bimolecular reaction templates used in this release. Each nonempty line is an RDKit reaction SMARTS record.

## Source

The templates come from the reaction set published with SynFormer:

> Wenhao Gao, Shitong Luo, Connor W. Coley. *Generative AI for navigating synthesizable chemical space.* Proceedings of the National Academy of Sciences 122(41), e2415665122, 2025. <https://doi.org/10.1073/pnas.2415665122>

That set mixes one-, two-, and three-component reactions. SynGenMol assembles products from pairs of building blocks, so it keeps the 90 two-component reactions and discards the rest.

## Both reactant orders

A route is generated autoregressively, so the first building block is chosen before the second, and a template is usable only when that first building block matches its *first* reactant pattern. Writing each reaction once would let the reactant order in the SMARTS decide which partner may open a route. Each reaction is therefore recorded in both orders, `R1.R2>>product` and `R2.R1>>product`, and dynamic masking admits either partner as the opening token. This is why the file holds 180 records for 90 reactions. A few records are repeated or missing their reversed form, inherited from the upstream set; a repeated template is redundant rather than harmful, because the duplicate branch yields the same product and the highest-scoring-product-per-route reduction collapses it.

## Preparation

Search-space preparation validates every record with RDKit, determines building-block compatibility for each reactant position by SMARTS substructure matching, and omits templates with no compatible building block in the supplied catalog. The prepared search space records the active template subset in its manifest.

These templates define virtual, template-level feasibility only. They do not establish experimental reaction outcome, yield, selectivity, purification, or route suitability.
