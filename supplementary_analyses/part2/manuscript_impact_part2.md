# Paper 4 part 2: manuscript impact (21 Nov 2026)
Applied in the new copy `documents_v2/paper4/paper4_ojcoms_overleaf_20261121_v6/` (v5 untouched; built by scripts/p4_rev2_build_v6.py and p4_rev2_fix_v6.py; not compiled, no TeX engine here):
- New subsection ESA3 (further open-set scores) with Table tab:scores: T3 feature-space scores, U10 logit scores and U2 head disagreement (numbers from part2/pooled_table_for_manuscript.csv and T3/paired_vs_energy.csv). It supports the existing statement that the energy score is uneven on the flood classes; it does not change H1 to H5.
- One Discussion sentence on absorption by sibling classes (U1).
- Limitations "Scoring functions and ablation": reworded because feature-space and logit scores are now evaluated; contrastive detectors, null-space-aware energy regularization and adversarial outlier training stay as future directions.
Not applied (kept in this folder for a revision request or for the supplement): U4 routing rarity, U5 protocol-token false-alarm rates, U6 false-alarm sources, U7 seed ensembles, U8 time to detection, U9 error detection, U11 per-class degradation, U12 threshold quantiles. Reasons: length, and none changes a verdict. Candidate one-liners if the editor asks: U12 operating-point trade-off; U6 Benign and scan windows as main false-alarm sources.
Not changed: the introduction roadmap sentence that lists the post hoc analyses (it already omits ESA2 and the event-level check of v5); update it if ESA3 stays.
No citation was added. No history of corrections appears in the text.
