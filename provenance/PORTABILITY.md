# Portable-copy changes

Original scientific files are preserved in the source project. This repository
contains a bounded copy, not the complete development archive.

- All raw EFELE CSV/XLS inputs are byte-identical to the source deposit.
- The early prediction file is restricted to the LightGBM baseline used here.
- TabPFN checkpoint paths use `TABPFN_CHECKPOINT`; no weights are redistributed.
- Figure helpers are repository-local. The small palette helper was implemented
  for this repository from the numerical colors actually used by the figures;
  the user's complete figure-skill implementation was not copied.
- The final figure script accepts `EFELE_FIGURE_OUTPUT` for isolated exports.
- Original frozen-run hash keys are normalized to project-relative labels and
  marked archival. They identify the original run, not changed portable scripts.
  `source_map.csv` maps original hashes to public hashes, while
  `public_manifest.json` verifies current public bytes.
- Evaluation skips archival original-source checks after a manifest check.
  Newly trained run records retain the original source-hash verification logic.
- Full training is started in a new directory; saved scientific files stay intact.

Figure font substitution can alter layout. Exact training predictions can vary
with accelerator/library versions; frozen-prediction recomputation is the
direct route to checking the reported numerical results.
