# K-Risk technical validation

These scripts produce the numbers and Figures 8 and 9 of the Technical Validation of the
K-Risk paper, and the in-context learning (ICL) experiment.
They read the released data (`K-Risk_data/`, unzipped from Figshare) and, for the frame-level
statistics, DRF percentiles and coverage (02, 06, 07, 08, 11, 23 `--rule scaled1s`), the labelled
source recordings (`trajectory_data/HV/`), which are not redistributed.

**Paths.** By default both folders are expected next to this repository
(`../K-Risk_data`, `../trajectory_data/HV`) and outputs go to `results/` and `figures/`.
Override with the environment variables `KR_ROOT` (data), `KR_RAW` (recordings), `KR_OUT`,
`KR_FIG`, `KR_GEN` (this repository) and `KR_STAGE` (LLM output folder of 13). Event paths
stored in the result tables are relative to the data root (`event_annotations/...`).

Environment: Python 3.8 with pandas, numpy, scipy, shapely 2 (analysis);
Python 3.12 with matplotlib and PyMuPDF (figures and figure QA).

| Script | Purpose | Main outputs (`results/`) |
|---|---|---|
| `kr_common.py` | event index (de-duplicated, as `compute_statistics.py`), normalised loader, geometry | – |
| `01_event_features.py` | per-event rule features and independent indicators (clearance, PET, DRAC, jerk), slot coverage | `event_features.csv` |
| `02_highd_frame_level.py` | DRF–TTC complementarity and rarity of joint conditions on all 60 highD recordings | `highd_frame_level.json`, `highd_frame_sets.csv` |
| `03_action_checker.py` | collision-free simulator (action-conditioned CA/CS checker); `--horizon --decel --accel --vlat` | `checker_<tag>.csv`, `checker_llm_<tag>.csv` |
| `04_conflict_scan.py` | independent re-implementation of the 2-s conflict test on every frame | `conflict_scan.csv` |
| `05_llm_audit.py` | deterministic audit of the 744 LLM annotations | `llm_audit.json`, `llm_audit_per_event.csv` |
| `06_audit_validity.py` | rule-consistency audit; stratified Cliff's delta; Jonckheere–Terpstra | `audit.json`, `validity_effects.csv`, `validity.json`, `event_audit_flags.csv` |
| `07_coverage.py` | share of original agent-frames contained in the released windows (E2) | `coverage.json` |
| `08_sensitivity.py` | threshold re-grading and checker sensitivity | `sensitivity_grades.csv`, `sensitivity_checker.csv` |
| `09_spotcheck_render.py` | renders the 100-event stratified sample | `figures/spotcheck/sheet_XX.png` |
| `09b_spotcheck_ngsim.py` | redraws the NGSIM part of the sample from the rebuilt NGSIM events (seed 20261006) | `spotcheck_sample.csv` |
| `10_spotcheck_judgements.py` | visual judgements and agreement with the automated checks | `spotcheck_judgements.csv`, `spotcheck_summary.json` |
| `11_facts.py` | remaining descriptive numbers quoted in the text | `facts.json` |
| `12_release_files.py` | release-ready validation files | `release/event_audit_flags.csv`, `release/llm_audit_flags.csv` |
| `13_regenerate_llm.py` | generates the 744 LLM annotations with `release/system_prompt.py` (OpenAI API with `OPENAI_API_KEY`, otherwise OpenRouter with the provider pinned to OpenAI and `OPENROUTER_API_KEY`) | `llm_v3/`, `release/llm_generation_log.jsonl` |
| `14_ngsim_diagnose.py` | compares the previously released NGSIM events with the complete I-80 trajectories | console |
| `15_ngsim_rebuild.py` | NGSIM role assignment and TTC (bumper-to-bumper gap / closing speed) from the complete I-80 data | module used by 16, 17 |
| `16_ngsim_verify.py` | consistency checks of rebuilt NGSIM records against the original data | console |
| `17_ngsim_extract.py` | NGSIM event extraction: one event per vehicle and period at its minimum TTC (< 3 s), 61-frame window, high if < 1.5 s | `ngsim_v3/`, `ngsim_v3/extract_log.csv` |
| `18_ind_heading.py` | median heading difference between the inD ego vehicle and its left-preceding agent (R1.2) | printed (175.3 deg over frames, 175.9 deg over events) |
| `fig7_validity.py`, `fig8_closedloop.py`, `fig_icl.py` | manuscript Figures 8 and 9 and the ICL panel (the optional panel-alignment audit runs when `KR_FIG_AUDIT` points to its helper) | `figures/fig_validity.*`, `figures/fig_closedloop*.*`, `figures/fig_icl.*` |
| `21_failure_cases.py` | fault-based unsafe verdicts (lane keeping: hit ahead/alongside; lane change: any hit or no lane; DEC unsafe if IDLE/ACC safe and the follower hits within 1 s), failure-case texts with surrounding vehicles, ICL case selection (`--pool all --n 20`, `--include --tag` for supersets) | `checker_fault.csv`, `failure_cases.csv`, `icl_failure_cases*.txt`, `icl_selected*.csv` |
| `22_eval_icl.py` | zero-shot vs ICL conditions on the held-out events, exact McNemar, action shares | `icl_eval_<tag>.csv` |
| `23_human_action.py` | human action in the 1 s after the decision frame (thresholds halfway to the checker's actions, `--rule matched1s`) judged with the same verdicts | `human_action_matched1s.csv`, `safety_table_matched1s.csv` |

Additions for the ICL experiment: `03_action_checker.py --hwin` sets the human-response window (thresholds scaled by window/2 s; default unchanged); `13_regenerate_llm.py --icl/--skip` appends a case block / skips events and calls the OpenAI API directly when `OPENAI_API_KEY` is set; `kr_common.py` paths and `KR_GEN`, `KR_STAGE`, `KR_ROOT` can be overridden by environment variables.

`release/system_prompt.py` holds the system prompt used for the released annotations
(identical to the prompt in `K-Risk/prompts.py` and `llm_test.ipynb`). The validity
analysis stratifies NGSIM by its three I-80 recording periods (from `Global_Time`).
The complete I-80 trajectories (Socrata dataset 8ect-6jqj, all 4,566,387 I-80 rows)
are cached in `ngsim_source/i80.parquet.pkl` (pandas pickle); `ngsim_source/dl*.sh`
re-download them in time chunks, and `14_ngsim_diagnose.py` rebuilds the cache from
the chunks (the cache and chunks are not tracked in the repository).
