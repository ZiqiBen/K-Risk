# K-Risk

**A knowledge-augmented dataset of high-risk driving scenarios with LLM annotations for autonomous driving**

[![Paper](https://img.shields.io/badge/Paper-arxiv-b31b1b?style=for-the-badge)](#)
[![Data](https://img.shields.io/badge/Data-Figshare-1f9d55?style=for-the-badge)](https://doi.org/10.6084/m9.figshare.32896772)
[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.32896772-1f9d55?style=for-the-badge)](https://doi.org/10.6084/m9.figshare.32896772)
[![Code](https://img.shields.io/badge/Code-MIT-blue?style=for-the-badge)](LICENSE)

K-Risk pairs **structured vehicle trajectories** with **LLM-generated natural-language annotations** for safety-critical driving. It aggregates **20 naturalistic (HV) and automated-vehicle (AV) trajectory sources** across Europe, China and the United States — covering highways, urban freeways, intersections and roundabouts — and curates **32,839 high-risk events**, including a **1,036-event extreme near-collision subset**.

This repository hosts the **data-processing pipeline** (scenario-description generation, the LLM annotation driver, the statistics/figure scripts, and the technical-validation scripts with the collision-free simulator and the in-context learning experiment). The **processed dataset itself is distributed separately on Figshare** (see [Data availability](#data-availability)) and is *not* tracked in this repository.

<p align="center">
  <img src="docs/fig_framework.png" alt="Overview of the K-Risk dataset and the LLM annotation framework" width="100%">
</p>


---

<!-- ## Table of contents

- [Highlights](#highlights)
- [Dataset at a glance](#dataset-at-a-glance)
- [Source datasets](#source-datasets)
- [Repository structure](#repository-structure)
- [Data layout (Figshare release)](#data-layout-figshare-release)
- [Event schema](#event-schema)
- [Installation](#installation)
- [Usage](#usage)
- [Data availability](#data-availability)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)

--- -->

## News

- 🔥 **[2026/10]** Updated release: validation scripts (`validation/`), per-event audit files and the in-context learning experiment; new Figshare version of the dataset.
- **[2026/07]** The full **K-Risk dataset** (event-level trajectories, annotations and LLM responses) is now available on **Figshare**: [10.6084/m9.figshare.32896772](https://doi.org/10.6084/m9.figshare.32896772).
- 🗓️ **[2026/06]** Initial release of the **K-Risk data-processing pipeline** (event extraction, scenario-description generators, the closed-loop LLM annotation driver, and the statistics/figure scripts).

---

## Highlights

- 🌍 **Multi-region, multi-environment** — 20 HV + AV trajectory sources from Europe, China and the United States, spanning highways, urban freeways, intersections and roundabouts.
- ⚠️ **Multi-dimensional risk definition** — events are screened by a *driver risk field (DRF)*, *hard-maneuver thresholds* (hard acceleration / braking, lane change), *time-to-collision (TTC)* and a *two-second trajectory-conflict test*, and HV events are assigned the risk level **moderate / high / extreme**.
- 🧩 **Synchronized records per event** — every event is stored as a **JSON** (per-frame trajectory + symbolic metadata) paired, for the LLM-annotated subset, with a **TXT** (natural-language annotation) that shares one event identifier.
- 🤖 **LLM annotation with verifiable decisions** — 372 extreme events carry LLM (GPT-4o, GPT-4.1) risk analyses and a discrete action recommendation from a **five-action schema**; every recommendation can be checked in a **collision-free simulator**, and unsafe decisions serve as failure cases for **in-context learning**.
- 🔁 **Reusable across post-training stages** — the release supports CPT, SFT, RLHF (preference pairs) and RLVR (verifiable simulator reward) for driving agents.

---

## Dataset at a glance

| | |
|---|---|
| Trajectory sources | **20** (6 human-driven + 14 automated-vehicle) |
| Regions | Europe, China, United States |
| Environments | Highway, urban freeway, intersection, roundabout |
| Curated high-risk events | **32,839** |
| Extreme near-collision subset | **1,036** events |
| Risk levels (HV) | Moderate 71.8% · High 24.7% · Extreme 3.4% |
| Unique agents | 53,969 (18.3% non-car: VRUs + heavy/special vehicles) |
| LLM-annotated subset | 372 events (responses from GPT-4o and GPT-4.1) |
| Per-event representation | trajectory + metadata (JSON), plus natural-language description/response (TXT) for the annotated subset |
| Action schema | IDLE · Turn Left · Turn Right · Acceleration · Deceleration |

<p align="center">
  <img src="docs/fig_statistic.png" alt="Composition and safety-critical properties of K-Risk" width="80%">
</p>


---

## Source datasets

K-Risk is a **secondary dataset**: it releases only the **event-level segments** derived from the sources below. The complete original recordings are **not** redistributed and remain available from their providers under their own licenses. Each source was used in accordance with its license and terms of use.

### Human-driven vehicles (HV)

| Source | Collection | Region | Environment | Link |
|---|---|---|---|---|
| highD | LevelXData | Germany | Highway | https://levelxdata.com/highd-dataset/ |
| inD | LevelXData | Germany | Intersection | https://levelxdata.com/ind-dataset/ |
| rounD | LevelXData | Germany | Roundabout | https://levelxdata.com/round-dataset/ |
| ExpresswayA | CitySim | China | Urban freeway | https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset |
| FreewayB | CitySim | China | Urban freeway | https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset |
| NGSIM (I-80) | FHWA NGSIM | United States | Highway | https://doi.org/10.21949/1504477 |

### Automated vehicles (AV)

The 14 AV sources follow the unified longitudinal trajectory collection of **Ultra-AV**; the CATS, Central Ohio and Vanderbilt recordings are obtained through Ultra-AV, which documents their original providers and access points.

| Source | Provider | Region | Link |
|---|---|---|---|
| Argoverse 2 Motion Forecasting | Argo AI | United States | https://www.argoverse.org/av2.html |
| Waymo Open Motion | Waymo | United States | https://waymo.com/open/ |
| Waymo Open Perception | Waymo | United States | https://waymo.com/open/ |
| MicroSimACC | microSIM-ACC | United States | https://github.com/microSIM-ACC/ICE |
| OpenACC — Casale | JRC (EU) | Europe | https://doi.org/10.2905/JRC.KMH3D00 |
| OpenACC — Vicolungo | JRC (EU) | Europe | https://doi.org/10.2905/JRC.KMH3D00 |
| OpenACC — AstaZero | JRC (EU) | Europe | https://doi.org/10.2905/JRC.KMH3D00 |
| OpenACC — ZalaZONE | JRC (EU) | Europe | https://doi.org/10.2905/JRC.KMH3D00 |
| CATS — ACC | Ultra-AV | United States | https://github.com/CATS-Lab/Filed-Experiment-Data-ULTra-AV |
| CATS — Platoon | Ultra-AV | United States | https://github.com/CATS-Lab/Filed-Experiment-Data-ULTra-AV |
| CATS — UWM | Ultra-AV | United States | https://github.com/CATS-Lab/Filed-Experiment-Data-ULTra-AV |
| Central Ohio — single-vehicle | Ultra-AV | United States | https://github.com/CATS-Lab/Filed-Experiment-Data-ULTra-AV |
| Central Ohio — two-vehicle | Ultra-AV | United States | https://github.com/CATS-Lab/Filed-Experiment-Data-ULTra-AV |
| Vanderbilt ACC | Ultra-AV | United States | https://github.com/CATS-Lab/Filed-Experiment-Data-ULTra-AV |

---

## Repository structure

This repository contains the pipeline code only. Scripts read from a local copy of the dataset (downloaded from Figshare) and produce the natural-language scenario descriptions, LLM responses, statistics and figures.

```
K-Risk/
├── generate_highd.py          # HighD scenario-description generator (+ highd_msg system prompt)
├── generate_expresswayA.py    # CitySim ExpresswayA generator (+ expresswayA_msg)
├── generate_freewayB.py       # CitySim FreewayB generator (+ freewayB_msg)
├── generate_ind.py            # inD (intersection) generator (+ ind_msg)
├── generate_round.py          # rounD (roundabout) generator (+ round_msg)
├── generate_ngsim.py          # NGSIM I-80 generator (+ ngsim_msg)
├── generate_AV.py             # AV (longitudinal / car-following) generator (+ AV_msg)
├── prompts.py                 # Shared system prompt + the five-action decision schema
├── llm_test.ipynb             # Driver notebook: build samples, query the LLM, store responses
├── compute_statistics.py      # Per-source / risk-level statistics and distribution CSVs
├── plot_overview.py           # Combined composition / statistics overview figure (PNG + PDF)
└── validation/                # Technical validation (see validation/README.md)
    ├── kr_common.py           #   shared event index, loader and geometry
    ├── 01–23_*.py             #   audits, collision-free simulator (03), ICL experiment (21–22), NGSIM extraction (14–17)
    ├── fig*.py                #   validation figures
    ├── release/               #   event_audit_flags.csv, llm_audit_flags.csv, system prompt, generation log
    ├── results/               #   outputs of the scripts (tables quoted in the paper)
    └── figures/               #   validation figures and the visual spot-check sheets
```

Each `generate_<source>.py` exports `generate_<source>(json_path, frame_step=...)`, returning a multi-frame English scenario description, plus a `<source>_msg` system prompt that frames the LLM as a driving assistant. The generators embed **source-specific lane topology** and **risk-reminder templates**, so they are **not interchangeable** — pick the one matching the source of the JSON. Both list-of-records and nested-frames (`{"frames": [...]}`) JSON shapes are supported.

---

## Data layout (Figshare release)

The dataset is organized into three top-level directories. Within `event_annotations/`, events are split into human-driven (`HV/`) and automated-driving (`AV/`) groups, then one folder per source. Each event shares the event identifier encoded in its file name, so the JSON and TXT records of one event can be matched across directories.

```
K-Risk_data/
├── event_annotations/               # Extracted per-event records (JSON)
│   ├── HV/
│   │   ├── highd/  {highd_high_risk/, highd_normal_risk/, info.txt}
│   │   ├── ind/    rounD/  ngsim/  expresswayA/  freewayB/
│   │   └── extreme/  {ttc_1s/, ttc_2s/, info.txt}   # cross-source near-collision subset
│   └── AV/<source>/{acc/, brake/, ttc/}             # split by risk-event criterion
├── llm_analysis/                    # Natural-language layer for the 372 LLM-annotated events
│   ├── event_discriptions/          # Structured scenario descriptions fed to the LLM
│   ├── analysis_gpt4o/              # GPT-4o decision responses
│   ├── analysis_gpt4.1/             # GPT-4.1 decision responses
│   ├── checker_fault.csv            # Collision-free simulator outcome of each of the five actions per event
│   └── failure_cases.csv            # Unsafe zero-shot decisions with their failure-case text
└── llm_analysis_icl/                # In-context learning experiment (20 failure cases, 352 held-out events)
    ├── analysis_gpt4o/  analysis_gpt4.1/
    ├── icl_failure_cases.txt        # Failure-case block appended to the system prompt
    ├── icl_selected.csv             # Events used as failure cases (excluded from the evaluation)
    └── generation_log.jsonl         # Model snapshot, tokens and time of each response
```

- **HV** sources are split into `*_high_risk` / `*_normal_risk` folders; near-collision cases are collected in a separate `extreme/` folder (`ttc_1s` / `ttc_2s`).
- **AV** sources each contain `acc/`, `brake/` and `ttc/` folders, named by the triggering risk-event criterion.
- Per-folder `info.txt` files document the column semantics for that group.

> **Event identifier conventions.** highD uses `highd_<track>_<egoId>_<relation>_<behavior>_frame_<start>_to_<end>.json` (e.g. `highd_01_113_precedingId_yaw_right_frame_2205_to_2312`); other sources use `<source>_track_<n>_car_<id>_frame_<start>_to_<end>.json`; AV uses `trajectory_<id>_lv_<leadId>_<n>.json`.

---

## Event schema

Each event is stored as a per-event **JSON** (per-frame trajectory records + symbolic metadata); the LLM-annotated subset additionally has synchronized **TXT** files (scenario description + LLM responses) sharing the same event identifier. Because every event keeps the **native column names of its source**, per-frame fields are described by role; a field absent in a source is simply not present in its files.

**Event-level descriptors** (from the file name / folder): `event_id`, `source_dataset`, `ego_id`, `risk_level` (normal/high/extreme), `frame_start`, `frame_end`, and for highD additionally `relation` (preceding / following / left-right-preceding / following / alongside) and `behavior` (`acc_high`, `brake_high`, `yaw_left`, `yaw_right`).

**Per-frame trajectory fields** (role → column names across sources):

| Role | Column name(s) across sources | Unit |
|---|---|---|
| frame index | `frame` · `frame_id` · `Frame_ID` | — |
| agent id | `id` · `car_id` · `trackId` · `Vehicle_ID` | — |
| position | `x,y` · `car_center_x,y` · `xCenter,yCenter` · `Local_X,Y` | m |
| size | `width,height` · `length` · `v_Length,v_Width` | m |
| speed | `speed` · `v_Vel` | m/s |
| velocity components | `xVelocity,yVelocity` · `vx,vy` · `lonVelocity,latVelocity` | m/s |
| acceleration | `xAcceleration,yAcceleration` · `acceleration` · `lon/latAcceleration` · `v_Acc` | m/s² |
| heading | `heading` · `course` | deg |
| yaw rate | `yaw_rate` (CitySim) | deg/s |
| lane | `laneId` · `lane_id` · `Lane_ID` | — |
| agent class | `vehicle_class` · `class` · `v_Class` | — |
| spacing | `dhw,thw` · `Space_Headway,Time_Headway` | m, s |
| time-to-collision | `ttc` · `TTC` · `TTC_0_1s,TTC_1_2s` | s |
| trajectory conflict | `collision_0_1s,collision_1_2s,collision_ids_*` (CitySim) | — |
| risk value | `total_risk` · `risk_value` (DRF) | — |
| surrounding agents | `precedingId,followingId,...` · `preceding_id,following_id,...` | — |
| behavioral flags | `acc_high,brake_high,yaw_left,yaw_right` · `acc/brake/left_turn/right_turn_label` | bool |

> The AV (Ultra-AV-derived) sources use a shared longitudinal schema instead: `Trajectory_ID`, `Time_Index`, `ID_LV`, `Type_LV`, `Pos_LV`, `Speed_LV`, `Acc_LV`, `ID_FAV`, `Pos_FAV`, `Speed_FAV`, `Acc_FAV`, `Space_Gap`, `Space_Headway`, `Speed_Diff`, `TTC`.

**Natural-language annotation (TXT)** components: a structured **scenario description**, rule-based **risk reminders**, an LLM **risk analysis** (representative subset), and a discrete **`action_id`** from the five-action schema (1 IDLE, 2 Turn Left, 3 Turn Right, 4 Acceleration, 5 Deceleration).

---

## Installation

The pipeline targets **Python 3.9+** and the standard scientific stack, plus a LangChain LLM client for the annotation step.

```bash
git clone https://github.com/benmagnifico/K-Risk.git
cd K-Risk

python -m venv .venv && source .venv/bin/activate   # optional
pip install pandas numpy matplotlib jupyter
pip install langchain langchain-openai openai tenacity   # only for the LLM annotation step
```

The validation scripts additionally need `scipy` and `shapely>=2` (`pip install scipy shapely`).

Set your API key before running the LLM annotation:

```bash
export OPENAI_API_KEY="sk-..."
```

> **Note.** The dataset is downloaded separately from Figshare (see [Data availability](#data-availability)). By default the scripts expect the unzipped `K-Risk_data/` next to this repository (`../K-Risk_data`); set `KR_ROOT` to point them elsewhere.

---

## Usage

### 1. Generate a natural-language scenario description

```python
from generate_highd import generate_highd, highd_msg

text = generate_highd(
    "K-Risk_data/event_annotations/HV/highd/highd_high_risk/"
    "highd_01_113_precedingId_yaw_right_frame_2205_to_2312.json",
    frame_step=5,
)
print(text)   # multi-frame English description; pair with `highd_msg` as the system prompt
```

Use the generator matching the source: `generate_expresswayA` / `generate_freewayB` (CitySim), `generate_ind`, `generate_round`, `generate_ngsim`, or `generate_AV` (longitudinal AV). The `extreme/` subset inherits the CitySim schema, so it is described with `generate_expresswayA` / `generate_freewayB`.

### 2. Query the LLM (closed-loop annotation)

Open `llm_test.ipynb`. It provides:

- `create_samples(scenario, frame_step, num_examples, risk_level)` — samples event JSONs and writes scenario descriptions.
- `llm_response(input_file, system_message, llm, output_folder)` — sends a description plus its system prompt to the LLM and stores the response.

The notebook reads events from `$KR_ROOT/event_annotations/HV` (default `../K-Risk_data`), writes descriptions to `llm_text/samples/` and responses to `llm_text/responses/`, and reads the API key from `OPENAI_API_KEY`. The released annotations were generated with `gpt-4o-2024-08-06` and `gpt-4.1-2025-04-14` at temperature 0 from the last described frame of each description; `validation/13_regenerate_llm.py` reproduces this step from the command line.

### 3. Recompute statistics and figures

```bash
python compute_statistics.py    # writes K-Risk_statistics.md + distribution CSVs
python plot_overview.py         # writes the combined overview figure to figures/ (PNG + PDF)
```

`compute_statistics.py` reads `$KR_ROOT/event_annotations` (default: `event_annotations/` next to the script, else `../K-Risk_data`).

> **Note.** `compute_statistics.py` also references a `trajectory_data/` folder (raw per-source trajectory tables) for its "K-Risk vs. original" comparison. Those raw tables are **not part of the Figshare release** (they belong to the original providers and are not redistributed — see [Source datasets](#source-datasets)). The dataset-only statistics run from `event_annotations/` alone; to reproduce the comparison tables, reconstruct `trajectory_data/` from the original recordings under their own licenses.

---

## In-context learning from verified failures

The scripts in `validation/` replay each LLM decision on the recorded trajectories and turn unsafe decisions into feedback for in-context learning (ICL):

```bash
cd validation
python 03_action_checker.py --llm-only --hwin 1.0 --tag hwin1   # per-event decision frame, lane and zero-shot actions
python 21_failure_cases.py --pool all --n 20                    # fault-based verdicts, failure-case texts, ICL block
python 13_regenerate_llm.py --icl results/icl_failure_cases.txt --skip results/icl_selected.csv
python 22_eval_icl.py --icl ICL-20=<output folder of 13>         # zero-shot vs ICL, exact McNemar test
python 23_human_action.py --rule matched1s                      # optional: human driver baseline
```

By default the scripts read `../K-Risk_data` and write to `validation/results/`; set `KR_ROOT` (data), `KR_RAW` (labelled source recordings, needed only by the frame-level analyses), `KR_OUT`, `KR_FIG` and `KR_STAGE` (output folders) to run elsewhere. `13_regenerate_llm.py` uses `OPENAI_API_KEY` (direct OpenAI API) if set, otherwise OpenRouter (`OPENROUTER_API_KEY`). The ICL responses and prompts are released in `K-Risk_data/llm_analysis_icl/`. The per-event audit flags and the independent indicators of the Technical Validation are in `validation/release/event_audit_flags.csv`, and the flags of every LLM annotation in `validation/release/llm_audit_flags.csv`.

---

## Data availability

The K-Risk dataset is archived on **Figshare** under a persistent DOI and is **not** stored in this Git repository:

- **Code & pipeline:** https://github.com/benmagnifico/K-Risk
- **Dataset (archived):** https://doi.org/10.6084/m9.figshare.32896772 (terms of use per source; see [License](#license))

The release (`K-Risk_data.zip`) contains the event-level trajectory segments (as per-event JSON records), the risk annotations and metadata, the natural-language descriptions and LLM responses, and the in-context learning experiment, organized into `event_annotations/`, `llm_analysis/` and `llm_analysis_icl/`. The raw per-source trajectory tables are **not** redistributed; the original recordings are available from the providers listed in [Source datasets](#source-datasets), each under its own license and terms of use.

---

## Citation

If you use K-Risk, please cite the Figshare dataset:

```bibtex
@dataset{huang2026krisk,
  author    = {Huang, Heye and Li, Jingguang and Zhou, Zhiyuan and Liang, Paul and
               Wu, Mingyu and Jang, Kitae and Wang, Jianqiang},
  title     = {{K-Risk}: A knowledge-augmented dataset of high-risk driving
               scenarios with {LLM} annotations for autonomous driving},
  year      = {2026},
  publisher = {figshare},
  doi       = {10.6084/m9.figshare.32896772},
  url       = {https://doi.org/10.6084/m9.figshare.32896772}
}
```

> Huang, Heye; Li, Jingguang; Zhou, Zhiyuan; Liang, Paul; Wu, Mingyu; Jang, Kitae; Wang, Jianqiang (2026). *K-Risk: A knowledge-augmented dataset of high-risk driving scenarios with LLM annotations for autonomous driving*. figshare. Dataset. https://doi.org/10.6084/m9.figshare.32896772

Please also cite the original source datasets you use (see [Source datasets](#source-datasets)), in particular Ultra-AV for the AV component.

---

## License

- **Pipeline code** (this repository): released under the [MIT License](LICENSE).
- **Processed K-Risk dataset** (Figshare release): the released records keep the terms of their source (see the table below and `README.md` in the release). Records derived from sources with non-commercial terms are restricted to non-commercial research use. The highD, inD and rounD windows are redistributed with the written permission of levelXdata, and the ExpresswayA and FreewayB windows with the written permission of the Southwest Jiaotong University Data Science and Traffic Safety Lab.

| Source | Terms of use |
|---|---|
| highD, inD, rounD (levelXdata) | non-commercial research license |
| ExpresswayA, FreewayB (CitySim) | Apache-2.0 repository; locations distributed on request |
| NGSIM I-80 (US DOT FHWA) | CC BY-SA 3.0 |
| Ultra-AV release (all AV sources) | CC0 1.0 |
| Argoverse 2 Motion Forecasting | CC BY-NC-SA 4.0 |
| Waymo Open Motion / Perception | Waymo Dataset License Agreement for Non-Commercial Use |
| MicroSimACC | public GitHub repository |
| OpenACC (4 sites) | open access, JRC Data Catalogue |
| CATS, Central Ohio, Vanderbilt | CC0 1.0 (Ultra-AV) |

The complete original recordings remain available from their providers under their own licenses and terms; obtain and use them accordingly.
