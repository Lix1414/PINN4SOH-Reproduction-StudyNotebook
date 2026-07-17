# PINN4SOH Reproduction and Learning Notes

[简体中文](README.zh-CN.md)

An **unofficial, learning-oriented reproduction** of *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis* (Nature Communications, 2024).

This repository reorganizes the public PINN4SOH workflow into readable modules and provides executable notebooks, validation notes, and reproducibility-oriented experiment scripts. It is intended for study and engineering verification, not as an official implementation or a claim of exact numerical reproduction.

## Why I open-sourced these notes

AI for batteries is moving from exploratory research toward engineering deployment. During this transition, the field inevitably includes work with uneven validation, limited reproducibility, or insufficient attention to engineering constraints. I do not see these issues as evidence that the direction itself lacks value; rather, they show that stronger links are still needed between algorithms, battery mechanisms, data quality, operating conditions, and deployable systems.

The scope of battery intelligence is also expanding from individual cells to battery packs, battery clusters, and complete energy-storage systems. I believe this progression will remain important, and that open, reviewable learning material can help connect research ideas with practical questions.

PINN4SOH is a representative and comparatively approachable case study for learning this area. I am sharing these notes to help readers—especially those without a strong programming background—follow the data flow, model structure, loss functions, and reproduction process step by step. If these notes help you enter the field, reproduce an experiment, identify an error, or raise a better engineering question, you are welcome to follow the repository and join the discussion.

## Who this repository is for

- Beginners exploring the intersection of AI and battery health estimation.
- Readers who want a guided path from a research paper to executable code.
- Researchers or engineers interested in checking implementation details and reproduction boundaries.
- Learners who would like to discuss errors, alternative interpretations, or possible engineering extensions.

## What is included

- A cleaned pipeline covering feature extraction, data loading, PINN forward propagation, loss design, training, and evaluation.
- Six Chinese learning notebooks that can be run from top to bottom.
- A minimal end-to-end training example.
- Experiment launchers and reports for repeated XJTU evaluations.
- Notes describing known differences between the paper, the upstream repository, and this reproduction.

## Repository layout

```text
src/          Cleaned Python implementation
scripts/      Verification and training entry points
notebooks/    Executable learning notebooks (Chinese)
docs/         Reproduction notes, manuals, and validation reports
experiments/  Experiment launchers; generated outputs are excluded
assets/       Selected explanatory figures
```

## Suggested learning path

1. Read the paper together with the upstream repository to understand the research objective and original implementation.
2. Work through notebooks `01` to `06` in order: feature extraction, data preprocessing, forward propagation, loss design, training, and evaluation.
3. Run the quick-verification commands below to confirm that your environment and data paths are correct.
4. Read the notes in `docs/` before interpreting metrics, especially the documented differences between the paper, upstream source, and this reproduction.
5. Use the minimal run for pipeline checks; use repeated experiments only when evaluating reproducibility or reporting performance.

## Quick start from the downloaded ZIP

Python 3.12 is the verified environment. The ZIP contains code and learning material but not third-party battery datasets, so complete the data step before running the pipeline.

1. Extract the ZIP and open a terminal in the extracted repository root.
2. Create an isolated environment and install dependencies.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   On Command Prompt use `.venv\Scripts\activate.bat`; on Linux or macOS use `source .venv/bin/activate`.

3. Obtain the upstream preprocessed XJTU CSV files and place them as follows:

   ```text
   PINN4SOH-Reproduction-StudyNotebook/
   └── data/
       └── XJTU data/
           ├── 2C_battery-1.csv
           ├── 2C_battery-2.csv
           └── ...
   ```

   The fastest route for model reproduction is to use the preprocessed CSV files from the upstream PINN4SOH repository. Raw-data feature extraction is a separate workflow because the authors did not publish the exact original feature-extraction implementation. See [DATA.md](DATA.md) for both routes and licensing notes.

4. Check the environment and data layout, then run the verified pipeline.

   ```powershell
   python scripts/check_setup.py
   python src/02_dataloader.py
   python src/03_model.py
   python src/04_loss.py
   python src/verify_pipeline.py
   ```

5. Run a deterministic 10-epoch smoke test or the complete repeated experiment.

   ```powershell
   python scripts/run_minimal_training.py
   python experiments/run_paper_60.py
   ```

   The minimal run writes to `results/minimal_run/`. The repeated experiment writes summaries to `experiments/results/` and figures to `experiments/images/`; generated artifacts are ignored by Git.

Neural-network training is stochastic. The commands reproduce the workflow and should produce results in the same range, but an unfixed 60-run suite is not expected to reproduce every reported decimal. Pass `--seed 42` to `run_paper_60.py` when deterministic reruns are more important than matching the original unfixed-seed protocol.

## Reproduction boundaries

- The upstream repository does not publish the original 16-feature extraction implementation. The feature extractor here is a reverse implementation based on the paper and public preprocessing resources.
- Data loading follows the upstream behavior: the 16 statistical features and `cycle index` are normalized together, while `capacity` (SOH) is excluded. The cleaned version only adds float-dtype and zero-range safeguards for pandas 3 compatibility.
- The monotonicity loss formula described in the paper differs from the public upstream source. The cleaned implementation follows the public source behavior and documents the difference.
- Neural-network training is stochastic. Matching the workflow does not imply bitwise-identical metrics.
- Minimal runs validate the pipeline only; they are not evidence of reproducing the full 200-epoch paper performance.

## Upstream project and paper

- Upstream repository: <https://github.com/wang-fujin/PINN4SOH>
- Paper: Wang et al., “Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis,” *Nature Communications* 15, 4332 (2024). DOI: <https://doi.org/10.1038/s41467-024-48779-z>

## Language

The main project overview is available in English and Chinese. Detailed notebooks and learning notes are currently written in Chinese; source-code identifiers are kept in English where practical.

If you would like to read an English version of a notebook, it is practical to provide the notebook to a large language model and ask it to translate the Markdown explanations while preserving code cells, formulas, variable names, and file paths. Because automated translation may alter technical terminology or mathematical meaning, please compare important passages with the paper, the source code, and the Chinese original.

## Generative AI disclosure and learning exchange

Generative AI and a self-authored reproduction-learning Skill were used to assist with code organization, documentation, consistency checks, and reproducibility verification. The repository author reviewed the implementation decisions, technical interpretations, and release contents. AI assistance does not replace the original paper or upstream source; readers should consult those primary materials when verifying research claims.

Questions, corrections, and discussions about reproduction details are welcome. You can contact me by email:

- Email: `lixiangl1iiowo@gmail.com`

## Copyright and licensing

This repository contains study materials and code derived from or informed by third-party research artifacts. No license is granted for third-party material unless its rights holder states otherwise. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A project-wide open-source license should only be added after the licensing status of derived code has been confirmed.
