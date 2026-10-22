# EFELE N₂O temporal reconciliation

Research code and frozen numerical outputs for **Separating cycle levels and daily
shapes for agricultural nitrous oxide estimation from incomplete flux records**.

The study compares 18 fixed settings on eight repeatedly used development cycles
at the EFELE site in France. Its primary cycle target is the arithmetic mean on
observed dates. The saved data contain 5,970 calendar treatment-days and 3,319
observed treatment-days. Conditional resampling uses eight management cycles,
retaining the two treatments together. These records support retrospective
development evaluation; they do not constitute an independent external test or
a complete-cycle integrated emission budget.

## Quick reproduction without training

Use Python 3.12 in a clean environment. All paths below are relative to this
repository root; Unicode directory names preserve the actual analysis-stage
layout. Repository-local helpers supply the required plotting functions. The
saved-prediction computations and figures require no pretrained checkpoint.

```bash
python -m pip install -r requirements.txt
python verify_package.py
python reproduce_metrics.py
python reproduce_figures.py --only Fig3 --preview
```

`reproduce_metrics.py` recalculates daily/weekly/monthly/cycle performance,
all 6,435 weighted cycle-resampling count vectors, paired score intervals,
uninterrupted observation-window diagnostics and the daily SSE decomposition.
It checks all 18 models against the frozen full-precision results. Outputs go
to a new `_runs/metrics/` directory. A repeated run requires a fresh path:

```bash
python reproduce_metrics.py --output _runs/metrics_second_run
python reproduce_figures.py --output _runs/all_figures
```

The second command exports the current six main and six supplementary figures
as PDF/PNG and independent panels. It reads saved predictions and diagnostics;
it does not train models. Figures can require several minutes and substantial
RAM at 600 dpi. The `--preview` option writes low-resolution images under the
chosen output's `_qa/` folder. Arial is requested with available-font fallbacks;
font substitution may change layout and does not change numerical results.

Reference values from the saved predictions:

| Model | Daily R² | Cycle R² | Window WAPE (%) |
| --- | ---: | ---: | ---: |
| LightGBM | 0.2777018390 | 0.7634799698 | 56.79 |
| TabPFN | 0.3320908181 | 0.6305 | 61.78 |
| Mean reconciliation | 0.3416 | 0.7634799698 | 56.56 |
| Affine reconciliation | 0.3454 | 0.8107 | 59.19 |
| Decoupled + pulse | 0.3425 | 0.8056 | 59.50 |

Rounded entries are for orientation. The authoritative full-precision values are
in `08_形状水平解耦与施肥约束_20261018/results/all_model_metrics.csv`.

## Files and provenance

| Directory or entry | Purpose |
| --- | --- |
| `00_原始数据/` | The three original EFELE flux, weather and management inputs, with unchanged bytes |
| `scripts/pilot_memory_n2o.py` | Original data loader, within-training imputation, features and baseline fit functions; also retains historical model utilities used during development |
| `03_首轮实测_20261018/` | Frozen baseline environment/configuration and only the held-out LightGBM rows used by this paper |
| `05_框架再评估_20261018/scripts/` | Common observed-support, window and resampling evaluation functions |
| `06_周期目标与Agent方法_20261018/scripts/` | Cycle-feature functions imported by the joint-loss experiment |
| `07_周期损失与TabPFN_20261018/` | Joint-loss and TabPFN sources, frozen predictions and fitting/selection records |
| `08_形状水平解耦与施肥约束_20261018/` | Nested residual-head fits, all 18 settings, outer feature folds, coefficients and full-precision evaluation |
| `11_图件诊断补强_20261018/` | Frozen-model reliance, soil-response and TreeSHAP sources/results; saved LightGBM models; geographic inputs |
| `17_图件精修_20261021/scripts/` | Final main/SI figure code and layout helpers |
| `portable_helpers/` | Minimal local colors and declaration validation for the retained plotting interface |
| `provenance/` | SHA-256 manifest, source/public hash mapping, portable-copy notes and package-level verification records |

Manuscripts, private author correspondence, literature PDFs, historical writing drafts, review
images, AI-generated images, virtual environments, installation caches and
TabPFN checkpoint weights are excluded. The 28 inner feature folds and cached
training predictions are regenerated during full training, rather than copied.
The eight included outer folds permit inspection of the held-out feature inputs
and saved LightGBM/SHAP models.

`provenance/source_map.csv` lists original project-relative paths, original hashes,
public paths and public hashes. Original run hashes have archival status because
portable paths and some scripts were adapted. `verify_package.py` checks the
current public bytes against `provenance/public_manifest.json`. The scientific
data values and model settings were not revised during packaging. The current
S6 checkpoint identity is retained. See [portability notes](provenance/PORTABILITY.md).

## Model labels

| Saved label | Manuscript/table label |
| --- | --- |
| `OriginalLightGBM` | LightGBM |
| `TabPFN_DailyRaw` | TabPFN |
| `Joint01` | Joint loss, fixed cycle-loss weight 0.1 |
| `JointSelected` | Joint loss with internally selected weight |
| `Blend50` | Equal-weight blend |
| `BlendCycle` | Learned blend |
| `MeanReconciled` | Mean reconciliation |
| `BiasReconciled` | Offset calibration |
| `AffineReconciled` | Affine reconciliation |
| `DecoupledNoPulse` | Free shape / no pulse |
| `DecoupledPulse` | Decoupled + pulse |
| `DecoupledUnconstrained` | Unconstrained pulse |
| `TiedPulse` | Shared pulse weights |
| `PulseShapeOnly` | Pulse in shape only |
| `PulseLevelOnly` | Pulse in level only |
| `DecoupledPulseWeak` | Pulse, shrinkage λ = 0.1 |
| `DecoupledPulseStrong` | Pulse, shrinkage λ = 10 |
| `DecoupledNoEnvironment` | Pulse without environmental modulation |

Other labels in the complete round-07 controls are retained to document that
fixed experiment. They are not additional candidates in the 18-setting analysis.

## Full training from the three raw inputs

Full training is substantially more expensive than recomputing saved results.
The original CPU environment used Python 3.12.3, NumPy 1.26.4, pandas 2.2.2,
SciPy 1.13.1, scikit-learn 1.5.1, LightGBM 4.6.0 and a CPU PyTorch build
recorded as 2.13.0+cpu. Install a compatible PyTorch in the CPU environment;
the loader imports PyTorch even when fitting only LightGBM. CPU fitting uses
two threads. `requirements.txt` records the numerical/figure environment used
for this package's saved-prediction checks.

To recompute TreeSHAP interactions during full training, additionally install
`requirements-interpretation.txt` in the CPU environment. Saved-result metrics
and figure exports do not import SHAP. The frozen run recorded SHAP 0.52.0 in
its pre-existing environment, but current SHAP 0.52.0 dependency metadata requires
NumPy >=2 and cannot be clean-installed with the pinned NumPy 1.26.4. The optional
portable specification therefore uses SHAP 0.46.0 and Numba 0.60.0; SHAP 0.46.0's
[official runtime dependency metadata](https://github.com/shap/shap/blob/v0.46.0/pyproject.toml)
permits NumPy 1.26. This alternative was checked for dependency compatibility;
full interaction recomputation with it was not executed during packaging.

The separate accelerator environment used TabPFN 2.1.2, PyTorch 2.7.1+cu126,
NumPy 1.26.4, pandas 2.2.3 and scikit-learn 1.6.1 on an NVIDIA RTX 4060 Laptop
GPU. Install a compatible CUDA PyTorch build following the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/),
then install `requirements-tabpfn.txt` in that environment. The inner crossfit
script explicitly requires CUDA. Full retraining has not been repeated as part
of this public-code packaging step. Hardware and library changes may affect
floating-point predictions; saved-prediction recomputation is tested separately.

Download `tabpfn-v2-regressor.ckpt` from the
[official Prior Labs model repository](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/tabpfn-v2-regressor.ckpt).
Consult its model card and license before downloading/using the weights. The
weights are a separate third-party artifact and are not redistributed here.
The required SHA-256 is:

```text
2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736
```

Set `TABPFN_CHECKPOINT` to the absolute local checkpoint path. For PowerShell:

```powershell
$env:TABPFN_CHECKPOINT='D:/models/tabpfn-v2-regressor.ckpt'
python reproduce_training.py --output _runs/full_training --tabpfn-python D:/envs/tabpfn/Scripts/python.exe --figures
```

For POSIX shells:

```bash
export TABPFN_CHECKPOINT=/path/to/tabpfn-v2-regressor.ckpt
python reproduce_training.py --output _runs/full_training --tabpfn-python /path/to/tabpfn/bin/python --figures
```

The output directory must not exist. The runner copies sources and inputs into
that new workspace and executes the following stages in order:

1. Refit the original eight-fold LightGBM baseline using its original fit and
   within-training imputation functions.
2. Build the round-07 outer inputs, fit all fixed joint-loss controls, run the
   TabPFN synthetic smoke check and the fixed daily/cycle comparisons, then
   evaluate the saved predictions.
3. Prepare all 36 six-/seven-cycle LightGBM feature contexts; infer the 28 inner
   TabPFN contexts; run synthetic structural checks; fit the 14 fixed combination
   settings; evaluate all 18 settings across temporal supports.
4. Reconstruct frozen-model temporal shifts/local responses, TreeSHAP values and
   interactions; prepare interpretation summaries.
5. Export the final six main and six supplementary figures if `--figures` is set.

The per-stage source scripts and frozen protocol files document the formulas,
constraints, candidates and stopping guards. CSVs are read with
`float_precision='round_trip'` where LightGBM threshold comparisons require
round-trip precision. TabPFN inputs are float32 and inference is offline after
checkpoint retrieval. The runner does not send field data to an inference API.

## Data attribution and third-party material

The original measurements, weather and management records are from **Massad,
Raia Silvia; Buysse, Pauline; Houot, Sabine; Levavasseur, Florent; Flechard, Chris;
Versini, Antoine; and Laville, Patricia (2024). Database for N2O emissions after
EOM application, v1. Zenodo.** [DOI: 10.5281/zenodo.14186466](https://doi.org/10.5281/zenodo.14186466).
The deposit is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Only the three EFELE input files are redistributed; original input bytes are
unchanged. Derived predictors and predictions are separate analysis products.

The country boundaries are the 1:110m Natural Earth administrative-country
dataset, [public domain](https://www.naturalearthdata.com/about/terms-of-use/).
Site coordinates and the map are contextual; only EFELE supplied evaluated
predictions in this paper. TabPFN, LightGBM, PyTorch and other dependencies retain
their own licenses. Public availability of this research code does not assign
a new license to third-party data or weights. No additional general software
license is granted by this release.

For checkpoint/library citation use Hollmann et al. (2025), *Accurate predictions
on small data with a tabular foundation model*, Nature,
[DOI: 10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6),
and Ke et al. (2017), *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*.
