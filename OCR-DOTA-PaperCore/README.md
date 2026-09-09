# OCR-DOTA Paper Core

OCR-DOTA introduces two corrections derived from one sample-class OCR compatibility: an OCR-calibrated posterior for the current prediction, and an OCR-calibrated Gaussian update responsibility for writing the sample into online class distributions.

## Base and evidence

The model uses the literal `LegacyState` DOTA implementation. All four arms share each dataset's original `configs/vit/<dataset>.yaml` values of epsilon, sigma, eta and rho. These parameters are frozen before development and are not searched. Consequently Original DOTA and Matched DOTA coincide in this campaign. Initial state, covariance update arithmetic, CLIP probability responsibilities and selected cached views follow DOTA exactly. Geometry and Enhanced Core are absent.

For the mean view feature, stable text cosine scores and dynamic DOTA Gaussian scores produce normalized descending ranks in [0,1]. `C_k=exp(-abs(r_stable-r_dynamic)/tau_rank)`. One tensor is computed per sample and reused by both consumers and every augmentation view. `m=sum softmax(g)_k*C_k` is the shared clean mass.

## Posterior and responsibility

The primary posterior is `g_ocr=g+lambda_max*m**prediction_power*alpha*log(C)`. The backup blends `p_dota` with normalized `p_dota*C**alpha` using the same gated coefficient, evaluated in log space. Exact identity paths return unchanged logits for alpha=0, lambda_max=0 or C=1.

Internal responsibility modes are `dota`, `gate_only`, `calibrated_clip`, and `posterior`. Their weights are respectively `p_zs`, `m**delta*p_zs`, `m**delta*normalize(p_zs*C**beta_resp)`, and `m**delta*p_ocr_view`. The last mode internally computes OCR posteriors even in the Responsibility Only arm, without changing its current prediction formula. For calibrated_clip, C=1 implies exact DOTA responsibilities. For posterior mode C=1 instead gives Gaussian allocation, which generally differs from CLIP; it does not satisfy that stronger clean-limit contract.

The official four arms are Base, Posterior Only, Responsibility Only and Full. Base/P share exact update trajectories; U/Full share exact update trajectories. Labels are accessed only for diagnostics after the online update.

## Development protocol

`paper_campaign.py` preregisters a deterministic dataset split by SHA256 of `paper-core-v1-dev-split|<dataset>`, taking the lowest 3 classic, 1 Office and 2 DomainNet datasets. The remaining 15, including VisDA, are held out from this campaign's hyperparameter selection. Historical experiments have exposed all 21 datasets: this split isolates selection in the current campaign, and is not a claim of pristine external validation.

Tau is .15. First evaluate all 18 combinations of alpha={.25,.5,1}, lambda_max={.25,.5,1}, prediction_power={1,2} using gated_log_prior. Only if none yields nonnegative development macro and strictly positive correct-count gain, evaluate the 18 probability_blend combinations. Select one global configuration. Failure of both grids produces a negative research report and stops before held-out evaluation; it is not hidden by redefining Base.

After posterior selection, compare R1/R2/R3 globally on development with delta=1 and beta_resp=1. Prefer the highest macro then correct-count result; prefer R3 when within .01 macro percentage points of the best and it still meets the positive development criterion. This numerical tolerance is predeclared. Freeze all settings, base identities and source code hashes in `FROZEN_PAPER_CONFIG.yaml` before evaluating any held-out candidate.

## Running and outputs

1. Run `verify_exact.py`: unit and exact DOTA reproduction at 32, 500 and full DTD samples. `results/exact_validation.json` is mandatory.
2. Run `python paper_campaign.py --phase all --output results/paper_core_v1_20260909`.
3. `tune_paper_dev.py` and `evaluate_paper.py` expose development-only and frozen-evaluation-only entry points.

The campaign has a PID lock, STOP sentinel, manifest identity checks and per-candidate checkpoints. Every candidate uses a new DOTA state and real full-stream GPU replay. Frozen formal arms each receive a second cold replay. Records include prediction, update trajectory, state, compatibility and semantic diagnostic-trace hashes. Resume does not mix changed code or data.

Results include development and held-out aggregates, 21-stream descriptive aggregates, per-dataset four-arm accuracy, late-half accuracy, correction counts, NLL/Brier/ECE, responsibility mass and WRM/CRM, temporal metrics and verification records. Negative or interfering module effects remain in the report. Formal parameters are never changed based on held-out results.

## Boundaries

Prior V3 and ablation directories remain historical evidence. Paper Core does not import their enhanced model as a state backend. Shared repository DOTA/cache helpers are read-only dependencies whose hashes are locked. No mix05/mix08clip, initialization search, geometry risk or per-dataset OCR mode selection is part of this method. The existing base YAML provenance is documented rather than assumed to be a newly verified original-publication configuration.
