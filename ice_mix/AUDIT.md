# IceMix documentation handoff audit

## Executive summary

No. At the current commit, a new student does not receive this handoff at all: every README, `docs/`, and all three ledgers are untracked, while the evidence fixture is explicitly ignored.

If the working tree is copied manually, the most dangerous failures are scientific: the documented “spatial checkerboard” study is actually a seeded random half-split of pulse positions, and normal evaluation silently reconstructs default splits instead of the run’s split. The repository also cannot recreate its custom GraphNeT environment. These can produce mislabeled physics results, evaluate the wrong events, or stop work before the first run.

The audit found five blockers, six major failures, two minor failures, and no fabricated output.

## BLOCKER

### 1. The documented spatial checkerboard study is not spatial or a checkerboard

**Evidence.** `docs/tutorials/05_robustness_studies.md`, lines 11 and 15–17, marks the perturbation algorithms `VERIFIED-STATIC` and says “Checkerboard removal tests spatially structured detector inefficiency by removing alternating cells”; lines 45–65 direct the student to run `checkerboard_test.py` and inspect “cell definitions.” `checkerboard_test.py`, lines 57–104, never reads coordinates or defines cells: it seeds a generator with 42, assigns random scores to padded sequence positions, and retains the first or second half of that random ordering. Its module docstring, lines 1–4, nevertheless calls this “spatial checkerboard pulse removal.” `MEMORY.md`, line 36, repeats the false spatial-cell description as durable knowledge.

**Consequence.** A student can produce polished “checkerboard” CSVs and plots and confidently report sensitivity to spatially structured detector failures even though the code tested an index-position random half-split. That physics conclusion is invalid without any runtime error to warn them.

**What would resolve it.** Either implement and verify a documented coordinate/cell/parity mask, or rename and document the existing operation as a deterministic random complementary half-split and withdraw spatial claims.

### 2. Prediction and robustness commands silently evaluate a regenerated default split, not the run’s configured split

**Evidence.** Current training reads `cfg.data.split.mode`, `.seed`, and `.ratio` and supports CSV selections (`train.py`, lines 119–136). The canonical config composes `data/split` as a nested group (`conf/config.yaml`, lines 1–5; `conf/data/split/random.yaml`, lines 1–3). In contrast, `predict.py`, lines 130–137, `resilience_test.py`, lines 117–124, and `checkerboard_test.py`, lines 214–221, look for obsolete flat keys `cfg.data.split_seed` and `cfg.data.split_ratio`, then fall back to seed 42 and 80/10/10 dynamic splitting. None handles CSV mode. Yet `docs/tutorials/04_inference_evaluation_plots.md`, lines 30–38, calls `--use-test-split` “the held-out test partition,” and Tutorial 5, lines 67–75, requires the same documented evaluation split and event identities.

**Consequence.** Any run trained with CSV selections or non-default split overrides is evaluated on different events than the student believes. Test-set isolation can be broken and comparisons can mix samples while still producing finite, correctly shaped CSVs and plausible plots.

**What would resolve it.** Centralize split reconstruction so training and every evaluation entry point consume the resolved `data.split` mode/seed/ratio or exact stored selections, and persist the evaluated event identities with each result.

### 3. The documented environment cannot be recreated from the repository

**Evidence.** `requirements.txt`, lines 1–10, leaves almost every numerical dependency unpinned and requires `graphnet>=1.0.0`. `docs/graphnet-boundary.md`, lines 35–51, admits that public releases may not contain the required `IceMixNodes` and `GraphNeTDataModulecustom` interfaces and instructs the student to recover an exact commit/environment from the departed author’s PACE files. `docs/pace-phoenix.md`, lines 39–61, further states that the requirement conflicts with this checkout and that the only locally working PyTorch/PyG combination is “not a replacement” for the missing PACE export. No environment export, lockfile, supported GraphNeT commit, or self-contained installation procedure is checked in.

**Consequence.** The new student has no repository-only path to obtain an environment that both resolves and supplies the APIs needed even for the smoke test. If the departed author’s files or Slack response are unavailable, work stops before training; if the student improvises with a public GraphNeT release, imports or behavior can drift.

**What would resolve it.** Check in a credential-free environment specification that pins the known-compatible GraphNeT commit and PyTorch/PyG stack, plus a clean bootstrap/import smoke procedure independent of the former author’s account.

### 4. The documented plotting command silently consults the former author’s personal reference CSV

**Evidence.** Tutorial 4 presents `python ice_mix/generate_plots.py --base-dir ...` as the aggregate plotting command (`docs/tutorials/04_inference_evaluation_plots.md`, lines 84–98). That command always invokes `plot_master.py` (`generate_plots.py`, lines 175–186). `plot_master.py`, line 20, hard-codes `/storage/home/hcoda1/8/cfilho3/.../JointLargeTC0.04results_LRNEW.csv`; lines 202–208 load it automatically when present and otherwise only warn, while lines 123–138 label its curve `TANGO (Ref)`. Neither the tutorial nor the plotting CLI identifies that input, its provenance, its selection, or that it belongs to the departed user. The job-script page’s account-string warning does not cover this Python path.

**Consequence.** On an inherited filesystem the student can silently include an unexplained stale comparison sample in master physics plots; on a clean account the same documented command silently omits it. Two students therefore get materially different figures from the same command, and one may publish a “TANGO” comparison whose data provenance they cannot state.

**What would resolve it.** Remove the implicit personal path and require an explicit, provenance-documented reference CSV argument (or an explicit no-reference mode) recorded in plot metadata.

### 5. None of the handoff documentation or local verification evidence is versioned

**Evidence.** At audit commit `4394131647b4a581e7d4923361b2814ab9e03ff5`, `git ls-files` returns no path for `ice_mix/README.md`, `ice_mix/docs/`, `ice_mix/MEMORY.md`, `ice_mix/TODO.md`, `ice_mix/QUESTIONS.md`, or any folder README; `git status --short` reports them all as untracked. The documentation agent’s code/docstring changes are only working-tree modifications as well. Worse, `TODO.md`, line 8, declares “Added `docs/_local_sample/` to `.gitignore`”; `.gitignore`, line 150, and `docs/.gitignore`, line 1, ignore that directory, and `git status --ignored` reports the whole fixture as `!!`. `git ls-files ice_mix/docs/_local_sample` is empty. Yet the docs call it a self-contained handoff sample and base multiple `VERIFIED-LOCAL` claims on it (`docs/_local_sample/README.md`, lines 1–25; `docs/handoff-status.md`, lines 35–48).

**Consequence.** A fresh clone of the repository contains none of the handoff, tutorials, ledgers, checker scripts, sample databases, checkpoint, prediction tables, metrics, or Slurm reports audited here. The only reproducible evidence for the local claims disappears with this laptop/worktree, so the student starts from the pre-handoff repository and cannot even discover what is missing.

**What would resolve it.** Commit the documentation and intended source-docstring changes, and deliver the large/sensitive fixture through an approved durable artifact store with a tracked manifest, checksums, retrieval instructions, and a small tracked verifier rather than an ignored laptop-only directory.

## MAJOR

### 6. The numeric physics/feature contract is undocumented and the only derivation is contradictory

**Evidence.** `src/utils.py`, lines 261–264, imports the seven-feature `FEATURES.ICECUBE86` list; `src/models/transformer.py`, lines 96–123 and 281–285, forwards only the configured six features into GraphNeT Fourier and spacetime encoders. `conf/attention/baseline.yaml`, lines 22–31, supplies the unexplained scales 4096, 1024, 18, clipping ±4, and 1024. The current guide (`docs/configuration.md`, lines 61–63) calls them conditioning choices, gives neither equations nor inherited GraphNeT normalization, and says their derivation is unverified. The explicitly stale `config_parameters_guide.md`, lines 75–85, instead claims that 18 approximates propagation speed “in deep ice.” The inherited implementation actually combines normalized position and time in a signed interval; IceCube86 divides position by 500 and maps time as `(t-10000)/30000`, while the spacetime encoder multiplies time differences by 18 (`src/graphnet/models/detector/icecube.py`, lines 21–48; `src/graphnet/models/components/embedding.py`, lines 174–191).

The same gap applies to the joint-loss scale: the overview gives only `alpha=0.026` and the equation (`docs/index.md`, lines 19–25), while the implementation adds `0.026 ×` a Euclidean position error to a 3D-vMF negative log-likelihood (`train.py`, lines 281–288; `src/metrics_logging.py`, lines 89–103). No provenance, tuning criterion, or consequence of changing that cross-unit weight is documented.

**Consequence.** A student changing units, detector normalization, feature order, or encoder scales cannot tell which transformations preserve the trained model’s meaning. The stale physical explanation can also be mistaken for an ice calibration, quietly changing the pairwise attention geometry rather than causing a shape error.

**What would resolve it.** Document the exact ordered features, inherited normalization equations, encoder equations, units after each transform, and the provenance/status of every scale and clipping bound in one authoritative current page.

### 7. Comparison plots infer experiment identity from job order and do not match events

**Evidence.** Tutorial 6 requires comparing fine-tuned and base checkpoints “on the same validation events” (`docs/tutorials/06_model_changes_and_fine_tuning.md`, line 72), and Tutorial 5 similarly requires identical event selections (`docs/tutorials/05_robustness_studies.md`, lines 67–75). The comparison loaders do not require `event_no` at all (`generate_fine_tune_comparison_plots.py`, lines 139–169; `generate_nutau_comparison_plots.py`, lines 157–191). They independently bin each table and divide aggregate medians (`generate_fine_tune_comparison_plots.py`, lines 216–237; `generate_nutau_comparison_plots.py`, lines 219–249). Worse, the tau script labels the two largest job IDs as “without” and “with” tau solely by order (`generate_nutau_comparison_plots.py`, lines 107–134), and the fine-tune script independently takes the largest job ID for each project name (`generate_fine_tune_comparison_plots.py`, lines 93–135). No resolved config, checkpoint lineage, database version, split, or event-set equality is checked.

**Consequence.** A later rerun, failed/resubmitted job, or split change can be mislabeled as the intended treatment and compared against a different event population. The resulting ratio curves remain smooth and plausible, so sample-composition changes can be reported as model or tau-training effects.

**What would resolve it.** Select runs from explicit manifests, require matching data/split provenance, validate identical `event_no` sets (or deliberately document an unmatched-population analysis), and fail rather than infer treatment from job-number chronology.

### 8. The “recommended” production family contains no submit-ready base launcher

**Evidence.** The catalog calls the base/resume files “the recommended workflow family” (`docs/job-scripts.md`, lines 71–87), and Tutorial 3 directs the student to submit a variant launcher after manually replacing values and adding three-flavor staging (`docs/tutorials/03_base_training_and_resume.md`, lines 13–55). The named baseline `run_standard.sbatch` declares itself deprecated in favor of `run_training.sbatch` (`run_standard.sbatch`, line 15), extracts literal YAML strings containing `${oc.env:DATA_ROOT}` and therefore stages nothing (`run_standard.sbatch`, lines 22–34; `conf/data/standard.yaml`, lines 1–5), and hard-codes eight ranks after filtering GPUs (`run_standard.sbatch`, lines 41–66 and 90–93). The supposed successor `run_training.sbatch` dynamically counts GPUs but copies only nu_mu and nu_e (`run_training.sbatch`, lines 46–52), omitting the current required nu_tau database. The same broken literal-YAML staging and fixed rank count recur in the documented rotation/drop base launchers. A mechanical audit covered all 42 `.sbatch` files and both shell helpers; 11 scripts hard-code eight ranks, and only the shared helper used by newer experimental workflows contains all three standard basenames. `MEMORY.md`, lines 18–21, nevertheless turns the idealized behavior into durable fact by saying the launcher copies all three databases and always uses the healthy-GPU count.

**Consequence.** The first “step-by-step” production submission requires the student to design an unprovided hybrid launcher. Following a named base file either bypasses local staging for all data, omits tau data from staging, or launches more DDP ranks than healthy GPUs, wasting allocation time and making failures appear to be model or PACE problems.

**What would resolve it.** Provide one successor-owned, shell-checked and Phoenix-smoke-tested base launcher that fails preflight unless all three databases, output parents, environment, and rank count are valid, then derive variants from that path.

### 9. Documented stochastic studies are not reproducible from recorded configuration

**Evidence.** The canonical rotation config sets `rotation_seed: null` (`conf/data/standard.yaml`, lines 20–21). With `null`, `RandomRotationCallback` seeds its private CPU generator from nondeterministic entropy rather than the global run seed (`src/utils.py`, lines 177–205), but `docs/configuration.md`, lines 11–18 and 34–39, does not warn that the run seed does not control augmentation. Separately, resilience subsampling uses unseeded `numpy.random.choice` (`resilience_test.py`, lines 247–274), and forced token-drop masks consume an unrecorded RNG stream across drop percentages (`resilience_test.py`, lines 276–290; `src/models/transformer.py`, lines 238–270). Tutorial 5, lines 19 and 67–75, tells the student to record seeds and compare identical events, but exposes no resilience seed and does not state the rotation exception.

**Consequence.** Re-running the same resolved config can train on different rotations, and repeated robustness runs can choose different 1% samples and masks. Apparent changes between models or removal levels can therefore be sampling noise that cannot be reconstructed from the saved configuration.

**What would resolve it.** Derive all augmentation, subsampling, and perturbation generators from explicit persisted seeds and document whether masks are independent, nested, or shared across comparison levels.

### 10. The handoff falsely implies that supplied physics plots use simulation weights

**Evidence.** The local-sample README states that `oneweight` “is the simulation weight used by the existing postprocessing” (`docs/_local_sample/README.md`, lines 35–37). In code, `oneweight` is only appended to truth/output attributes (`src/utils.py`, line 264; `predict.py`, line 256; `resilience_test.py`, line 230; `checkerboard_test.py`, line 314). No plotting Python file references it. The shared statistic computes ordinary medians and 16th/84th percentiles from raw rows (`plot_utils.py`, lines 153–205), and all master/fine-tune/tau plots call that unweighted function. Elsewhere the glossary admits that oneweight’s normalization/use is undefined (`docs/glossary.md`, line 21), while Tutorial 4 tells the student to read plotting source before interpreting weighted/unweighted distributions (`docs/tutorials/04_inference_evaluation_plots.md`, lines 84–98). The sample README assertion carries no verification marker despite the root policy.

**Consequence.** A student can reasonably describe supplied resolution curves as simulation-weighted when they are event-count-weighted. For flavor/topology samples whose generated spectrum differs from the target population, that changes the physics quantity being plotted without changing file shape or producing an error.

**What would resolve it.** State unambiguously that current plots are unweighted, define the intended oneweight normalization/selection, and implement an explicit selectable weighted statistic with that choice recorded in plot metadata.

### 11. The ledgers close unresolved prerequisites as “answered” and report no remaining work

**Evidence.** `TODO.md`, lines 3–20, marks the documentation, runtime verification, contradiction scan, and “final” consistency pass Done; lines 22–28 say both Blocked and Deferred-data are “None.” The supposedly final documentation itself still lists seven unresolved cluster/environment/data prerequisites (`docs/handoff-status.md`, lines 7–25). `QUESTIONS.md` marks Q002 “answered” after asking for exact onboarding, queue, allocation, and checkpoint procedures but receiving only “ask Jiyuan” (`QUESTIONS.md`, lines 33–40); marks the requested database provenance, preprocessing, units, and authoritative location answered with the same deferral (`QUESTIONS.md`, lines 42–49); and marks the exact environment question answered by saying the student must search the former author’s PACE folders (`QUESTIONS.md`, lines 95–98). `MEMORY.md` still lists these as open uncertainties at lines 128–132, then claims all markers and local-versus-PACE statements reconcile at line 144. It also preserves the false checkerboard and launcher descriptions cited in findings 1 and 8.

**Consequence.** An agent handed only these ledgers has no honest resume point: the task list says nothing remains, the question ledger falsely closes the missing prerequisites, and the durable memory teaches two wrong execution/physics behaviors. Work that should be escalated or verified is instead silently treated as complete.

**What would resolve it.** Reopen every externally deferred question with a concrete owner/evidence requirement, put the known code/documentation contradictions and missing environment/artifact delivery into TODO/Blocked, and remove “final consistency” claims until an independent audit passes.

## MINOR

### 12. Mechanical docstring coverage conceals missing and vacuous documentation on the model’s change surface

**Evidence.** A temporary AST inventory mechanically parsed all 36 `.py` files: all 36 modules have module docstrings, but only 22/23 classes, 54/72 methods, and 104/119 public top-level functions have docstrings. All 15 undocumented public functions are fixture-builder/checker functions under `docs/_local_sample`; 17 undocumented methods are in live `src`, including token-drop generator, rotation helpers, EMA construction, joint-shape validation, and physics-metric accumulation. The central `IceMix` class docstring is only “IceMix model (formerly DeepIce)” (`src/models/transformer.py`, lines 33–34), and its constructor documentation stops at `drop_chance`, omitting all nine numeric encoder/network controls that remain in the signature (`src/models/transformer.py`, lines 55–91). `Block` is documented only as “Transformer block” (`src/models/layers.py`, line 513). Separately, `test_fast_splits.py`, `test_labels_layout.py`, and `test_splits.py` are undocumented historical scripts named as pytest tests; they execute hard-coded another-user PACE database access at import time (`test_fast_splits.py`, lines 42–44; `test_labels_layout.py`, lines 13–39; `test_splits.py`, lines 8–12). The tutorial names only three curated tests but never says ordinary repository-wide test discovery includes these scripts (`docs/tutorials/06_model_changes_and_fine_tuning.md`, lines 23–32).

**Consequence.** A student modifying the backbone or its stochastic/metric plumbing still has to reverse-engineer the exact code paths, and a conventional `pytest`/collection attempt can fail on a foreign storage path before reaching the maintained tests. Both are recoverable, but they create precisely the source archaeology and misleading first test failure that the handoff should prevent.

**What would resolve it.** Document the constructor and stateful private contracts that changes depend on, label/exclude historical diagnostics from test discovery, and state the maintained test boundary explicitly.

### 13. The local evidence page gives a self-contradictory command sequence

**Evidence.** `docs/_local_sample/README.md`, lines 41–46, tells the reader to `cd ice_mix/docs/_local_sample`. Without telling them to return to the repository root, lines 48–53 then prescribe `PYTHONPATH=ice_mix python ice_mix/docs/_local_sample/verify_graphnet.py`. Followed in sequence, that resolves the script below `ice_mix/docs/_local_sample/ice_mix/docs/_local_sample/verify_graphnet.py`; the audit reproduced the resulting “can't open file” error with `/usr/bin/python3`. The first command also assumes a `python` alias, which this documented workstation shell does not have; `python3 verify_sample.py` does pass.

**Consequence.** The first stronger verification command fails before importing anything, misleading the student into debugging the recovered environment or GraphNeT when the immediate problem is the documentation’s working directory.

**What would resolve it.** Give commands with an explicit repository-root working directory (and the supported interpreter/environment invocation) or make every path relative to the sample directory consistently.

## NIT

_No nits recorded; low-consequence polish is intentionally deferred._

## What is genuinely solid

- The Slurm catalog really does cover all 42 `.sbatch` files and both shell helpers, and it candidly identifies inherited account paths, two-flavor staging, fixed-rank launchers, unsafe `cd`, and deprecated probes. All 44 scripts pass `bash -n`.
- The staged artifacts are internally credible: all six dependency-light checks reproduce; the 337 MiB checkpoint is a substantive PyTorch archive; the metrics contain 31 finite train and 31 finite validation summaries; the nine-row inference output, 50-row plot sample, and success/failure log landmarks exist. No fabricated output was found.
- The GraphNeT boundary page correctly warns that IceMix depends on checkout-specific interfaces, the production-versus-laptop scale warning is accurate, CSV mode fails fast on its known two-versus-three-file mismatch, and the dead/historical classifications for `src/trainer.py`, old launchers, and node probes held up.

## What could not be audited from this environment

- PACE was unreachable, so current onboarding, allocation/QoS behavior, node health, `$TMPDIR` staging, eight-L40S DDP, W&B, checkpoint resume, runtime, memory, and full-scale convergence remain unverified.
- Production database provenance, authoritative units, selections, population weights, and full event counts were unavailable; the small ignored fixture establishes structure and internal consistency, not scientific provenance.
- None of the five installed conda environments contains PyTorch/GraphNeT/Hydra/PyG/Lightning, and the NVIDIA driver is currently unavailable. The historical strict checkpoint load, 21 tests, Hydra composition, and full CPU inference therefore could not be rerun, although their artifacts are consistent with the claims.

## Three highest-benefit changes

1. **Make the handoff exist outside this working tree:** commit the documentation and intended docstring changes, and publish a tracked, checksummed retrieval manifest for the ignored verification bundle.
2. **Stop silent scientific mislabeling:** withdraw the checkerboard claim and make all evaluation/comparison paths fail unless split provenance, event identity, treatment identity, and weighting mode are explicit and validated.
3. **Create one reproducible operational path:** pin the checkout/environment and provide one successor-owned launcher that preflights three-flavor staging, output paths, healthy-GPU rank count, and a low-cost Phoenix smoke run.
