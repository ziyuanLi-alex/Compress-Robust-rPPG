# Compression Robustness Experiment Design

**Created:** 2026-03-26

## Purpose

This note records the current experiment-design decisions for studying robustness to compressed video in this repository.

The central goal is to keep the main study as strong and apples-to-apples as possible:

- stay within the PhysFormer family for the main comparison
- reuse PhysFormer checkpoints pretrained on different source datasets
- compare robustness-enhanced PhysFormer variants under the same target compression protocol
- avoid mixing too many architectural families into the core claim

## Core Experimental Principle

The main paper story should be:

`source PhysFormer pretraining dataset -> PhysFormer-family architecture -> compression-aware fine-tuning -> compressed-video evaluation`

The source checkpoints currently available are:

- `UBFC-rPPG_PhysFormer_DiffNormalized.pth`
- `PURE_PhysFormer_DiffNormalized.pth`
- `SCAMPS_PhysFormer_DiffNormalized.pth`

These released PhysFormer checkpoints share the same parameter structure, so they can be treated as interchangeable initialization points for PhysFormer-based studies.

The main architectures of interest are:

- vanilla `PhysFormer`
- `STVEN-PhysFormer`
- `QAFC-PhysFormer`

## Recommended Main Study

The main experiment should fix the target dataset and target compression protocol, then vary only:

- source checkpoint: `UBFC`, `PURE`, `SCAMPS`
- architecture: `PhysFormer`, `STVEN-PhysFormer`, `QAFC-PhysFormer`
- test compression level: `CRF 0, 16, 18, 20, 22, 24`

This gives a clean three-factor study:

1. effect of source pretraining dataset
2. effect of robustness-enhanced architecture
3. effect of compression severity at test time

Recommended interpretation:

- `vanilla PhysFormer` shows how much robustness comes from source pretraining alone
- `STVEN-PhysFormer` shows whether a restoration frontend improves robustness over the same backbone family
- `QAFC-PhysFormer` shows whether explicit quality-aware conditioning improves robustness over the same backbone family

## Suggested Experimental Sequence

To keep the study focused and efficient, run the experiments in two stages.

### Stage 1: Main Apples-to-Apples Matrix

Use one fixed target training setup and evaluate all methods across the same CRF sweep.

- `3 source checkpoints x 3 methods x 1 target fine-tuning setup x 6 test CRFs`

This stage answers:

- which source checkpoint transfers best to compressed-video robustness
- whether STVEN and QAFC consistently improve over vanilla PhysFormer
- whether robustness gains hold across mild and severe compression

### Stage 2: Fine-Tuning Regime Ablation

After identifying the strongest source checkpoint, vary the fine-tuning setup:

- clean fine-tuning
- mixed-CRF fine-tuning
- single-CRF fine-tuning

This stage answers:

- whether robustness comes mainly from initialization or from compression-aware adaptation
- whether the proposed architecture still helps after compression-aware fine-tuning

## Recommended Main Claims

The strongest claims supported by this design are:

- source pretraining dataset affects compression robustness even within the same PhysFormer family
- PhysFormer-based robustness enhancements improve performance under compression beyond what source pretraining alone provides
- different robustness mechanisms may help at different compression levels

Examples:

- `STVEN-PhysFormer` may help more at severe compression by restoring useful visual content
- `QAFC-PhysFormer` may help more at moderate compression by conditioning features on quality cues

## What To Keep Fixed

For the main matrix, keep the following as fixed as possible:

- target training dataset
- preprocessing type
- chunk length
- spatial resolution
- evaluation window policy
- train/val/test split policy
- compression levels used for testing

This is important because the main purpose is not to compare pipelines broadly, but to isolate source-pretraining and architecture effects.

## Metrics To Report

Do not report only raw MAE at each CRF. The study should summarize robustness directly.

Recommended reporting:

- performance at each CRF
- average performance over compressed CRFs only
- drop from clean to worst compression
- robustness curve across CRFs
- worst-case compressed performance

Useful summary columns:

- `Clean`
- `CRF16`
- `CRF20`
- `CRF24`
- `Compressed Avg`
- `Drop from Clean`

Useful plots:

- line plot of `MAE` versus `CRF`
- line plot of `Pearson` versus `CRF`
- one line per method, with source checkpoint fixed

## Role of PhysMamba

`PhysMamba` should not be part of the core apples-to-apples study.

Reason:

- it is a different backbone family
- the main study is intended to isolate effects within the PhysFormer family
- the current contribution is based on PhysFormer-derived robustness enhancements

Recommended use of PhysMamba:

- include it only as a secondary external baseline
- do not expand the main source-pretraining matrix to include PhysMamba unless matched source checkpoints and a matched study design are also available

This keeps the paper focused:

- main story: controlled PhysFormer-family robustness study
- secondary story: comparison against one strong modern non-PhysFormer baseline

## Repo-Specific Notes

### STVEN-PhysFormer

The current implementation is already aligned with the intended reuse of pretrained PhysFormer checkpoints.

- `JointSTVENPhysFormerTrainer` loads PhysFormer weights from `MODEL.PHYSFORMER.PRETRAINED_PATH`
- the backend PhysFormer is then frozen during joint training

This makes STVEN-PhysFormer a natural branch for the main source-checkpoint comparison.

### QAFC-PhysFormer

QAFC should be treated carefully if the goal is a strict apples-to-apples comparison.

Current concerns:

- the current QAFC training config uses backbone hyperparameters that differ from vanilla PhysFormer
- the current QAFC pretrained-loader logic does not cleanly load a plain PhysFormer checkpoint into the QAFC backbone as written

Implication:

- if QAFC is included in the main apples-to-apples study, its backbone settings should be aligned with vanilla PhysFormer as closely as possible
- otherwise QAFC becomes a broader architectural variant rather than a controlled PhysFormer-family enhancement

Recommended QAFC rule for the main study:

- either fix QAFC backbone loading and align backbone dimensions with vanilla PhysFormer
- or clearly label QAFC as a partially exploratory branch rather than a fully controlled apples-to-apples branch

## Final Recommendation

The strongest primary experiment design is:

- source checkpoints: `UBFC`, `PURE`, `SCAMPS`
- methods: `PhysFormer`, `STVEN-PhysFormer`, `QAFC-PhysFormer`
- fixed target compressed-video setup
- identical CRF sweep for evaluation
- PhysMamba only as a secondary baseline

This is the recommended design because it is:

- controlled
- interpretable
- directly connected to the proposed contribution
- strong enough to support a clean paper narrative
