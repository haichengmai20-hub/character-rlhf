# Character-RLHF SFT 数据工程：从真实 Session 到 Multi-turn Character SFT Actor

## Core Pipeline

    Production Conversation
    ↓
    Session Reconstruction
    ↓
    Trajectory Sampling
    ↓
    Pre-Gemini Rule
    ↓
    Gemini Teacher Generation
    ↓
    Post-Gemini Validation
    ↓
    Dual Judge
    ↓
    Repair
    ↓
    Balance
    ↓
    Assistant-only SFT

## Data Funnel

  Stage                    Groups   Assistant Turns
  ---------------------- -------- -----------------
  Candidate                  6922             41532
  Pre-Gemini Rule            5760             34560
  Gemini + Post Filter       5184             31104
  Accept                     2700             16200
  Repair Success             1238              7428
  High Quality Pool          3938             23628
  Final SFT                  3780             22680

## SFT Objective

$$
\mathcal L_{SFT}
=
-
\frac{
\sum_i M_i\log\pi_\theta(y_i|y_{<i})
}{
\sum_i M_i
}
$$

Assistant tokens contribute to loss; system and user tokens are masked.

## Sampling

Head learns cold start:

    Role Card
    u1 → a1*
    ...
    u6 → a6*

Continuation learns existing dialogue state:

    History context (mask)
    u1 → old_a1
    ...
    u6 → old_a6

    Supervision
    u7 → new_a7*
    ...
    u12 → new_a12*

## Quality Pipeline

Pre-Gemini: - Role card validation - User corruption filtering - Session
consistency - Token feasibility

Teacher: - Gemini 2.5 Flash - causal turn-by-turn generation

Post-Gemini: - API status - Safety - Empty/truncated output -
Disclosure - Language detection - Repetition - Length - Format

Judge: - DeepSeek - Gemini 2.5 Pro

Score:

$$
S=0.30S_f+0.30S_r+0.25S_l+0.15S_{len}
$$

## Training

-   30B model scale
-   8× RTX5090 32GB
-   NF4 QLoRA
-   LoRA rank 32
-   DeepSpeed ZeRO-2/FSDP
-   assistant-only MLE

## Final Loop

    Real User Behavior
    ↓
    Session State
    ↓
    Teacher Target
    ↓
    Quality Control
    ↓
    Balanced Dataset
    ↓
    QLoRA SFT
    ↓
    Character SFT Actor
