# Paper foundation

Advisor is grounded in the paper:
- How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models
- arXiv: https://arxiv.org/abs/2510.02453
- PDF: https://arxiv.org/pdf/2510.02453

This paper is the basis for major design and roadmap decisions in this repo.

## What that means in practice

The repo should be optimized around the paper's core loop:
1. build a task packet for a specific instance
2. run a small advisor model that emits dynamic per-instance advice
3. inject that advice into a stronger black-box executor
4. measure the executor outcome with verifiers, human review, or both
5. convert outcomes into replayable traces and rewards
6. use those rewards to evaluate and improve the advisor model

## Repo-level design rules

1. Keep the advisor separate from the executor
- Advisor is the steering layer.
- The frontier or black-box model remains the main executor.
- Avoid blending advisor logic into executor-specific hacks when a generic interface is possible.

2. Prefer generic abstractions in the core
- Core packet, advice, reward, replay, and orchestration layers should be domain-agnostic.
- Coding, UI/image, research, and other domains should attach through adapters.
- If a feature is coding-specific, keep it behind a coding adapter or extension field.

3. Optimize for measurable lift, not plausible demos
- New work should improve a repeatable benchmark or online outcome.
- "It looked good once" is not enough.
- Baseline vs advisor-assisted comparisons should stay central.

4. Treat reward as a first-class product component
- Reward is not a future add-on.
- Every execution path should make it possible to recover outcome signals, human ratings, or verifier results that can become reward.
- Dataset lineage, reward configs, and replayability matter.

5. Keep traces replayable
- Packet inputs, injected advice, executor outputs, verifier outputs, and reward calculations should be traceable and reproducible.
- Architectural convenience should not destroy replayability.

6. Build for transfer
- The paper's value includes advisor transfer across executors.
- Avoid tight coupling to one frontier model or one agent harness unless the coupling is explicitly isolated.

## Non-goals for the core

The core should not hardcode assumptions like:
- source repo/file ranking is always the main retrieval problem
- every task is a coding task
- executor success is only build/test success
- advice must always be formatted as coding instructions

Those may exist in adapters, not in the product core.

## Pull request bar

A strong change in this repo should answer at least one of these:
- does this improve the generic advisor -> executor -> reward loop?
- does this make the system more replayable or more measurable?
- does this improve transfer across domains or executors?
- does this improve reward quality, evaluation quality, or training quality?

If the answer is no, the change should justify why it still serves the paper-aligned roadmap.
