# First experiment: a coding model that uses tools

Companion to the [project outline](reasoning-coder-outline.md). All quantities below are proposed pilot sizes, not established requirements.

## Scope and baseline

Start with small bug fixes in one language. Elixir is a candidate given this repository; choose based on actual intended use. Defer broad autonomous feature development.

1. Assemble roughly 50–100 reproducible tasks with starting commits, dependencies, and independent correctness checks.
2. Separate training, development, and final evaluation tasks before collecting trajectories. Split by repository where possible; exclude related fixes and near duplicates across splits.
3. Run the unmodified candidate model through the same tools and budgets we will use after training.

Record solve rate, regressions, invalid tool calls, timeouts, generated tokens, and wall time. Count one attempt separately from success after multiple samples. A small evaluation gives directional evidence; repeat runs and enlarge it before claiming reliable improvements.

## Data to collect

Each trajectory needs:

- Task, repository revision, environment version, and available tool schemas.
- Ordered assistant messages, tool calls, actual tool responses, and final patch.
- Independent test results, resource usage, model/version, and sampling settings.
- Provenance and permission to use the code and any teacher-generated examples.

Collect a few hundred verified trajectories initially; expand only if useful. Include successful recovery from failures, searches, and test-driven corrections. Keep failed attempts for diagnosis or preference comparisons rather than labeling them all as demonstrations to imitate. A passing test alone does not prove a patch is correct: inspect a sample and strengthen weak checks.

## First training run

Choose a model after checking hardware, license, baseline coding quality, context needs, and native tool support. Try adapter SFT, preserve the tokenizer/chat template, and inspect several serialized examples before training. Compare trained and original models with identical tool access and token/call budgets.

Continue only if held-out task success improves without unacceptable regressions or cost. If training loss falls but task success does not, investigate data leakage, weak examples, formatting, and runtime failures before increasing scale.

## Optional RL experiment

Use training-only tasks with reliable automated grading. Sample multiple complete tool-use trajectories per task and reward independently verified correctness. Keep regression tests and grading files outside the model's writable workspace. An unparseable call, timeout, or changed test runner must not manufacture a success.

Add only modest efficiency penalties after correctness is working. Do not reward verbose plans or tool-call counts. If every attempt fails, simplify the curriculum or improve demonstrations: a flat failure signal provides little learning direction.

## Compute and cost

No GPU or dollar budget has been specified. Start by measuring one inference rollout and one training step on the intended context length.

- Raw weights alone need approximately `parameters × bytes per parameter`; an 8B model at 2 bytes needs about 16 GB, before caches or training memory.
- Quantization and adapters can reduce memory needs, but activations, context length, optimizer state, and rollout concurrency still matter. Weight size is not a VRAM recommendation.
- Estimate training cost from measured GPU-hours. Estimate data/rollout cost from tasks × attempts × measured tokens and execution time. Include teacher inference separately if used.

Set a fixed pilot spend cap before collecting data at scale. RL multiplies generation and test execution; it should earn its additional cost against the SFT baseline.

## Decisions for our next discussion

1. Is the priority a useful personal coder, ML research, or a pure Elixir implementation?
2. Which GPU/VRAM is available, and what training spend is acceptable?
3. Which language and task type should the model handle first?
4. Does “my own” mean customized open weights or pretraining from scratch?

First deliverable: a working tool loop, reproducible evaluation results, and a baseline failure analysis. Those tell us what is worth training.
