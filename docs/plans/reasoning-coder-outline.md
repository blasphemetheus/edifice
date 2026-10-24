# A personal reasoning model for coding

Working notes — 2026-09-26. Goal: explore a model we can run, customize, and improve for our coding work. This is a proposal, not a committed implementation or compute budget.

## What we would build

Two parts: model weights that choose useful next actions, and a runtime that executes tools and returns results. The model emits a structured tool request; our code actually reads files, applies edits, or runs tests.

“Reasoning” here means solving unfamiliar, multi-step tasks: inspect code, form a hypothesis, change something, test it, and recover from mistakes. Longer explanations alone are not evidence of better reasoning.

## Three possible routes

| Route | What we own | Main cost | Fit |
| --- | --- | --- | --- |
| Existing weights + our tool runtime | Agent behavior and execution environment | Inference and engineering | Establish a useful baseline |
| Fine-tune existing weights | Above, plus custom adapters or weights | Training data and GPU time | Recommended first model project |
| Pretrain from random weights | Architecture, tokenizer, weights, training pipeline | Corpus preparation and much more compute | Research or learning objective |

Recommendation: baseline → supervised fine-tuning → evaluate → optionally reinforcement learning. Start with a coding-capable model that already has useful reasoning and tool behavior. Fine-tuning should specialize it to our tasks; creating broad coding competence from scratch is a separate undertaking.

## The tool loop

```text
Task + relevant code + tool schemas
    → model returns tool call
    → runtime validates and executes it in an isolated workspace
    → tool result becomes the next observation
    → repeat until final answer or budget limit
```

Initial tools: `list_files`, `search`, `read_file`, `apply_patch`, and `run_tests`. Define argument schemas, call IDs, error responses, and output limits. Match the model's native chat/tool template during training and inference.

Keep tool outputs as observations, not instructions. Bound runtime, output size, and total calls; run repository code without host credentials. The model must learn to respond to failed tests and malformed calls, too.

## Training stages

1. **Supervised fine-tuning (SFT):** imitate successful coding trajectories, including tool calls and recovery steps. Start with LoRA adapters, which train a smaller set of additional parameters. Train on assistant actions while masking user messages and tool results; inspect the actual token masks. TRL supports conversational SFT, assistant-only loss, and adapters. [SFT documentation](https://huggingface.co/docs/trl/sft_trainer)
2. **Iterative data improvement:** sample new attempts, independently verify their patches, curate good examples, and retrain. This is a simpler experiment than immediately building online RL.
3. **Optional reinforcement learning (RL):** sample several attempts per task, score their outcomes, and update the model toward better attempts. GRPO is one available method; tool integration and trajectory handling depend on the pinned trainer version. [GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer)

DeepSeek-R1 provides evidence that RL can develop reasoning behavior and that reasoning capabilities can transfer through distillation. It does not establish that a small personal training run will reproduce its results. [Research paper](https://arxiv.org/abs/2501.12948)

Use observable actions, test feedback, and optional concise plans as training material. Hidden reasoning from another assistant is neither necessary nor assumed available.

## If we really mean training from scratch

Build or select a tokenizer and causal language-model architecture; prepare and deduplicate a licensed text/code corpus; pretrain with next-token prediction; evaluate basic code generation; then add instruction, tool-use, and reasoning training. Start with a small educational model to validate this entire pipeline. A successful small experiment would demonstrate the pipeline, not parity with a general coding assistant.

## Where Edifice fits

Edifice describes itself as an Nx/Axon architecture library. It could support architecture research or a native Elixir training experiment. We have not audited pretrained checkpoint compatibility, tokenizer support, generation performance, or distributed training for this project.

Proposed practical split: Elixir for orchestration and task environments; an established training stack for the first fine-tune. If a pure Elixir implementation is the main objective, first scope a separate compatibility/performance spike.

Next: [first experiment and decision gates](reasoning-coder-first-experiment.md).
