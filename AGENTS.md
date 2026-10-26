# Edifice contributor instructions

Edifice implements neural network architectures and reusable blocks in Elixir
with Nx and Axon. ExPhil is a downstream consumer. Keep the library useful to
other callers as well. These guidelines adapt the conventions in the sibling
Nx repository's `AGENTS.md` to Edifice.

## Languages and reusable tooling

- Prefer Elixir for implementation, benchmarks, orchestration, data processing,
  validation, and report generation. Use Nx/Axon for numerical and model code.
- Use Rust when native code or systems tooling is appropriate. Keep existing
  CUDA/native integration conventions when working on those components.
- Do not fall back to Python, including temporary scripts and inline analysis
  snippets, unless the user explicitly requests Python. Existing Python tooling
  does not justify adding more.
- Deliver reusable tools that the user can inspect, rerun, and maintain. Put
  shared logic in modules; use `.exs` entry points or Mix tasks as appropriate.
  Avoid a new framework when a small module will do.
- Shell commands are fine for invoking tools and inspecting files. Do not move
  substantial application logic into shell to avoid writing Elixir.

## Explain clearly

- Explain things so a beginner can understand. Lead with the main point.
- Be direct. Use simple sentences and familiar words. Define necessary jargon.
- Use concrete examples when helpful. Preserve technical accuracy and important
  limitations without talking down to the reader.
- Apply this style to conversation, docs, comments, reports, and commit/PR
  descriptions. Describe what changed and why. Avoid filler and elaborate prose.

## Environment and shared work

- Use this repository's `devenv shell` for Elixir, Erlang, Mix, and native tools.
  Sibling repos may use different OTP versions; do not reuse their build output.
- Check for active training or GPU work before building or benchmarking. A native
  library rebuild can crash another process using that shared library. Use an
  existing bare-VM runner when rebuilding would interfere with a live workload.
- Do not stop someone else's jobs. For timing runs, record contention and exclude
  contaminated measurements from claims about idle performance.
- Preserve unrelated edits in shared checkouts. Stage only the files belonging
  to the task. Keep commits focused on one concern.

## Code and numerical behavior

- Keep portable Nx behavior separate from backend-specific optimizations. EXLA
  is an optional dependency; generic library code must not require it at runtime.
- Keep public APIs about the caller's problem. Do not expose compiler details
  unless callers need them to use the API correctly.
- Prefer the simplest working mechanism and clear names. Avoid unnecessary
  processes, protocols, options, or duplicate implementations.
- Fix architecture approximations against the actual equations. Preserve
  checkpoint shapes and names where possible, and document behavior changes.
- Compute loss/objective math and its backward path in f32 even when network
  compute uses reduced precision. Cast at the loss entry point, where every
  caller receives the protection.
- Keep comments focused on reasoning that is not obvious from the code. Use Git
  history for issue/PR provenance instead of embedding issue numbers in comments.

## Validation

- Run focused tests for the changed behavior. Do not run the full suite after a
  small change. See `docs/TESTING.md` for targeted commands and backend tags.
- Prefer a regression that fails before the fix. Extend an existing test group
  when it already covers the affected API.
- Use `Nx.Testing.assert_equal/2` or `assert_all_close/3` for tensor comparisons
  where appropriate. Exact bytes are appropriate when testing a bit-for-bit
  contract such as a random-number stream; explain that requirement.
- Cover relevant shapes, types, edge cases, gradients, vectorized axes, and
  carried-state behavior. Choose the cases that exercise the change.
- Fix numerical tolerances at the specific test that needs them. Do not widen
  tolerances or skip a backend to conceal a regression.
- Build large/shared tensors at runtime rather than in test module attributes.
- For performance changes, separate compilation from warm inference. Record
  backend, shapes, inputs, timing method, and resource contention. Verify outputs
  before claiming a speedup; include tail latency when deadlines matter.
- Run `mix format` on changed Elixir files before committing, when shared-job
  safety permits it. Run the checks appropriate to the change.

## Documentation

- Use normal `iex>` examples for doctests.
- Introduce nontrivial machinery by explaining the problem it solves before its
  API details. Include a runnable reproduction command for benchmark tooling.
