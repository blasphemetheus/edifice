defmodule Edifice.Recipes do
  @moduledoc """
  Pre-built training configurations with sensible defaults.

  Each recipe returns a configured `Axon.Loop` ready to run with
  `Axon.Loop.run/4`. Recipes compose Edifice's mixed precision,
  gradient checkpointing, and PEFT modules with Axon's training
  infrastructure.

  ## Quick Start

      model = Edifice.build(:decoder_only, embed_dim: 128, ...)

      # Classification
      loop = Edifice.Recipes.classify(model, num_classes: 10)
      state = Axon.Loop.run(loop, train_data, %{}, epochs: 10)

      # Language modeling
      loop = Edifice.Recipes.language_model(model, vocab_size: 32000)
      state = Axon.Loop.run(loop, train_data, %{}, epochs: 5)

      # Fine-tuning with frozen base
      loop = Edifice.Recipes.fine_tune(model, base_params, strategy: :head_only)
      state = Axon.Loop.run(loop, train_data, %{}, epochs: 3)

  ## Recipes

  - `classify/2` — Supervised classification (cross-entropy, AdamW, cosine LR, early stopping)
  - `language_model/2` — Autoregressive language modeling (causal LM loss, gradient clipping, warmup)
  - `contrastive/2` — Contrastive learning (InfoNCE loss, cosine schedule)
  - `fine_tune/3` — Transfer learning (freeze base + train head, optional LoRA)
  - `regress/2` — Regression (MSE/Huber loss, AdamW, cosine LR)

  ## Common Options

  All recipes accept these shared options:

  - `:validation_data` — When provided, attaches `Axon.Loop.validate` to log
    validation metrics each epoch. Accepts any enumerable of `{input, target}` batches.
  - `:checkpoint_path` — Directory path for saving checkpoints. When set, saves the
    best model (by validation loss if validation_data is provided, otherwise every epoch).
  """

  @doc """
  Build a supervised classification training loop.

  ## Options

  - `:num_classes` — Number of output classes (required)
  - `:learning_rate` — Initial learning rate (default: 1.0e-3)
  - `:weight_decay` — AdamW weight decay (default: 1.0e-2)
  - `:epochs` — Number of training epochs, used for LR schedule (default: 10)
  - `:steps_per_epoch` — Steps per epoch for LR schedule (default: 1000)
  - `:patience` — Early stopping patience in epochs (default: 5)
  - `:label_smoothing` — Label smoothing factor, 0.0 to disable (default: 0.0)
  - `:precision` — Mixed precision preset, nil to disable (default: nil)
  - `:validation_data` — Validation data for per-epoch evaluation (default: nil)
  - `:checkpoint_path` — Directory for saving checkpoints (default: nil)
  - `:log` — Log training progress (default: true)

  ## Returns

  Configured `Axon.Loop` with:
  - Loss: categorical cross-entropy (with optional label smoothing)
  - Optimizer: AdamW with cosine decay LR schedule
  - Metrics: loss + accuracy
  - Early stopping on loss

  ## Example

      model = Edifice.build(:vit, embed_dim: 128, ...)
      loop = Edifice.Recipes.classify(model, num_classes: 10)
      state = Axon.Loop.run(loop, train_data, %{}, epochs: 10)
  """
  @spec classify(Axon.t(), keyword()) :: Axon.Loop.t()
  def classify(model, opts \\ []) do
    _num_classes = Keyword.fetch!(opts, :num_classes)
    lr = Keyword.get(opts, :learning_rate, 1.0e-3)
    weight_decay = Keyword.get(opts, :weight_decay, 1.0e-2)
    epochs = Keyword.get(opts, :epochs, 10)
    steps_per_epoch = Keyword.get(opts, :steps_per_epoch, 1000)
    patience = Keyword.get(opts, :patience, 5)
    label_smoothing = Keyword.get(opts, :label_smoothing, 0.0)
    precision = Keyword.get(opts, :precision, nil)
    validation_data = Keyword.get(opts, :validation_data, nil)
    checkpoint_path = Keyword.get(opts, :checkpoint_path, nil)
    monitor = Keyword.get(opts, :monitor, nil)
    adaptive = Keyword.get(opts, :adaptive, nil)
    optimizer_name = Keyword.get(opts, :optimizer, :adamw)
    schedule_name = Keyword.get(opts, :schedule, :cosine_decay)
    extra_metrics = Keyword.get(opts, :extra_metrics, nil)
    reduce_lr = Keyword.get(opts, :reduce_lr_on_plateau, nil)
    log = Keyword.get(opts, :log, true)

    model = maybe_apply_precision(model, precision)

    decay_steps = epochs * steps_per_epoch
    schedule = build_schedule(schedule_name, lr, decay_steps)
    optimizer = build_optimizer(optimizer_name, schedule, weight_decay)

    loss =
      if label_smoothing > 0.0 do
        fn y_true, y_pred ->
          Axon.Losses.categorical_cross_entropy(y_true, y_pred,
            reduction: :mean,
            label_smoothing: label_smoothing
          )
        end
      else
        :categorical_cross_entropy
      end

    loop =
      model
      |> Axon.Loop.trainer(loss, optimizer)
      |> Axon.Loop.metric(:accuracy)
      |> maybe_extra_metrics(extra_metrics)
      |> maybe_validate(model, validation_data)
      |> Axon.Loop.early_stop("loss", patience: patience, mode: :min)
      |> maybe_reduce_lr(reduce_lr)
      |> maybe_checkpoint(checkpoint_path, validation_data)
      |> maybe_attach_monitor(monitor)
      |> maybe_attach_adaptive(adaptive)

    maybe_log(loop, log)
  end

  @doc """
  Build an autoregressive language modeling training loop.

  ## Options

  - `:vocab_size` — Vocabulary size (required)
  - `:learning_rate` — Peak learning rate (default: 3.0e-4)
  - `:weight_decay` — AdamW weight decay (default: 0.1)
  - `:warmup_steps` — Linear warmup steps (default: 100)
  - `:decay_steps` — Total decay steps for cosine schedule (default: 10000)
  - `:max_grad_norm` — Global gradient norm clip (default: 1.0)
  - `:precision` — Mixed precision preset (default: nil)
  - `:validation_data` — Validation data for per-epoch evaluation (default: nil)
  - `:checkpoint_path` — Directory for saving checkpoints (default: nil)
  - `:log` — Log training progress (default: true)

  ## Returns

  Configured `Axon.Loop` with:
  - Loss: categorical cross-entropy (causal LM: predict next token)
  - Optimizer: AdamW with warmup + cosine decay, gradient clipping
  - Metrics: loss + perplexity

  ## Example

      model = Edifice.build(:decoder_only, embed_dim: 256, ...)
      loop = Edifice.Recipes.language_model(model, vocab_size: 32000)
      state = Axon.Loop.run(loop, train_data, %{}, epochs: 5)
  """
  @spec language_model(Axon.t(), keyword()) :: Axon.Loop.t()
  def language_model(model, opts \\ []) do
    _vocab_size = Keyword.fetch!(opts, :vocab_size)
    lr = Keyword.get(opts, :learning_rate, 3.0e-4)
    weight_decay = Keyword.get(opts, :weight_decay, 0.1)
    warmup_steps = Keyword.get(opts, :warmup_steps, 100)
    decay_steps = Keyword.get(opts, :decay_steps, 10_000)
    max_grad_norm = Keyword.get(opts, :max_grad_norm, 1.0)
    precision = Keyword.get(opts, :precision, nil)
    validation_data = Keyword.get(opts, :validation_data, nil)
    checkpoint_path = Keyword.get(opts, :checkpoint_path, nil)
    monitor = Keyword.get(opts, :monitor, nil)
    adaptive = Keyword.get(opts, :adaptive, nil)
    optimizer_name = Keyword.get(opts, :optimizer, :adamw)
    log = Keyword.get(opts, :log, true)

    model = maybe_apply_precision(model, precision)

    schedule = warmup_cosine_schedule(lr, warmup_steps, decay_steps)

    base_optimizer = build_optimizer(optimizer_name, schedule, weight_decay)

    optimizer =
      Polaris.Updates.compose(
        Polaris.Updates.clip_by_global_norm(max_norm: max_grad_norm),
        base_optimizer
      )

    loss = :categorical_cross_entropy

    loop =
      model
      |> Axon.Loop.trainer(loss, optimizer)
      |> Axon.Loop.metric(&perplexity_metric/2, "perplexity")
      |> maybe_validate(model, validation_data)
      |> maybe_checkpoint(checkpoint_path, validation_data)
      |> maybe_attach_monitor(monitor)
      |> maybe_attach_adaptive(adaptive)

    maybe_log(loop, log)
  end

  @doc """
  Build a contrastive learning training loop.

  Uses InfoNCE (NT-Xent) loss to learn representations by pulling
  positive pairs together and pushing negatives apart.

  ## Options

  - `:temperature` — InfoNCE temperature (default: 0.07)
  - `:learning_rate` — Initial learning rate (default: 3.0e-4)
  - `:weight_decay` — AdamW weight decay (default: 1.0e-4)
  - `:decay_steps` — Cosine decay steps (default: 10000)
  - `:precision` — Mixed precision preset (default: nil)
  - `:validation_data` — Validation data for per-epoch evaluation (default: nil)
  - `:checkpoint_path` — Directory for saving checkpoints (default: nil)
  - `:log` — Log training progress (default: true)

  ## Data Format

  Each batch should be `{input, _labels}` where the input contains
  concatenated positive pairs along the batch dimension. The first
  half of the batch is view 1, the second half is view 2.

  ## Returns

  Configured `Axon.Loop` with:
  - Loss: InfoNCE / NT-Xent
  - Optimizer: AdamW with cosine decay
  - Metrics: loss

  ## Example

      encoder = Edifice.build(:mlp, layers: [128, 64])
      loop = Edifice.Recipes.contrastive(encoder, temperature: 0.1)
      state = Axon.Loop.run(loop, paired_data, %{}, epochs: 100)
  """
  @spec contrastive(Axon.t(), keyword()) :: Axon.Loop.t()
  def contrastive(model, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 0.07)
    lr = Keyword.get(opts, :learning_rate, 3.0e-4)
    weight_decay = Keyword.get(opts, :weight_decay, 1.0e-4)
    decay_steps = Keyword.get(opts, :decay_steps, 10_000)
    precision = Keyword.get(opts, :precision, nil)
    validation_data = Keyword.get(opts, :validation_data, nil)
    checkpoint_path = Keyword.get(opts, :checkpoint_path, nil)
    monitor = Keyword.get(opts, :monitor, nil)
    adaptive = Keyword.get(opts, :adaptive, nil)
    optimizer_name = Keyword.get(opts, :optimizer, :adamw)
    schedule_name = Keyword.get(opts, :schedule, :cosine_decay)
    log = Keyword.get(opts, :log, true)

    model = maybe_apply_precision(model, precision)

    schedule = build_schedule(schedule_name, lr, decay_steps)
    optimizer = build_optimizer(optimizer_name, schedule, weight_decay)

    loss_fn = fn _y_true, y_pred ->
      infonce_loss(y_pred, temperature)
    end

    loop =
      model
      |> Axon.Loop.trainer(loss_fn, optimizer)
      |> maybe_validate(model, validation_data)
      |> maybe_checkpoint(checkpoint_path, validation_data)
      |> maybe_attach_monitor(monitor)
      |> maybe_attach_adaptive(adaptive)

    maybe_log(loop, log)
  end

  @doc """
  Build a fine-tuning training loop.

  Freezes the base model parameters and trains only the head (and
  optionally PEFT adapter layers like LoRA).

  ## Parameters

  - `model` — Full Axon model (base + head)
  - `base_params` — Pre-trained base model parameters

  ## Options

  - `:strategy` — Fine-tuning strategy (default: `:head_only`)
    - `:head_only` — Freeze all base layers, train only the classification head
    - `:lora` — Freeze base, unfreeze head + any LoRA adapter params.
      Note: LoRA layers must be injected at model build time using
      `Edifice.Meta.LoRA.wrap/3`. This strategy only handles the
      freeze/unfreeze pattern for already-present LoRA layers.
    - `:full` — Fine-tune all parameters (no freezing)
  - `:head_pattern` — Regex for head layers to keep trainable (default: `~r/head|classifier|out/`)
  - `:learning_rate` — Learning rate (default: 2.0e-5)
  - `:weight_decay` — AdamW weight decay (default: 1.0e-2)
  - `:epochs` — Number of epochs for LR schedule (default: 3)
  - `:steps_per_epoch` — Steps per epoch (default: 1000)
  - `:warmup_ratio` — Fraction of total steps for warmup (default: 0.1)
  - `:precision` — Mixed precision preset (default: nil)
  - `:log` — Log training progress (default: true)

  ## Returns

  Configured `Axon.Loop` with frozen base parameters.

  ## Examples

      # Head-only fine-tuning
      loop = Edifice.Recipes.fine_tune(model, base_params, strategy: :head_only)

      # Full fine-tuning with low LR
      loop = Edifice.Recipes.fine_tune(model, base_params,
        strategy: :full,
        learning_rate: 1.0e-5
      )
  """
  @spec fine_tune(Axon.t(), map(), keyword()) :: Axon.Loop.t()
  def fine_tune(model, base_params, opts \\ []) do
    strategy = Keyword.get(opts, :strategy, :head_only)
    head_pattern = Keyword.get(opts, :head_pattern, ~r/head|classifier|out/)
    lr = Keyword.get(opts, :learning_rate, 2.0e-5)
    weight_decay = Keyword.get(opts, :weight_decay, 1.0e-2)
    epochs = Keyword.get(opts, :epochs, 3)
    steps_per_epoch = Keyword.get(opts, :steps_per_epoch, 1000)
    warmup_ratio = Keyword.get(opts, :warmup_ratio, 0.1)
    precision = Keyword.get(opts, :precision, nil)
    validation_data = Keyword.get(opts, :validation_data, nil)
    checkpoint_path = Keyword.get(opts, :checkpoint_path, nil)
    monitor = Keyword.get(opts, :monitor, nil)
    adaptive = Keyword.get(opts, :adaptive, nil)
    optimizer_name = Keyword.get(opts, :optimizer, :adamw)
    extra_metrics = Keyword.get(opts, :extra_metrics, nil)
    log = Keyword.get(opts, :log, true)

    model = maybe_apply_precision(model, precision)

    total_steps = epochs * steps_per_epoch
    warmup_steps = round(total_steps * warmup_ratio)
    schedule = warmup_cosine_schedule(lr, warmup_steps, total_steps)
    optimizer = build_optimizer(optimizer_name, schedule, weight_decay)

    init_state = build_frozen_state(base_params, strategy, head_pattern)

    loop =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, optimizer)
      |> Axon.Loop.metric(:accuracy)
      |> maybe_extra_metrics(extra_metrics)
      |> maybe_validate(model, validation_data)
      |> attach_init_state(init_state)
      |> maybe_checkpoint(checkpoint_path, validation_data)
      |> maybe_attach_monitor(monitor)
      |> maybe_attach_adaptive(adaptive)

    maybe_log(loop, log)
  end

  @doc """
  Build a regression training loop.

  ## Options

  - `:learning_rate` — Initial learning rate (default: 1.0e-3)
  - `:weight_decay` — AdamW weight decay (default: 1.0e-2)
  - `:epochs` — Number of training epochs for LR schedule (default: 10)
  - `:steps_per_epoch` — Steps per epoch for LR schedule (default: 1000)
  - `:patience` — Early stopping patience in epochs (default: 5)
  - `:loss` — Loss function, `:mse` or `:huber` (default: `:mse`)
  - `:precision` — Mixed precision preset (default: nil)
  - `:validation_data` — Validation data for per-epoch evaluation (default: nil)
  - `:checkpoint_path` — Directory for saving checkpoints (default: nil)
  - `:log` — Log training progress (default: true)

  ## Returns

  Configured `Axon.Loop` with:
  - Loss: MSE or Huber
  - Optimizer: AdamW with cosine decay
  - Metrics: loss + MAE
  - Early stopping on loss

  ## Example

      model = Edifice.build(:mlp, embed_dim: 64, hidden_sizes: [128, 64])
      loop = Edifice.Recipes.regress(model, loss: :huber)
      state = Axon.Loop.run(loop, train_data, %{}, epochs: 10)
  """
  @spec regress(Axon.t(), keyword()) :: Axon.Loop.t()
  def regress(model, opts \\ []) do
    lr = Keyword.get(opts, :learning_rate, 1.0e-3)
    weight_decay = Keyword.get(opts, :weight_decay, 1.0e-2)
    epochs = Keyword.get(opts, :epochs, 10)
    steps_per_epoch = Keyword.get(opts, :steps_per_epoch, 1000)
    patience = Keyword.get(opts, :patience, 5)
    loss_type = Keyword.get(opts, :loss, :mse)
    precision = Keyword.get(opts, :precision, nil)
    validation_data = Keyword.get(opts, :validation_data, nil)
    checkpoint_path = Keyword.get(opts, :checkpoint_path, nil)
    monitor = Keyword.get(opts, :monitor, nil)
    adaptive = Keyword.get(opts, :adaptive, nil)
    optimizer_name = Keyword.get(opts, :optimizer, :adamw)
    schedule_name = Keyword.get(opts, :schedule, :cosine_decay)
    reduce_lr = Keyword.get(opts, :reduce_lr_on_plateau, nil)
    gradient_opts = Keyword.get(opts, :gradient_updates, [])
    log = Keyword.get(opts, :log, true)

    model = maybe_apply_precision(model, precision)

    decay_steps = epochs * steps_per_epoch
    schedule = build_schedule(schedule_name, lr, decay_steps)
    optimizer = build_optimizer(optimizer_name, schedule, weight_decay)
    optimizer = maybe_compose_gradient_updates(optimizer, gradient_opts)

    loss =
      case loss_type do
        :huber ->
          &Axon.Losses.huber(&1, &2, reduction: :mean)

        _mse ->
          :mean_squared_error
      end

    loop =
      model
      |> Axon.Loop.trainer(loss, optimizer)
      |> Axon.Loop.metric(&Axon.Metrics.mean_absolute_error/2, "mae")
      |> maybe_validate(model, validation_data)
      |> Axon.Loop.early_stop("loss", patience: patience, mode: :min)
      |> maybe_reduce_lr(reduce_lr)
      |> maybe_checkpoint(checkpoint_path, validation_data)
      |> maybe_attach_monitor(monitor)
      |> maybe_attach_adaptive(adaptive)

    maybe_log(loop, log)
  end

  @doc """
  Return a summary of what a recipe will configure.

  ## Example

      Edifice.Recipes.describe(:classify, num_classes: 10)
      #=> %{loss: :categorical_cross_entropy, optimizer: :adamw, ...}
  """
  @spec describe(atom(), keyword()) :: map()
  def describe(recipe, opts \\ []) do
    case recipe do
      :classify ->
        %{
          loss: if(Keyword.get(opts, :label_smoothing, 0.0) > 0, do: :categorical_cross_entropy_smoothed, else: :categorical_cross_entropy),
          optimizer: :adamw,
          schedule: :cosine_decay,
          metrics: [:loss, :accuracy],
          callbacks: [:early_stop],
          weight_decay: Keyword.get(opts, :weight_decay, 1.0e-2),
          learning_rate: Keyword.get(opts, :learning_rate, 1.0e-3)
        }

      :language_model ->
        %{
          loss: :categorical_cross_entropy,
          optimizer: :adamw,
          schedule: :warmup_cosine,
          metrics: [:loss, :perplexity],
          callbacks: [:gradient_clipping],
          weight_decay: Keyword.get(opts, :weight_decay, 0.1),
          learning_rate: Keyword.get(opts, :learning_rate, 3.0e-4),
          max_grad_norm: Keyword.get(opts, :max_grad_norm, 1.0)
        }

      :contrastive ->
        %{
          loss: :infonce,
          optimizer: :adamw,
          schedule: :cosine_decay,
          metrics: [:loss],
          callbacks: [],
          temperature: Keyword.get(opts, :temperature, 0.07)
        }

      :fine_tune ->
        %{
          loss: :categorical_cross_entropy,
          optimizer: :adamw,
          schedule: :warmup_cosine,
          metrics: [:loss, :accuracy],
          callbacks: [],
          strategy: Keyword.get(opts, :strategy, :head_only),
          learning_rate: Keyword.get(opts, :learning_rate, 2.0e-5),
          warmup_ratio: Keyword.get(opts, :warmup_ratio, 0.1)
        }

      :regress ->
        loss_type = Keyword.get(opts, :loss, :mse)

        %{
          loss: if(loss_type == :huber, do: :huber, else: :mean_squared_error),
          optimizer: :adamw,
          schedule: :cosine_decay,
          metrics: [:loss, :mae],
          callbacks: [:early_stop],
          weight_decay: Keyword.get(opts, :weight_decay, 1.0e-2),
          learning_rate: Keyword.get(opts, :learning_rate, 1.0e-3)
        }
    end
  end

  # ===========================================================================
  # Loss Functions
  # ===========================================================================

  @doc """
  Compute InfoNCE (NT-Xent) contrastive loss.

  Expects embeddings where the first half and second half of the batch
  are positive pairs.

  ## Parameters

  - `embeddings` — `[2*N, dim]` tensor of embeddings
  - `temperature` — Scaling temperature (default: 0.07)
  """
  @spec infonce_loss(Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def infonce_loss(embeddings, temperature \\ 0.07) do
    batch_size = div(Nx.axis_size(embeddings, 0), 2)

    z1 = Nx.slice_along_axis(embeddings, 0, batch_size, axis: 0)
    z2 = Nx.slice_along_axis(embeddings, batch_size, batch_size, axis: 0)

    z1_norm = l2_normalize(z1)
    z2_norm = l2_normalize(z2)

    # Similarity: [N, N] — positive pairs on diagonal
    sim = Nx.dot(z1_norm, [1], z2_norm, [1])
    sim_scaled = Nx.divide(sim, temperature)

    # InfoNCE: -log(exp(sim_ii / t) / sum_j(exp(sim_ij / t)))
    log_softmax = stable_log_softmax(sim_scaled, axis: 1)
    nll = Nx.negate(Nx.take_diagonal(log_softmax))
    Nx.mean(nll)
  end

  # ===========================================================================
  # Internal Helpers
  # ===========================================================================

  defp maybe_apply_precision(model, nil), do: model
  defp maybe_apply_precision(model, preset), do: Edifice.MixedPrecision.apply(model, preset)

  defp maybe_validate(loop, _model, nil), do: loop

  defp maybe_validate(loop, model, validation_data) do
    Axon.Loop.validate(loop, model, validation_data)
  end

  defp maybe_checkpoint(loop, nil, _validation_data), do: loop

  defp maybe_checkpoint(loop, path, _validation_data) do
    Axon.Loop.checkpoint(loop, event: :epoch_completed, file_pattern: path)
  end

  defp maybe_log(loop, true) do
    Axon.Loop.log(loop, :epoch_completed, fn state ->
      {:continue, state}
    end)
  end

  defp maybe_log(loop, false), do: loop

  defp maybe_attach_monitor(loop, nil), do: loop
  defp maybe_attach_monitor(loop, true), do: Edifice.Training.Monitor.attach(loop)
  defp maybe_attach_monitor(loop, opts) when is_list(opts), do: Edifice.Training.Monitor.attach(loop, opts)

  defp maybe_attach_adaptive(loop, nil), do: loop

  defp maybe_attach_adaptive(loop, opts) when is_list(opts) do
    loop
    |> maybe_skip_spikes(Keyword.get(opts, :skip_spikes))
    |> maybe_nan_halt(Keyword.get(opts, :nan_halt))
    |> maybe_log_grad_norm(Keyword.get(opts, :log_grad_norm))
  end

  defp maybe_skip_spikes(loop, nil), do: loop
  defp maybe_skip_spikes(loop, true), do: Edifice.Training.Adaptive.skip_on_loss_spike(loop)
  defp maybe_skip_spikes(loop, opts) when is_list(opts), do: Edifice.Training.Adaptive.skip_on_loss_spike(loop, opts)

  defp maybe_nan_halt(loop, nil), do: loop
  defp maybe_nan_halt(loop, true), do: Edifice.Training.Adaptive.halt_on_persistent_nan(loop)
  defp maybe_nan_halt(loop, opts) when is_list(opts), do: Edifice.Training.Adaptive.halt_on_persistent_nan(loop, opts)

  defp maybe_log_grad_norm(loop, nil), do: loop
  defp maybe_log_grad_norm(loop, true), do: Edifice.Training.Adaptive.log_grad_norm(loop)
  defp maybe_log_grad_norm(loop, opts) when is_list(opts), do: Edifice.Training.Adaptive.log_grad_norm(loop, opts)

  # Build optimizer from name atom + options
  defp build_optimizer(name, lr_or_schedule, weight_decay) do
    opts = [learning_rate: lr_or_schedule]
    opts = if weight_decay > 0, do: Keyword.put(opts, :decay, weight_decay), else: opts

    case name do
      :adamw -> Polaris.Optimizers.adamw(opts)
      :adam -> Polaris.Optimizers.adam(opts)
      :radam -> Polaris.Optimizers.radam(opts)
      :lamb -> Polaris.Optimizers.lamb(opts)
      :rmsprop -> Polaris.Optimizers.rmsprop(learning_rate: lr_or_schedule)
      :sgd -> Polaris.Optimizers.sgd(learning_rate: lr_or_schedule)
      other -> raise ArgumentError, "unknown optimizer: #{inspect(other)}"
    end
  end

  # Build schedule from name atom + options
  defp build_schedule(name, lr, decay_steps) do
    case name do
      :cosine_decay -> Polaris.Schedules.cosine_decay(lr, decay_steps: decay_steps)
      :exponential_decay -> Polaris.Schedules.exponential_decay(lr, decay_steps: decay_steps)
      :polynomial_decay -> Polaris.Schedules.polynomial_decay(lr, decay_steps: decay_steps)
      :linear_decay -> Polaris.Schedules.linear_decay(lr, decay_steps: decay_steps)
      :constant -> Polaris.Schedules.constant(lr)
      other -> raise ArgumentError, "unknown schedule: #{inspect(other)}"
    end
  end

  # Compose gradient update transforms
  defp maybe_compose_gradient_updates(optimizer, opts) do
    max_grad_norm = Keyword.get(opts, :max_grad_norm)
    centralize = Keyword.get(opts, :centralize, false)
    add_noise = Keyword.get(opts, :add_noise, false)

    updates =
      []
      |> then(fn u -> if centralize, do: [Polaris.Updates.centralize() | u], else: u end)
      |> then(fn u -> if add_noise, do: [Polaris.Updates.add_noise() | u], else: u end)
      |> then(fn u -> if max_grad_norm, do: [Polaris.Updates.clip_by_global_norm(max_norm: max_grad_norm) | u], else: u end)

    case updates do
      [] -> optimizer
      transforms -> Enum.reduce(transforms, optimizer, &Polaris.Updates.compose(&1, &2))
    end
  end

  # Add extra metrics to loop
  defp maybe_extra_metrics(loop, nil), do: loop

  defp maybe_extra_metrics(loop, metrics) when is_list(metrics) do
    Enum.reduce(metrics, loop, fn
      :precision, loop -> Axon.Loop.metric(loop, &Axon.Metrics.precision/2, "precision")
      :recall, loop -> Axon.Loop.metric(loop, &Axon.Metrics.recall/2, "recall")
      :sensitivity, loop -> Axon.Loop.metric(loop, &Axon.Metrics.sensitivity/2, "sensitivity")
      :specificity, loop -> Axon.Loop.metric(loop, &Axon.Metrics.specificity/2, "specificity")
      :top_k_accuracy, loop -> Axon.Loop.metric(loop, &Axon.Metrics.top_k_categorical_accuracy(&1, &2, k: 5), "top_5_accuracy")
      _, loop -> loop
    end)
  end

  # Reduce LR on plateau
  defp maybe_reduce_lr(loop, nil), do: loop
  defp maybe_reduce_lr(loop, true), do: Axon.Loop.reduce_lr_on_plateau(loop, "loss", mode: :min)
  defp maybe_reduce_lr(loop, opts) when is_list(opts), do: Axon.Loop.reduce_lr_on_plateau(loop, "loss", opts)

  defp warmup_cosine_schedule(peak_lr, warmup_steps, total_steps) do
    decay_steps = max(total_steps - warmup_steps, 1)

    cosine = Polaris.Schedules.cosine_decay(peak_lr, decay_steps: decay_steps)

    if warmup_steps > 0 do
      fn step ->
        warmup_factor = Nx.min(Nx.divide(step, max(warmup_steps, 1)), 1.0)
        cosine_value = cosine.(Nx.max(Nx.subtract(step, warmup_steps), 0))
        Nx.multiply(warmup_factor, cosine_value)
      end
    else
      cosine
    end
  end

  defp l2_normalize(x) do
    norm = Nx.sqrt(Nx.sum(Nx.pow(x, 2), axes: [-1], keep_axes: true) |> Nx.add(1.0e-8))
    Nx.divide(x, norm)
  end

  defp stable_log_softmax(x, opts) do
    axis = Keyword.get(opts, :axis, -1)
    max = Nx.reduce_max(x, axes: [axis], keep_axes: true)
    shifted = Nx.subtract(x, max)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted), axes: [axis], keep_axes: true))
    Nx.subtract(shifted, log_sum_exp)
  end

  defp build_frozen_state(base_params, strategy, head_pattern) do
    state =
      case base_params do
        %Axon.ModelState{} -> base_params
        %{} -> Axon.ModelState.new(base_params)
      end

    head_filter = fn key_path ->
      Enum.any?(key_path, fn key ->
        Regex.match?(head_pattern, to_string(key))
      end)
    end

    lora_filter = fn key_path ->
      Enum.any?(key_path, fn key ->
        String.contains?(to_string(key), "lora")
      end)
    end

    case strategy do
      :head_only ->
        state
        |> Axon.ModelState.freeze()
        |> Axon.ModelState.unfreeze(head_filter)

      :lora ->
        state
        |> Axon.ModelState.freeze()
        |> Axon.ModelState.unfreeze(fn kp -> head_filter.(kp) or lora_filter.(kp) end)

      :full ->
        state
    end
  end

  defp attach_init_state(loop, init_state) do
    Axon.Loop.handle_event(loop, :started, fn state ->
      new_step_state = put_in(state.step_state[:model_state], init_state)
      {:continue, %{state | step_state: new_step_state}}
    end)
  end

  defp perplexity_metric(y_true, y_pred) do
    loss = Axon.Losses.categorical_cross_entropy(y_true, y_pred, reduction: :mean)
    Nx.exp(loss)
  end

  # mean_absolute_error removed — use Axon.Metrics.mean_absolute_error/2 directly
end
