defmodule Edifice.Meta.DistillationHead do
  @moduledoc """
  Distillation Head — projects student hidden states to match teacher representations.

  Provides a projection network for knowledge distillation, mapping student hidden
  states to the teacher's representation space. Also provides static utility
  functions for computing distillation losses.

  ## Architecture

  ```
  Student hidden [batch, seq_len, student_dim]
        |
  Dense(hidden) -> SiLU -> Dropout -> Dense(teacher_dim)
        |
  [batch, seq_len, teacher_dim]
  ```

  ## Loss Functions

  - `distillation_loss/3` — KL divergence with temperature scaling
  - `hidden_state_loss/2` — MSE between projected student and teacher hidden states

  ## Usage

      model = DistillationHead.build(
        embed_dim: 128,    # student_dim
        teacher_dim: 512,
        hidden_size: 256
      )

      # Compute losses
      kl = DistillationHead.distillation_loss(teacher_logits, student_logits, temperature: 4.0)
      mse = DistillationHead.hidden_state_loss(projected_student, teacher_hidden)

  ## References

  - Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
  """

  @default_hidden_size 256
  @default_num_layers 2
  @default_dropout 0.1

  @doc """
  Build a distillation projection head.

  ## Options

    - `:embed_dim` - Student hidden dimension (required, used as student_dim)
    - `:teacher_dim` - Teacher hidden dimension (required)
    - `:hidden_size` - Intermediate hidden dimension (default: 256)
    - `:num_layers` - Number of projection layers (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    An Axon model mapping `[batch, seq_len, student_dim]` to `[batch, seq_len, teacher_dim]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:teacher_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    student_dim = Keyword.fetch!(opts, :embed_dim)
    teacher_dim = Keyword.fetch!(opts, :teacher_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    input = Axon.input("state_sequence", shape: {nil, nil, student_dim})

    x =
      Enum.reduce(1..num_layers, input, fn layer_idx, acc ->
        if layer_idx < num_layers do
          acc
          |> Axon.dense(hidden_size, name: "distill_dense_#{layer_idx}")
          |> Axon.activation(:silu, name: "distill_act_#{layer_idx}")
          |> maybe_dropout(dropout, "distill_dropout_#{layer_idx}")
        else
          # Final layer projects to teacher_dim
          Axon.dense(acc, teacher_dim, name: "distill_dense_#{layer_idx}")
        end
      end)

    x
  end

  @doc """
  Compute KL divergence distillation loss with temperature scaling.

  Computes `KL(softmax(teacher/T), softmax(student/T)) * T²`.

  ## Parameters

    - `teacher_logits` - Teacher's logits `[batch, ..., vocab]`
    - `student_logits` - Student's logits `[batch, ..., vocab]`
    - `opts` - Options
      - `:temperature` - Temperature for softening (default: 4.0)

  ## Returns

    Scalar loss tensor.
  """
  @spec distillation_loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def distillation_loss(teacher_logits, student_logits, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 4.0)

    # Soft targets
    teacher_soft = stable_softmax(Nx.divide(teacher_logits, temperature))
    student_log_soft = stable_log_softmax(Nx.divide(student_logits, temperature))

    # KL(teacher || student) = sum(teacher * (log(teacher) - log(student)))
    teacher_log_soft = stable_log_softmax(Nx.divide(teacher_logits, temperature))

    kl =
      Nx.multiply(teacher_soft, Nx.subtract(teacher_log_soft, student_log_soft))
      |> Nx.sum(axes: [-1])
      |> Nx.mean()

    # Scale by T²
    Nx.multiply(kl, temperature * temperature)
  end

  @doc """
  Compute MSE loss between projected student and teacher hidden states.

  ## Parameters

    - `student_projected` - Projected student hidden states `[batch, seq, teacher_dim]`
    - `teacher_hidden` - Teacher hidden states `[batch, seq, teacher_dim]`

  ## Returns

    Scalar MSE loss tensor.
  """
  @spec hidden_state_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def hidden_state_loss(student_projected, teacher_hidden) do
    diff = Nx.subtract(student_projected, teacher_hidden)
    Nx.multiply(diff, diff) |> Nx.mean()
  end

  defp stable_softmax(logits) do
    max = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    exp = Nx.exp(Nx.subtract(logits, max))
    Nx.divide(exp, Nx.add(Nx.sum(exp, axes: [-1], keep_axes: true), 1.0e-8))
  end

  defp stable_log_softmax(logits) do
    max = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max)
    log_sum_exp = Nx.log(Nx.add(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true), 1.0e-8))
    Nx.subtract(shifted, log_sum_exp)
  end

  @doc "Get the output size of the distillation head."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :teacher_dim)
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 2,
      dropout: 0.1,
      temperature: 4.0
    ]
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)
end
