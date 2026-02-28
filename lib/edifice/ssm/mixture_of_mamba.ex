defmodule Edifice.SSM.MixtureOfMamba do
  @moduledoc """
  Mixture-of-Mamba (MoM): Modality-Aware Sparse Mamba Blocks.

  Implements the Mixture-of-Mamba architecture from "Mixture-of-Mamba:
  Enhancing Multi-Modal State-Space Models with Modality-Aware Sparsity"
  (Liang et al., ICLR 2025).

  ## Key Innovation

  Modality-specific parameterization within Mamba blocks. Each projection
  (in_proj, x_proj, dt_proj, out_proj) has separate weights per modality
  (text, image, speech, etc.), while the Conv1D and SSM state transition A
  remain shared. A modality mask routes each token to the correct projection
  set, achieving equivalent performance at 35-65% of baseline FLOPs.

  ## Architecture

  ```
  Inputs:
    sequence: {batch, seq_len, d_model}
    modality_mask: {batch, seq_len}  (integer: 0=text, 1=image, ...)

  For each MoM block:
    [H; Z] = x * W_in^{(m)}         <- modality-specific in_proj
    U = SiLU(DepthwiseConv1d(H))    <- shared conv
    B, C = U * W_x^{(m)}            <- modality-specific SSM proj
    Delta = Softplus(U * W_dt^{(m)}) <- modality-specific dt proj
    h_t = A_bar * h_{t-1} + B_bar * u_t  <- shared SSM scan
    Y = C * h_t
    Y = Y * SiLU(Z)                 <- gating
    O = Y * W_out^{(m)}             <- modality-specific out_proj
  ```

  ## Dual Inputs

  This model takes two inputs: `"state_sequence"` and `"modality_mask"`.
  The modality mask is an integer tensor with values in 0..num_modalities-1.

  ## Usage

      model = MixtureOfMamba.build(
        embed_dim: 287,
        hidden_size: 256,
        num_modalities: 2,
        num_layers: 4
      )

      # Forward pass requires both inputs
      output = predict_fn.(params, %{
        "state_sequence" => sequence_tensor,
        "modality_mask" => mask_tensor
      })

  ## References
  - Paper: https://arxiv.org/abs/2501.16295
  - Code: https://github.com/Weixin-Liang/Mixture-of-Mamba
  """

  alias Edifice.SSM.Common

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:state_size, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:conv_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_modalities, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Mixture-of-Mamba model for multimodal sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:state_size` - SSM state dimension (default: 16)
    - `:expand_factor` - Expansion factor for inner dim (default: 2)
    - `:conv_size` - 1D convolution kernel size (default: 4)
    - `:num_layers` - Number of MoM blocks (default: 2)
    - `:num_modalities` - Number of modality-specific projection sets (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that takes `"state_sequence"` and `"modality_mask"` inputs
    and outputs the last hidden state `[batch, hidden_size]`.

  ## Examples

      iex> model = Edifice.SSM.MixtureOfMamba.build(embed_dim: 32, hidden_size: 16, state_size: 4, num_modalities: 2)
      iex> %Axon{} = model
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, Common.default_num_layers())
    dropout = Keyword.get(opts, :dropout, 0.0)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    sequence = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})
    modality_mask = Axon.input("modality_mask", shape: {nil, input_seq_dim})

    # Project to hidden_size if needed
    x =
      if embed_dim != hidden_size do
        Axon.dense(sequence, hidden_size, name: "input_projection")
      else
        sequence
      end

    # Stack MoM blocks
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block = build_mom_block(acc, modality_mask, layer_idx, opts)

        residual = Axon.add(acc, block, name: "residual_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(residual, rate: dropout, name: "dropout_#{layer_idx}")
        else
          residual
        end
      end)

    # Extract last timestep
    Axon.nx(
      output,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # Build a single MoM block with modality-specific projections
  defp build_mom_block(input, modality_mask, layer_idx, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    expand_factor = Keyword.get(opts, :expand_factor, Common.default_expand_factor())
    conv_size = Keyword.get(opts, :conv_size, Common.default_conv_size())
    num_modalities = Keyword.get(opts, :num_modalities, 2)
    name = "mom_block_#{layer_idx}"

    inner_size = hidden_size * expand_factor
    dt_rank = max(div(hidden_size, 16), 1)

    # Normalize input
    normalized = Axon.layer_norm(input, name: "#{name}_norm")

    # Modality-specific in_proj: [batch, seq, hidden] -> [batch, seq, 2*inner]
    xz =
      modality_linear(normalized, modality_mask,
        in_size: hidden_size,
        out_size: inner_size * 2,
        num_modalities: num_modalities,
        name: "#{name}_in_proj"
      )

    # Split x/z branches
    x_branch =
      Axon.nx(xz, fn t -> Nx.slice_along_axis(t, 0, inner_size, axis: 2) end,
        name: "#{name}_x_split"
      )

    z_branch =
      Axon.nx(xz, fn t -> Nx.slice_along_axis(t, inner_size, inner_size, axis: 2) end,
        name: "#{name}_z_split"
      )

    # Shared depthwise conv + SiLU
    x_conv = Common.build_depthwise_conv1d(x_branch, inner_size, conv_size, "#{name}_conv")
    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_conv_silu")

    # Modality-specific BC projection
    bc_proj =
      modality_linear(x_activated, modality_mask,
        in_size: inner_size,
        out_size: state_size * 2,
        num_modalities: num_modalities,
        name: "#{name}_bc_proj"
      )

    b_matrix =
      Axon.nx(bc_proj, fn t -> Nx.slice_along_axis(t, 0, state_size, axis: 2) end,
        name: "#{name}_B"
      )

    c_matrix =
      Axon.nx(bc_proj, fn t -> Nx.slice_along_axis(t, state_size, state_size, axis: 2) end,
        name: "#{name}_C"
      )

    # Modality-specific dt projection (low-rank bottleneck)
    dt_low =
      modality_linear(x_activated, modality_mask,
        in_size: inner_size,
        out_size: dt_rank,
        num_modalities: num_modalities,
        name: "#{name}_dt_rank"
      )

    dt_proj =
      dt_low
      |> Axon.dense(inner_size, name: "#{name}_dt_proj")
      |> Axon.activation(:softplus, name: "#{name}_dt_softplus")

    # Shared SSM scan
    ssm_out =
      Axon.layer(
        &mom_ssm_impl/5,
        [x_activated, b_matrix, c_matrix, dt_proj],
        name: "#{name}_ssm",
        state_size: state_size,
        hidden_size: inner_size,
        op_name: :mom_ssm
      )

    # Gating: Y * SiLU(Z)
    z_activated = Axon.activation(z_branch, :silu, name: "#{name}_gate_silu")
    gated = Axon.multiply(ssm_out, z_activated, name: "#{name}_gated")

    # Modality-specific out_proj
    modality_linear(gated, modality_mask,
      in_size: inner_size,
      out_size: hidden_size,
      num_modalities: num_modalities,
      name: "#{name}_out_proj"
    )
  end

  # Modality-specific linear projection using Nx.take for weight dispatch.
  # Creates K weight matrices (one per modality) as an Axon.param, then
  # uses Nx.take to select per-token weights based on modality_mask.
  defp modality_linear(input, modality_mask, opts) do
    in_size = Keyword.fetch!(opts, :in_size)
    out_size = Keyword.fetch!(opts, :out_size)
    num_modalities = Keyword.fetch!(opts, :num_modalities)
    name = Keyword.fetch!(opts, :name)

    # Stacked weight parameter: {num_modalities, in_size, out_size}
    kernel =
      Axon.param("#{name}_kernel", {num_modalities, in_size, out_size},
        initializer: :glorot_uniform
      )

    Axon.layer(
      &modality_linear_impl/4,
      [input, modality_mask, kernel],
      name: name,
      op_name: :modality_linear
    )
  end

  # Gather per-modality weights based on mask, then batched matmul.
  defp modality_linear_impl(input, mask, kernel, _opts) do
    # mask: {batch, seq_len} integer indices into [0, K)
    mask_int = Nx.as_type(mask, :s32)

    # Gather per-token weights: {batch, seq_len, in_size, out_size}
    w_per_token = Nx.take(kernel, mask_int, axis: 0)

    # Batched matmul: {batch, seq, 1, in} @ {batch, seq, in, out} -> {batch, seq, 1, out}
    x_expanded = Nx.new_axis(input, 2)
    result = Nx.dot(x_expanded, [3], [0, 1], w_per_token, [2], [0, 1])
    Nx.squeeze(result, axes: [2])
  end

  # Shared SSM scan (same as Mamba)
  defp mom_ssm_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]
    seq_len = Nx.axis_size(x, 1)

    {a_bar, bx} = Common.discretize_ssm(x, b, dt, state_size)

    h =
      if seq_len <= 32 do
        Common.sequential_scan(a_bar, bx)
      else
        Common.blelloch_scan(a_bar, bx)
      end

    Common.compute_ssm_output(h, c)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a MixtureOfMamba model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  defdelegate output_size(opts \\ []), to: Common

  @doc """
  Calculate approximate parameter count for a MixtureOfMamba model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, Common.default_hidden_size())
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    expand_factor = Keyword.get(opts, :expand_factor, Common.default_expand_factor())
    num_layers = Keyword.get(opts, :num_layers, Common.default_num_layers())
    conv_size = Keyword.get(opts, :conv_size, Common.default_conv_size())
    num_modalities = Keyword.get(opts, :num_modalities, 2)

    inner_size = hidden_size * expand_factor
    dt_rank = max(div(hidden_size, 16), 1)

    # Per layer (modality-specific projections scale by K):
    # - in_proj: K * hidden * (2 * inner)
    # - Conv kernel: conv_size * inner (shared)
    # - BC proj: K * inner * (2 * state)
    # - DT proj: K * inner * dt_rank + dt_rank * inner (shared up-proj)
    # - Out proj: K * inner * hidden
    per_layer =
      num_modalities * hidden_size * (2 * inner_size) +
        conv_size * inner_size +
        num_modalities * inner_size * (2 * state_size) +
        num_modalities * inner_size * dt_rank + dt_rank * inner_size +
        num_modalities * inner_size * hidden_size

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end
end
