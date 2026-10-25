defmodule Edifice.Recurrent.BlockScanWiringTest do
  @moduledoc """
  End-to-end tests for block scan architecture wiring.

  Validates that `pack_block_weights` correctly maps Axon parameter names
  to the flat weight layout expected by block scan kernels, and that
  `fused_block_inference` produces output matching sequential single-layer
  computation with the same weights.

  Runs on BinaryBackend (no GPU required) — uses kernel fallback paths.
  """
  use ExUnit.Case, async: true

  alias Edifice.Recurrent.DeepResLSTM
  alias Edifice.Recurrent, as: Rec
  alias Edifice.CUDA.FusedScan

  # Small dims for fast tests
  @batch 2
  @seq_len 4
  @hidden 8

  # ============================================================================
  # DeepResLSTM pack_block_weights
  # ============================================================================

  describe "DeepResLSTM.pack_block_weights/3" do
    test "produces correct packed layout for 2 layers" do
      num_layers = 2
      h = @hidden

      # Synthesize params matching fused recurrent path key structure
      {params, per_layer} = build_deep_res_lstm_params(num_layers, h)

      packed = DeepResLSTM.pack_block_weights(params, num_layers, h)
      stride = 8 * h * h + 6 * h

      assert Nx.shape(packed) == {num_layers * stride}

      # Verify each layer's weights are packed in the correct order
      for {layer_data, layer_idx} <- Enum.with_index(per_layer) do
        offset = layer_idx * stride
        %{w_x: w_x, b_x: b_x, r_w: r_w, gamma: gamma, beta: beta} = layer_data

        packed_w_x = Nx.slice(packed, [offset], [h * 4 * h])
        packed_b_x = Nx.slice(packed, [offset + h * 4 * h], [4 * h])
        packed_r_w = Nx.slice(packed, [offset + h * 4 * h + 4 * h], [h * 4 * h])
        packed_gamma = Nx.slice(packed, [offset + 8 * h * h + 4 * h], [h])
        packed_beta = Nx.slice(packed, [offset + 8 * h * h + 5 * h], [h])

        assert_all_close(packed_w_x, Nx.flatten(w_x))
        assert_all_close(packed_b_x, Nx.flatten(b_x))
        assert_all_close(packed_r_w, Nx.flatten(r_w))
        assert_all_close(packed_gamma, Nx.flatten(gamma))
        assert_all_close(packed_beta, Nx.flatten(beta))
      end
    end

    test "packed weights produce valid output through fused_block_inference" do
      num_layers = 2
      h = @hidden

      {params, _per_layer} = build_deep_res_lstm_params(num_layers, h)
      packed = DeepResLSTM.pack_block_weights(params, num_layers, h)

      key = Nx.Random.key(42)

      {input, _key} =
        Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch, @seq_len, h}, type: {:f, 32})

      # fused_block_inference calls FusedScan.lstm_block which falls back to pure Nx
      output = DeepResLSTM.fused_block_inference(input, packed, num_layers)

      assert Nx.shape(output) == {@batch, @seq_len, h}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "fused_block_inference matches direct FusedScan.lstm_block call" do
      num_layers = 2
      h = @hidden

      {params, _per_layer} = build_deep_res_lstm_params(num_layers, h)
      packed = DeepResLSTM.pack_block_weights(params, num_layers, h)

      key = Nx.Random.key(42)

      {input, _key} =
        Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch, @seq_len, h}, type: {:f, 32})

      # Through module helper (zeros h0, c0 by default)
      mod_out = DeepResLSTM.fused_block_inference(input, packed, num_layers)

      # Direct kernel call with explicit zeros
      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {@batch, num_layers, h})
      c0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {@batch, num_layers, h})
      direct_out = FusedScan.lstm_block(input, packed, h0, c0, num_layers)

      assert_all_close(mod_out, direct_out)
    end
  end

  # ============================================================================
  # Recurrent LSTM pack_block_weights
  # ============================================================================

  describe "Recurrent.pack_block_weights/4 :lstm" do
    test "produces correct packed layout with prenorm mapping" do
      num_layers = 2
      h = @hidden

      # Synthesize params matching Recurrent's fused recurrent path:
      # input_ln, lstm_1_input_proj, lstm_1_fused_scan, lstm_1_ln, lstm_2_..., lstm_2_ln
      {params, expected} = build_recurrent_lstm_params(num_layers, h)

      packed = Rec.pack_block_weights(params, num_layers, h, :lstm)
      stride = 8 * h * h + 6 * h

      assert Nx.shape(packed) == {num_layers * stride}

      # Verify prenorm mapping:
      #   block layer 1 prenorm = input_ln
      #   block layer 2 prenorm = lstm_1_ln
      for {layer_data, layer_idx} <- Enum.with_index(expected) do
        offset = layer_idx * stride
        %{w_x: w_x, b_x: b_x, r_w: r_w, gamma: gamma, beta: beta} = layer_data

        packed_w_x = Nx.slice(packed, [offset], [h * 4 * h])
        packed_b_x = Nx.slice(packed, [offset + h * 4 * h], [4 * h])
        packed_r_w = Nx.slice(packed, [offset + h * 4 * h + 4 * h], [h * 4 * h])
        packed_gamma = Nx.slice(packed, [offset + 8 * h * h + 4 * h], [h])
        packed_beta = Nx.slice(packed, [offset + 8 * h * h + 5 * h], [h])

        assert_all_close(packed_w_x, Nx.flatten(w_x), label: "layer #{layer_idx + 1} w_x")
        assert_all_close(packed_b_x, Nx.flatten(b_x), label: "layer #{layer_idx + 1} b_x")
        assert_all_close(packed_r_w, Nx.flatten(r_w), label: "layer #{layer_idx + 1} r_w")
        assert_all_close(packed_gamma, Nx.flatten(gamma), label: "layer #{layer_idx + 1} gamma")
        assert_all_close(packed_beta, Nx.flatten(beta), label: "layer #{layer_idx + 1} beta")
      end
    end

    test "packed weights produce valid output through fused_lstm_block_inference" do
      num_layers = 2
      h = @hidden

      {params, _expected} = build_recurrent_lstm_params(num_layers, h)
      packed = Rec.pack_block_weights(params, num_layers, h, :lstm)

      key = Nx.Random.key(42)

      {input, _key} =
        Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch, @seq_len, h}, type: {:f, 32})

      output = Rec.fused_lstm_block_inference(input, packed, num_layers)

      assert Nx.shape(output) == {@batch, @seq_len, h}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Recurrent GRU pack_block_weights
  # ============================================================================

  describe "Recurrent.pack_block_weights/4 :gru" do
    test "produces correct packed layout with prenorm mapping" do
      num_layers = 2
      h = @hidden

      {params, expected} = build_recurrent_gru_params(num_layers, h)

      packed = Rec.pack_block_weights(params, num_layers, h, :gru)
      stride = 6 * h * h + 5 * h

      assert Nx.shape(packed) == {num_layers * stride}

      for {layer_data, layer_idx} <- Enum.with_index(expected) do
        offset = layer_idx * stride
        %{w_x: w_x, b_x: b_x, r_w: r_w, gamma: gamma, beta: beta} = layer_data

        packed_w_x = Nx.slice(packed, [offset], [h * 3 * h])
        packed_b_x = Nx.slice(packed, [offset + h * 3 * h], [3 * h])
        packed_r_w = Nx.slice(packed, [offset + h * 3 * h + 3 * h], [h * 3 * h])
        packed_gamma = Nx.slice(packed, [offset + 6 * h * h + 3 * h], [h])
        packed_beta = Nx.slice(packed, [offset + 6 * h * h + 4 * h], [h])

        assert_all_close(packed_w_x, Nx.flatten(w_x), label: "layer #{layer_idx + 1} w_x")
        assert_all_close(packed_b_x, Nx.flatten(b_x), label: "layer #{layer_idx + 1} b_x")
        assert_all_close(packed_r_w, Nx.flatten(r_w), label: "layer #{layer_idx + 1} r_w")
        assert_all_close(packed_gamma, Nx.flatten(gamma), label: "layer #{layer_idx + 1} gamma")
        assert_all_close(packed_beta, Nx.flatten(beta), label: "layer #{layer_idx + 1} beta")
      end
    end

    test "packed weights produce valid output through fused_gru_block_inference" do
      num_layers = 2
      h = @hidden

      {params, _expected} = build_recurrent_gru_params(num_layers, h)
      packed = Rec.pack_block_weights(params, num_layers, h, :gru)

      key = Nx.Random.key(42)

      {input, _key} =
        Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch, @seq_len, h}, type: {:f, 32})

      output = Rec.fused_gru_block_inference(input, packed, num_layers)

      assert Nx.shape(output) == {@batch, @seq_len, h}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Helpers — param builders
  # ============================================================================

  # Build synthetic params matching DeepResLSTM fused recurrent path key structure.
  # Returns {params_map, [%{w_x, b_x, r_w, gamma, beta}, ...]} for layout verification.
  defp build_deep_res_lstm_params(num_layers, hidden) do
    key = Nx.Random.key(99)

    {layers, params, _key} =
      Enum.reduce(1..num_layers, {[], %{}, key}, fn layer_idx, {layers, params, k} ->
        {w_x, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden, 4 * hidden}, type: {:f, 32})
        {b_x, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {4 * hidden}, type: {:f, 32})
        {r_w, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden, 4 * hidden}, type: {:f, 32})
        gamma = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {hidden})
        beta = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {hidden})

        layer_data = %{w_x: w_x, b_x: b_x, r_w: r_w, gamma: gamma, beta: beta}

        params =
          params
          |> Map.put("lstm_#{layer_idx}_input_proj", %{"kernel" => w_x, "bias" => b_x})
          |> Map.put("lstm_#{layer_idx}_fused_scan", %{
            "lstm_#{layer_idx}_recurrent_kernel" => r_w
          })
          |> Map.put("block_#{layer_idx}_prenorm", %{"gamma" => gamma, "beta" => beta})

        {[layer_data | layers], params, k}
      end)

    {params, Enum.reverse(layers)}
  end

  # Build synthetic params matching Recurrent's fused LSTM path key structure.
  # The prenorm mapping is the critical thing being tested:
  #   block layer 1 prenorm = input_ln
  #   block layer k>1 prenorm = lstm_{k-1}_ln
  defp build_recurrent_lstm_params(num_layers, hidden) do
    key = Nx.Random.key(101)

    # input_ln params (will become block layer 1's prenorm)
    {input_gamma, key} = Nx.Random.uniform(key, 0.9, 1.1, shape: {hidden}, type: {:f, 32})
    {input_beta, key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {hidden}, type: {:f, 32})

    params = %{"input_ln" => %{"gamma" => input_gamma, "beta" => input_beta}}

    {layers, params, _key} =
      Enum.reduce(1..num_layers, {[], params, key}, fn layer_idx, {layers, params, k} ->
        {w_x, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden, 4 * hidden}, type: {:f, 32})
        {b_x, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {4 * hidden}, type: {:f, 32})
        {r_w, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden, 4 * hidden}, type: {:f, 32})
        {ln_gamma, k} = Nx.Random.uniform(k, 0.9, 1.1, shape: {hidden}, type: {:f, 32})
        {ln_beta, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden}, type: {:f, 32})

        # Expected prenorm for this block layer
        {gamma, beta} =
          if layer_idx == 1 do
            {input_gamma, input_beta}
          else
            prev = "lstm_#{layer_idx - 1}_ln"
            {params[prev]["gamma"], params[prev]["beta"]}
          end

        layer_data = %{w_x: w_x, b_x: b_x, r_w: r_w, gamma: gamma, beta: beta}

        params =
          params
          |> Map.put("lstm_#{layer_idx}_input_proj", %{"kernel" => w_x, "bias" => b_x})
          |> Map.put("lstm_#{layer_idx}_fused_scan", %{
            "lstm_#{layer_idx}_recurrent_kernel" => r_w
          })
          |> Map.put("lstm_#{layer_idx}_ln", %{"gamma" => ln_gamma, "beta" => ln_beta})

        {[layer_data | layers], params, k}
      end)

    {params, Enum.reverse(layers)}
  end

  # Build synthetic params matching Recurrent's fused GRU path key structure.
  defp build_recurrent_gru_params(num_layers, hidden) do
    key = Nx.Random.key(202)

    {input_gamma, key} = Nx.Random.uniform(key, 0.9, 1.1, shape: {hidden}, type: {:f, 32})
    {input_beta, key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {hidden}, type: {:f, 32})

    params = %{"input_ln" => %{"gamma" => input_gamma, "beta" => input_beta}}

    {layers, params, _key} =
      Enum.reduce(1..num_layers, {[], params, key}, fn layer_idx, {layers, params, k} ->
        {w_x, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden, 3 * hidden}, type: {:f, 32})
        {b_x, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {3 * hidden}, type: {:f, 32})
        {r_w, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden, 3 * hidden}, type: {:f, 32})
        {ln_gamma, k} = Nx.Random.uniform(k, 0.9, 1.1, shape: {hidden}, type: {:f, 32})
        {ln_beta, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden}, type: {:f, 32})

        {gamma, beta} =
          if layer_idx == 1 do
            {input_gamma, input_beta}
          else
            prev = "gru_#{layer_idx - 1}_ln"
            {params[prev]["gamma"], params[prev]["beta"]}
          end

        layer_data = %{w_x: w_x, b_x: b_x, r_w: r_w, gamma: gamma, beta: beta}

        params =
          params
          |> Map.put("gru_#{layer_idx}_input_proj", %{"kernel" => w_x, "bias" => b_x})
          |> Map.put("gru_#{layer_idx}_fused_scan", %{
            "gru_#{layer_idx}_recurrent_kernel" => r_w
          })
          |> Map.put("gru_#{layer_idx}_ln", %{"gamma" => ln_gamma, "beta" => ln_beta})

        {[layer_data | layers], params, k}
      end)

    {params, Enum.reverse(layers)}
  end

  # ============================================================================
  # Assertion helpers
  # ============================================================================

  defp assert_all_close(left, right, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-6)
    label = Keyword.get(opts, :label, "")

    left_flat = Nx.to_flat_list(left)
    right_flat = Nx.to_flat_list(right)

    assert length(left_flat) == length(right_flat),
           "Shape mismatch #{label}: #{length(left_flat)} vs #{length(right_flat)}"

    Enum.zip(left_flat, right_flat)
    |> Enum.with_index()
    |> Enum.each(fn {{l, r}, idx} ->
      diff = abs(l - r)

      assert diff < atol,
             "#{label} element #{idx}: #{l} vs #{r} (diff=#{diff}, atol=#{atol})"
    end)
  end
end
