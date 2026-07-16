defmodule Edifice.Interpretability.LEACETest do
  use ExUnit.Case, async: true
  @moduletag :interpretability

  alias Edifice.Interpretability.LEACE
  alias Edifice.Interpretability.Probe

  @batch 4
  @input_size 32
  @concept_dim 2

  @opts [input_size: @input_size, concept_dim: @concept_dim]

  defp template, do: %{"leace_input" => Nx.template({@batch, @input_size}, :f32)}

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = LEACE.build(@opts)
      assert %Axon{} = model
    end

    test "produces same shape as input (erasure preserves dimension)" do
      model = LEACE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"leace_input" => random_input()})
      assert Nx.shape(output) == {@batch, @input_size}
    end

    test "output contains finite values" do
      model = LEACE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"leace_input" => random_input()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "erased output differs from input (projection is non-trivial)" do
      model = LEACE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      input = random_input()
      output = predict_fn.(params, %{"leace_input" => input})

      # Output should differ from input (concept component was subtracted)
      diff = Nx.mean(Nx.abs(Nx.subtract(output, input))) |> Nx.to_number()
      # With random initialization, the projection is non-zero
      assert diff > 0.0
    end

    test "uses bias-free projection layers" do
      model = LEACE.build(@opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      data = params.data

      assert Map.has_key?(data, "leace_concept_proj")
      refute Map.has_key?(data["leace_concept_proj"], "bias")
      assert Map.has_key?(data, "leace_concept_recon")
      refute Map.has_key?(data["leace_concept_recon"], "bias")
    end
  end

  describe "build/1 with concept_dim=1" do
    test "erases a single direction" do
      model = LEACE.build(input_size: @input_size, concept_dim: 1)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"leace_input" => random_input()})
      assert Nx.shape(output) == {@batch, @input_size}
    end
  end

  describe "output_size/1" do
    test "returns input_size" do
      assert LEACE.output_size(@opts) == @input_size
    end
  end

  describe "Edifice.build/2" do
    test "builds leace via registry" do
      model = Edifice.build(:leace, @opts)
      assert %Axon{} = model
    end
  end

  # ==========================================================================
  # Closed-form fit — FUNCTIONAL tests (the guarantees, not just shapes)
  # ==========================================================================

  describe "fit/2 + erase/2 (closed form)" do
    # Plant a 2-dim concept into representations, then verify the LEACE
    # guarantee directly: after erasure the cross-covariance of X with Z is
    # ~zero (no linear probe can recover Z), while independent signal
    # survives.
    defp planted_data do
      key = Nx.Random.key(7)
      {z, key} = Nx.Random.normal(key, shape: {512, 2})
      {noise, key} = Nx.Random.normal(key, shape: {512, 16})
      {dirs, _key} = Nx.Random.normal(key, shape: {2, 16})
      x = Nx.add(noise, Nx.multiply(Nx.dot(z, dirs), 3.0))
      {x, z}
    end

    defp max_abs_crosscov(x, z) do
      n = Nx.axis_size(x, 0)
      xc = Nx.subtract(x, Nx.mean(x, axes: [0]))
      zc = Nx.subtract(z, Nx.mean(z, axes: [0])) |> Nx.as_type(Nx.type(x))

      Nx.dot(Nx.transpose(xc), zc)
      |> Nx.divide(n)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()
    end

    test "zeroes cross-covariance with the concept (the LEACE guarantee)" do
      {x, z} = planted_data()
      x32 = Nx.as_type(x, :f32)

      before_cc = max_abs_crosscov(x32, z)
      eraser = LEACE.fit(x, z)
      after_cc = max_abs_crosscov(LEACE.erase(eraser, x32), z)

      assert before_cc > 0.5
      assert after_cc < 1.0e-3
    end

    test "recovers the planted concept rank" do
      {x, z} = planted_data()
      assert LEACE.fit(x, z).rank == 2
    end

    test "preserves signal independent of the concept" do
      # Add an extra column carrying an independent signal; it must
      # survive erasure nearly unchanged.
      key = Nx.Random.key(11)
      {s, _} = Nx.Random.normal(key, shape: {512, 1})
      {x, z} = planted_data()
      x = Nx.concatenate([Nx.as_type(x, :f32), Nx.as_type(s, :f32)], axis: 1)

      eraser = LEACE.fit(x, z)
      xe = LEACE.erase(eraser, x)

      last = fn t -> Nx.slice_along_axis(t, 16, 1, axis: 1) end
      change = Nx.mean(Nx.abs(Nx.subtract(last.(xe), last.(x)))) |> Nx.to_number()
      scale = Nx.mean(Nx.abs(last.(x))) |> Nx.to_number()

      # Bar accounts for finite-sample effects: at n=512, an "independent"
      # random column carries ~1/sqrt(n) ≈ 4-5% spurious sample correlation
      # with Z, which LEACE correctly removes — observed change ~9%.
      assert change < 0.15 * scale
    end

    test "apply_layer/2 matches erase/2" do
      {x, z} = planted_data()
      x32 = Nx.as_type(x, :f32)
      eraser = LEACE.fit(x, z)

      input = Axon.input("t", shape: {nil, 16})
      model = LEACE.apply_layer(input, eraser)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(%{"t" => Nx.template({512, 16}, :f32)}, Axon.ModelState.empty())

      from_layer = predict_fn.(params, %{"t" => x32})
      from_fn = LEACE.erase(eraser, x32)

      diff = Nx.reduce_max(Nx.abs(Nx.subtract(from_layer, from_fn))) |> Nx.to_number()
      assert diff < 1.0e-5
    end
  end

  describe "fit/2 numerical + probe pins" do
    defp max_abs_diff(a, b) do
      Nx.reduce_max(Nx.abs(Nx.subtract(a, b))) |> Nx.to_number()
    end

    test "matches the hand-computed eraser on an exactly-white 2-d case" do
      # Sample covariance of x is exactly I (4 points at the corners of a
      # square) and z IS the first coordinate, so Σ_xz = e₁. Whitening is a
      # scalar multiple of the identity (the sqrt(1+ridge) factors cancel in
      # A = W⁺ P W), so the closed form collapses exactly to
      #   μ = 0,  A = e₁e₁ᵀ = [[1, 0], [0, 0]]
      # i.e. erasure zeroes coordinate 1 and leaves coordinate 2 untouched.
      x = Nx.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
      z = Nx.tensor([[1.0], [1.0], [-1.0], [-1.0]])

      eraser = LEACE.fit(x, z)

      assert eraser.rank == 1
      assert max_abs_diff(eraser.mu, Nx.tensor([0.0, 0.0])) < 1.0e-6
      assert max_abs_diff(eraser.a, Nx.tensor([[1.0, 0.0], [0.0, 0.0]])) < 1.0e-5

      erased = LEACE.erase(eraser, Nx.tensor([[2.0, 3.0]]))
      assert max_abs_diff(erased, Nx.tensor([[0.0, 3.0]])) < 1.0e-4
    end

    test "linear probe accuracy drops to ~chance after erasure" do
      {x, z} = planted_data()
      x = Nx.as_type(x, :f32)

      y =
        z
        |> Nx.slice_along_axis(0, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
        |> Nx.greater(0.0)
        |> Nx.as_type(:s64)

      half = 256
      split = fn t ->
        {Nx.slice_along_axis(t, 0, half, axis: 0), Nx.slice_along_axis(t, half, half, axis: 0)}
      end

      {x_tr, x_ev} = split.(x)
      {y_tr, y_ev} = split.(y)

      before_acc = Probe.fit_eval(x_tr, y_tr, x_ev, y_ev, 2).balanced_accuracy
      assert before_acc > 0.9

      eraser = LEACE.fit(x, z)
      {xe_tr, xe_ev} = split.(LEACE.erase(eraser, x))
      after_acc = Probe.fit_eval(xe_tr, y_tr, xe_ev, y_ev, 2).balanced_accuracy

      # ~chance band (0.5 ± sampling noise on 256 eval rows)
      assert after_acc < 0.6
      assert after_acc > 0.4
    end
  end
end
