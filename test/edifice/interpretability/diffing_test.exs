defmodule Edifice.Interpretability.DiffingTest do
  use ExUnit.Case, async: true

  alias Edifice.Interpretability.{Diffing, SAETrainer}

  @dict 8
  @d 4
  @n 64

  defp acts(seed, scale \\ 1.0) do
    key = Nx.Random.key(seed)
    {x, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {@n, @d}, type: :f32)
    Nx.multiply(x, scale)
  end

  defp quick_fit(x, opts \\ []) do
    SAETrainer.fit(
      :batch_top_k_sae,
      x,
      [dict_size: @dict, batch_k: 16, steps: 25, lr: 5.0e-2] ++ opts
    )
  end

  describe "warm-start (init_params)" do
    test "finetune resumes from the fitted dictionary, not fresh init" do
      a = quick_fit(acts(1))
      b = SAETrainer.finetune(a, acts(1), steps: 1)

      # One step on the same data barely moves unit-normed decoder rows
      shift = Diffing.decoder_shift(a.params, b.params)
      assert length(shift) == @dict
      assert Enum.all?(shift, &(&1.cosine > 0.98))
    end

    test "init_params with mismatched build raises" do
      a = quick_fit(acts(1))

      assert_raise ArgumentError, ~r/do not match/, fn ->
        SAETrainer.fit(:batch_top_k_sae, acts(1),
          dict_size: @dict * 2,
          batch_k: 16,
          steps: 1,
          init_params: a.params
        )
      end
    end
  end

  describe "batch_size (minibatch fit)" do
    test "minibatch fit runs and returns the same result shape" do
      r = quick_fit(acts(2), batch_size: 16)

      assert length(r.history) == 25
      assert %Axon.ModelState{} = r.params
    end

    test "batch_size >= n behaves as full-batch (deterministic vs full)" do
      r1 = quick_fit(acts(3))
      r2 = quick_fit(acts(3), batch_size: @n * 2)

      assert r1.history == r2.history
    end
  end

  describe "decoder_shift/2" do
    test "identical params -> all cosines 1.0, sorted list of dict_size" do
      a = quick_fit(acts(4))
      shift = Diffing.decoder_shift(a.params, a.params)

      assert length(shift) == @dict
      assert Enum.all?(shift, &(&1.cosine >= 0.9999))
      assert shift == Enum.sort_by(shift, & &1.cosine)
    end

    test "finetuning on DIFFERENT data moves features; moved_features filters" do
      a = quick_fit(acts(5))
      b = SAETrainer.finetune(a, acts(99, 3.0), steps: 25)

      shift = Diffing.decoder_shift(a.params, b.params)
      moved = Diffing.moved_features(shift, 0.999)

      # Distribution shift must move at least one direction measurably
      assert moved != []
      assert Enum.all?(shift, &(&1.cosine <= 1.0001 and &1.cosine >= -1.0001))
    end
  end

  describe "firing_shift/2" do
    test "returns a {dict_size} delta tensor in [-1, 1]" do
      a = quick_fit(acts(6))
      b = SAETrainer.finetune(a, acts(7), steps: 10)

      delta = Diffing.firing_shift({a, acts(6)}, {b, acts(7)})

      assert Nx.shape(delta) == {@dict}
      assert delta |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() <= 1.0
    end
  end
end
