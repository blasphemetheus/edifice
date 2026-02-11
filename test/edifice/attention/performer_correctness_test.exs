defmodule Edifice.Attention.PerformerCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.Performer

  # ============================================================================
  # Orthogonal Feature Generation
  # ============================================================================

  describe "generate_orthogonal_features" do
    test "features are truly orthogonal (dot products near zero)" do
      head_dim = 16
      num_features = 16
      omega = Performer.generate_orthogonal_features(head_dim, num_features)

      assert Nx.shape(omega) == {num_features, head_dim}

      # omega @ omega^T should be identity (rows are orthonormal)
      gram = Nx.dot(omega, [1], omega, [1])
      eye = Nx.eye(num_features)

      # Off-diagonal elements should be near zero
      diff = Nx.subtract(gram, eye) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end

    test "features have unit norm rows" do
      omega = Performer.generate_orthogonal_features(32, 32)

      row_norms =
        Nx.sum(Nx.multiply(omega, omega), axes: [1])
        |> Nx.sqrt()
        |> Nx.to_flat_list()

      # Each row should have norm ~1.0
      for norm <- row_norms do
        assert_in_delta norm, 1.0, 1.0e-5
      end
    end

    test "handles num_features > head_dim via multiple orthogonal blocks" do
      head_dim = 8
      num_features = 20
      omega = Performer.generate_orthogonal_features(head_dim, num_features)

      assert Nx.shape(omega) == {20, 8}

      # First block (rows 0-7) should be orthogonal within themselves
      block1 = Nx.slice(omega, [0, 0], [8, 8])
      gram1 = Nx.dot(block1, [1], block1, [1])
      diff1 = Nx.subtract(gram1, Nx.eye(8)) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff1 < 1.0e-5

      # Second block (rows 8-15) should also be orthogonal within themselves
      block2 = Nx.slice(omega, [8, 0], [8, 8])
      gram2 = Nx.dot(block2, [1], block2, [1])
      diff2 = Nx.subtract(gram2, Nx.eye(8)) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff2 < 1.0e-5
    end

    test "features are deterministic (same seed gives same result)" do
      omega1 = Performer.generate_orthogonal_features(16, 16)
      omega2 = Performer.generate_orthogonal_features(16, 16)

      diff = Nx.subtract(omega1, omega2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end
end
