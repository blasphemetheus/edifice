defmodule Edifice.Generative.TRELLISCodecTest do
  use ExUnit.Case, async: true
  import Nx.Testing
  alias Edifice.Generative.TRELLIS

  test "scatter preserves coordinates, channels, batches, and empty cells" do
    slat = %{
      features: Nx.tensor([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]]),
      positions: Nx.tensor([[[1, 0, 1], [0, 1, 0]], [[1, 0, 1], [1, 1, 1]]]),
      mask: Nx.tensor([[1, 0], [1, 1]])
    }

    actual = TRELLIS.decode_from_slat(slat, resolution: 2)

    expected =
      Nx.broadcast(0.0, {2, 2, 2, 2, 2})
      |> Nx.put_slice([0, 1, 0, 1, 0], Nx.tensor([[[[[2.0, 3.0]]]]]))
      |> Nx.put_slice([1, 1, 0, 1, 0], Nx.tensor([[[[[6.0, 7.0]]]]]))
      |> Nx.put_slice([1, 1, 1, 1, 0], Nx.tensor([[[[[8.0, 9.0]]]]]))

    assert_equal(actual, expected)
  end

  test "duplicates add; invalid coordinates and masked NaNs do not contaminate the grid" do
    slat = %{
      features: Nx.tensor([[[2.0], [3.0], [:nan], [9.0], [9.0], [9.0]]]),
      positions:
        Nx.tensor([
          [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [:nan, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 0.0, 0.0]
          ]
        ]),
      mask: Nx.tensor([[1, 1, 0, 1, 1, 1]])
    }

    dense = TRELLIS.decode_from_slat(slat, resolution: 2)
    assert Nx.to_number(Nx.sum(dense)) == 5.0
    assert Nx.to_number(dense[0][1][1][1][0]) == 5.0
    zero = TRELLIS.decode_from_slat(%{slat | mask: Nx.broadcast(0, {1, 6})}, resolution: 2)
    assert_equal(zero, Nx.broadcast(0.0, {1, 2, 2, 2, 1}))
  end

  test "sparse round trip includes occupied cells beyond the first max_voxels grid indices" do
    dense =
      Nx.broadcast(0.0, {2, 3, 3, 3, 2})
      |> Nx.put_slice([0, 2, 2, 2, 0], Nx.tensor([[[[[2.0, -1.0]]]]]))
      |> Nx.put_slice([1, 1, 2, 1, 0], Nx.tensor([[[[[3.0, 4.0]]]]]))

    slat = TRELLIS.encode_to_slat(dense, max_voxels: 2)
    assert Nx.to_list(slat.mask) == [[1.0, 0.0], [1.0, 0.0]]
    assert_equal(TRELLIS.decode_from_slat(slat, resolution: 3), dense)
  end

  test "occupancy-only encoding retains scalar densities and truncates deterministically" do
    dense = Nx.tensor([[[[0.0, 0.0], [0.0, 0.6]], [[0.0, 0.0], [0.7, 0.8]]]])
    slat = TRELLIS.encode_to_slat(dense, max_voxels: 2)
    assert Nx.to_list(slat.positions) == [[[0, 1, 1], [1, 1, 0]]]
    assert_all_close(slat.features, Nx.tensor([[[0.6], [0.7]]]))
    roundtrip = TRELLIS.decode_from_slat(TRELLIS.encode_to_slat(dense), resolution: 2)
    assert_all_close(roundtrip, Nx.new_axis(dense, -1))
  end

  test "scatter differentiates only through active feature values" do
    positions = Nx.tensor([[[0, 0, 0], [1, 1, 1]]])
    mask = Nx.tensor([[1, 0]])

    grad =
      Nx.Defn.grad(Nx.tensor([[[2.0], [3.0]]]), fn features ->
        TRELLIS.decode_from_slat(%{features: features, positions: positions, mask: mask},
          resolution: 2
        )
        |> Nx.sum()
      end)

    assert_equal(grad, Nx.tensor([[[1.0], [0.0]]]))
  end

  @tag :exla_only
  test "dense scatter runs inside EXLA with dynamic coordinates" do
    decode =
      Nx.Defn.jit(
        fn features, positions, mask ->
          TRELLIS.decode_from_slat(%{features: features, positions: positions, mask: mask},
            resolution: 2
          )
        end,
        compiler: EXLA,
        client: :host
      )

    features = Nx.tensor([[[2.0], [3.0]]])
    positions = Nx.tensor([[[1, 1, 1], [1, 1, 1]]])
    mask = Nx.tensor([[1, 1]])

    assert_all_close(
      decode.(features, positions, mask) |> Nx.backend_transfer(Nx.BinaryBackend),
      TRELLIS.decode_from_slat(%{features: features, positions: positions, mask: mask},
        resolution: 2
      )
    )
  end
end
