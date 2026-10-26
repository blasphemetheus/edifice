defmodule Edifice.DocCountsTest do
  @moduledoc """
  The registry (`Edifice.list_architectures/0`) is the single source of truth
  for how many architectures Edifice ships. Every document that quotes the
  number must agree with it, and every registered module must appear in the
  architecture index.
  """
  use ExUnit.Case, async: true

  @root Path.expand("../..", __DIR__)

  defp read!(rel), do: File.read!(Path.join(@root, rel))

  defp count_claims(text) do
    ~r/\b(\d{2,3}) (?:neural network )?architectures\b/
    |> Regex.scan(text, capture: :all_but_first)
    |> Enum.map(fn [n] -> String.to_integer(n) end)
    |> Enum.uniq()
  end

  setup_all do
    archs = Edifice.list_architectures()
    %{count: length(archs), archs: archs}
  end

  test "README, mix.exs, TESTING.md and the index quote the registry size", %{count: count} do
    for rel <- ["README.md", "mix.exs", "docs/TESTING.md", "guides/architecture_index.md"] do
      claims = count_claims(read!(rel))

      assert claims != [], "#{rel}: no architecture count found"

      assert claims == [count],
             "#{rel} says #{inspect(claims)} architectures but the registry has #{count}"
    end
  end

  test "README quotes the registry family count", %{archs: _} do
    families = Edifice.list_families() |> Enum.count()
    assert read!("README.md") =~ "#{families} families"
  end

  test "every registered module appears in the architecture index", %{archs: archs} do
    index = read!("guides/architecture_index.md")

    missing =
      archs
      |> Enum.map(&Edifice.module_for/1)
      |> Enum.uniq()
      |> Enum.reject(&String.contains?(index, "`#{inspect(&1)}`"))

    assert missing == [], "missing from guides/architecture_index.md: #{inspect(missing)}"
  end
end
