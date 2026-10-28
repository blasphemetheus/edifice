defmodule Edifice.DocCountsTest do
  @moduledoc """
  The registry (`Edifice.list_architectures/0`) is the single source of truth
  for how many architectures Edifice ships. Every document that quotes the
  number must agree with it, and every registered module must appear in the
  architecture index.
  """
  use ExUnit.Case, async: true

  @root Path.expand("../..", __DIR__)

  # These documents must quote the architecture count. Every other guide may.
  @must_quote ["README.md", "mix.exs", "docs/TESTING.md", "guides/architecture_index.md"]

  defp read!(rel), do: File.read!(Path.join(@root, rel))

  defp counted_docs do
    guides =
      Path.wildcard(Path.join(@root, "guides/*.md"))
      |> Enum.map(&Path.relative_to(&1, @root))

    Enum.uniq(@must_quote ++ guides ++ ["CONTRIBUTING.md"])
  end

  defp count_claims(text) do
    ~r/\b(\d{2,3}) (?:registered |neural network )?architectures\b/
    |> Regex.scan(text, capture: :all_but_first)
    |> Enum.map(fn [n] -> String.to_integer(n) end)
    |> Enum.uniq()
  end

  defp family_claims(text) do
    ~r/\b(\d{2}) (?:architecture )?families\b/
    |> Regex.scan(text, capture: :all_but_first)
    |> Enum.map(fn [n] -> String.to_integer(n) end)
    |> Enum.uniq()
  end

  setup_all do
    archs = Edifice.list_architectures()
    %{count: length(archs), archs: archs}
  end

  test "every document that quotes the architecture count agrees with the registry", %{
    count: count
  } do
    for rel <- counted_docs() do
      claims = count_claims(read!(rel))

      if rel in @must_quote do
        assert claims != [], "#{rel}: no architecture count found"
      end

      assert claims in [[], [count]],
             "#{rel} says #{inspect(claims)} architectures but the registry has #{count}"
    end
  end

  test "every document that quotes the family count agrees with the registry" do
    families = Edifice.list_families() |> Enum.count()
    assert read!("README.md") =~ "#{families} families"

    for rel <- counted_docs() do
      claims = family_claims(read!(rel))

      assert claims in [[], [families]],
             "#{rel} says #{inspect(claims)} families but the registry has #{families}"
    end
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

  test "the index's Verified-in-use table matches Edifice.verified_architectures/0" do
    index = read!("guides/architecture_index.md")

    [_, section | _] = String.split(index, "## Verified in use", parts: 2)
    [section | _] = String.split(section, "\n## ", parts: 2)

    listed =
      ~r/`:([a-z0-9_]+)`/
      |> Regex.scan(section, capture: :all_but_first)
      |> Enum.map(fn [n] -> String.to_atom(n) end)
      |> MapSet.new()

    verified =
      Edifice.verified_architectures()
      |> Map.values()
      |> List.flatten()
      |> MapSet.new()

    assert MapSet.difference(verified, listed) |> Enum.sort() == [],
           "verified but not in the index table"

    assert MapSet.difference(listed, verified) |> Enum.sort() == [],
           "in the index table but not in @verification"

    for {level, names} <- Edifice.verified_architectures(), name <- names do
      assert level in Edifice.verification(name)
    end
  end
end
