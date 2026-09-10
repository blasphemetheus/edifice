defmodule Edifice.FamilyDerivationTest do
  @moduledoc """
  `@registry_by_family` is the ONE source for names, modules and families;
  `list_architectures/0` and `list_families/0` are derived from it (the
  257-line hand-written family map it replaced had to be kept in sync by
  hand). `registry_integrity_test.exs` proves every entry builds; this pins
  the derivation itself.
  """
  use ExUnit.Case, async: true

  test "families partition the registry exactly (no orphan, no double-listing)" do
    names = Edifice.list_architectures()
    members = Edifice.list_families() |> Map.values() |> List.flatten()

    assert members == Enum.uniq(members), "an architecture is listed under two families"
    assert Enum.sort(members) == names
  end

  test "no family is empty" do
    assert (for {fam, []} <- Edifice.list_families(), do: fam) == []
  end
end
