defmodule Edifice.SpecTest do
  use ExUnit.Case, async: true

  alias Edifice.Spec

  describe "new/3" do
    test "fills versions and created_at by default" do
      spec = Spec.new(:mlp, input_size: 8, hidden_sizes: [16])

      assert spec.arch == :mlp
      assert spec.build_opts[:input_size] == 8
      assert is_binary(spec.edifice_version)
      assert is_binary(spec.nx_version)
      # nx is a loaded application in the test env, so the real version
      # (not the "unknown" fallback) must be reported
      assert spec.nx_version =~ ~r/^\d+\.\d+/
      assert %DateTime{} = spec.created_at
    end

    test "honors an explicit created_at (no hidden clock dependency)" do
      created_at = ~U[2026-07-15 12:00:00Z]
      spec = Spec.new(:mlp, [input_size: 8], created_at: created_at)

      assert spec.created_at == created_at
    end

    test "external: true admits a composite arch owned by a downstream library" do
      spec = Edifice.Spec.new(:exphil_policy, [embed_size: 288, backbone: :gru], external: true)

      assert spec.arch == :exphil_policy
      assert spec.external == true
      assert spec.build_opts[:backbone] == :gru
      # still round-trips through metadata serialization
      assert {:ok, %Edifice.Spec{arch: :exphil_policy, external: true}} =
               spec |> Edifice.Spec.to_map() |> Edifice.Spec.from_map()
    end

    test "external: true still rejects non-serializable build opts" do
      assert_raise ArgumentError, fn ->
        Edifice.Spec.new(:whatever, [fun: &Enum.map/2], external: true)
      end
    end

    test "raises on unknown architecture" do
      assert_raise ArgumentError, ~r/Unknown architecture/, fn ->
        Spec.new(:not_a_real_arch, [])
      end
    end

    test "raises on non-serializable build opts, naming the offending keys" do
      assert_raise ArgumentError, ~r/:activation_fn/, fn ->
        Spec.new(:mlp, input_size: 8, activation_fn: fn x -> x end)
      end

      assert_raise ArgumentError, ~r/:init_tensor/, fn ->
        Spec.new(:mlp, input_size: 8, init_tensor: Nx.tensor([1.0]))
      end
    end

    test "accepts plain-term opts including tuples and nested lists" do
      spec = Spec.new(:mlp, input_size: 8, input_shape: {nil, 32, 32, 3}, hidden_sizes: [16, 8])
      assert spec.build_opts[:input_shape] == {nil, 32, 32, 3}
    end
  end

  describe "to_map/1 and from_map/1" do
    test "round-trips a spec exactly" do
      created_at = ~U[2026-07-15 12:34:56.789Z]

      spec =
        Spec.new(:mamba, [embed_dim: 8, state_size: 4, num_layers: 1],
          created_at: created_at
        )

      map = Spec.to_map(spec)

      # created_at is stored as an ISO 8601 string (plain term, safe for
      # binary_to_term(bin, [:safe]))
      assert is_binary(map.created_at)

      assert {:ok, restored} = Spec.from_map(map)
      assert restored.arch == spec.arch
      assert restored.build_opts == spec.build_opts
      assert restored.edifice_version == spec.edifice_version
      assert restored.nx_version == spec.nx_version
      assert DateTime.compare(restored.created_at, spec.created_at) == :eq
    end

    test "survives a term_to_binary/binary_to_term(:safe) round-trip" do
      spec = Spec.new(:mamba, embed_dim: 8, state_size: 4)

      binary = :erlang.term_to_binary(Spec.to_map(spec))
      restored_map = :erlang.binary_to_term(binary, [:safe])

      assert {:ok, restored} = Spec.from_map(restored_map)
      assert restored.arch == :mamba
      assert restored.build_opts[:state_size] == 4
    end

    test "from_map tolerates missing version fields" do
      assert {:ok, spec} = Spec.from_map(%{arch: :mlp, build_opts: [input_size: 4]})
      assert spec.edifice_version == "unknown"
      assert spec.nx_version == "unknown"
    end

    test "from_map rejects garbage without raising" do
      assert {:error, _} = Spec.from_map(%{"bogus" => 1})
      assert {:error, _} = Spec.from_map(42)
      assert {:error, _} = Spec.from_map(nil)
      assert {:error, _} = Spec.from_map(%{arch: :mlp, build_opts: "not a keyword list"})
      assert {:error, _} = Spec.from_map(%{arch: :not_a_real_arch, build_opts: []})

      assert {:error, reason} =
               Spec.from_map(%{arch: :mlp, build_opts: [], created_at: "not a date"})

      assert reason =~ "created_at"
    end
  end

  describe "Edifice.build_with_spec/3" do
    test "returns the model plus a spec that rebuilds it" do
      {model, spec} = Edifice.build_with_spec(:mlp, input_size: 6, hidden_sizes: [7])

      assert %Axon{} = model
      assert spec.arch == :mlp
      assert spec.build_opts[:hidden_sizes] == [7]

      assert %Axon{} = Edifice.build(spec.arch, spec.build_opts)
    end

    test "captures merged registry default opts (tuple registry entries)" do
      # :gru resolves to {Edifice.Recurrent, [cell_type: :gru]} — the spec must
      # record cell_type or a rebuild would silently produce the default cell
      {_model, spec} = Edifice.build_with_spec(:gru, embed_dim: 8, hidden_size: 8)

      assert spec.arch == :gru
      assert spec.build_opts[:cell_type] == :gru
    end

    test "captures normalized input-dimension aliases" do
      {_model, spec} = Edifice.build_with_spec(:mlp, embed_dim: 12)

      # normalize_input_dim fans embed_dim out to input_size and friends;
      # the merged opts must carry what the module actually consumed
      assert spec.build_opts[:input_size] == 12
    end
  end
end
