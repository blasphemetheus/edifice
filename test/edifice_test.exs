defmodule EdificeTest do
  use ExUnit.Case, async: true

  describe "list_architectures/0" do
    test "returns a sorted list of atoms" do
      archs = Edifice.list_architectures()
      assert is_list(archs)
      assert length(archs) >= 90
      assert archs == Enum.sort(archs)
      assert :mamba in archs
      assert :mlp in archs
      assert :lstm in archs
      assert :attention in archs
      assert :vae in archs
      assert :gcn in archs
      assert :resnet in archs
    end

    test "includes architectures from all families" do
      archs = Edifice.list_architectures()

      # Original families
      assert :gan in archs
      assert :deep_sets in archs
      assert :hopfield in archs
      assert :moe in archs
      assert :liquid in archs
      assert :snn in archs

      # New architectures from expansion
      assert :vit in archs
      assert :gqa in archs
      assert :s4 in archs
      assert :dit in archs
      assert :min_gru in archs
      assert :simclr in archs
      assert :graph_sage in archs
      assert :lora in archs
      assert :tabnet in archs
      assert :neural_ode in archs
    end
  end

  describe "list_families/0" do
    test "returns a map of family atoms to architecture lists" do
      families = Edifice.list_families()
      assert is_map(families)
      assert Map.has_key?(families, :ssm)
      assert Map.has_key?(families, :feedforward)
      assert Map.has_key?(families, :attention)
      assert Map.has_key?(families, :generative)
      assert Map.has_key?(families, :convolutional)
      assert :mamba in families.ssm
      assert :mlp in families.feedforward
    end

    test "includes all expected families" do
      families = Edifice.list_families()

      assert Map.has_key?(families, :graph)
      assert Map.has_key?(families, :sets)
      assert Map.has_key?(families, :energy)
      assert Map.has_key?(families, :meta)
      assert Map.has_key?(families, :liquid)
      assert Map.has_key?(families, :neuromorphic)
      assert Map.has_key?(families, :vision)
      assert Map.has_key?(families, :contrastive)
      assert Map.has_key?(families, :probabilistic)
      assert Map.has_key?(families, :memory)
    end

    test "family members are all atoms" do
      families = Edifice.list_families()

      Enum.each(families, fn {_family, members} ->
        assert is_list(members)
        assert Enum.all?(members, &is_atom/1)
      end)
    end

    test "all family members appear in list_architectures" do
      architectures = Edifice.list_architectures()
      families = Edifice.list_families()

      all_members =
        families
        |> Map.values()
        |> List.flatten()

      Enum.each(all_members, fn member ->
        assert member in architectures,
               "#{inspect(member)} from families not found in list_architectures/0"
      end)
    end
  end

  describe "module_for/1" do
    test "returns the correct module for known architectures" do
      assert Edifice.module_for(:mamba) == Edifice.SSM.Mamba
      assert Edifice.module_for(:mlp) == Edifice.Feedforward.MLP
      assert Edifice.module_for(:lstm) == Edifice.Recurrent
      assert Edifice.module_for(:attention) == Edifice.Attention.MultiHead
      assert Edifice.module_for(:vae) == Edifice.Generative.VAE
      assert Edifice.module_for(:gcn) == Edifice.Graph.GCN
    end

    test "returns correct modules for additional architectures" do
      assert Edifice.module_for(:gan) == Edifice.Generative.GAN
      assert Edifice.module_for(:deep_sets) == Edifice.Sets.DeepSets
      assert Edifice.module_for(:hopfield) == Edifice.Energy.Hopfield
      assert Edifice.module_for(:moe) == Edifice.Meta.MoE
      assert Edifice.module_for(:liquid) == Edifice.Liquid
    end

    test "returns correct modules for new expansion architectures" do
      assert Edifice.module_for(:vit) == Edifice.Vision.ViT
      assert Edifice.module_for(:gqa) == Edifice.Attention.GQA
      assert Edifice.module_for(:s4) == Edifice.SSM.S4
      assert Edifice.module_for(:dit) == Edifice.Generative.DiT
      assert Edifice.module_for(:min_gru) == Edifice.Recurrent.MinGRU
      assert Edifice.module_for(:simclr) == Edifice.Contrastive.SimCLR
      assert Edifice.module_for(:graph_sage) == Edifice.Graph.GraphSAGE
      assert Edifice.module_for(:lora) == Edifice.Meta.LoRA
      assert Edifice.module_for(:tabnet) == Edifice.Feedforward.TabNet
      assert Edifice.module_for(:neural_ode) == Edifice.Energy.NeuralODE
      assert Edifice.module_for(:evidential) == Edifice.Probabilistic.EvidentialNN
      assert Edifice.module_for(:ann2snn) == Edifice.Neuromorphic.ANN2SNN
    end

    test "raises for unknown architecture" do
      assert_raise ArgumentError, ~r/Unknown architecture/, fn ->
        Edifice.module_for(:nonexistent)
      end
    end
  end

  describe "build/2" do
    test "builds MLP model" do
      model = Edifice.build(:mlp, input_size: 32, hidden_sizes: [64, 32])
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 32}))
      assert Nx.shape(output) == {2, 32}
    end

    test "builds MLP with single hidden layer" do
      model = Edifice.build(:mlp, input_size: 32, hidden_sizes: [64])
      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 32}))

      assert Nx.shape(output) == {2, 64}
    end

    test "raises for unknown architecture" do
      assert_raise ArgumentError, ~r/Unknown architecture/, fn ->
        Edifice.build(:nonexistent, [])
      end
    end
  end
end
