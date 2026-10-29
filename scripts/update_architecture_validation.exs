# AI-generated. Evidence definitions and authorship poem: guides/validation.md.
# Run with --check in CI, or without arguments to refresh the index column.
index = Path.expand("../guides/architecture_index.md", __DIR__)
original = File.read!(index)

reference = ~w(
  Edifice.SSM.Mamba2 Edifice.SSM.Mamba Edifice.SSM.MambaSSD
  Edifice.Recurrent.MinGRU Edifice.Recurrent.DeepResLSTM
  Edifice.Recurrent.DeltaNet Edifice.Graph.GAT Edifice.Attention.GQA
  Edifice.Generative.DiT
)
checkpoint = ~w(Edifice.Convolutional.ResNet Edifice.Vision.ViT
  Edifice.Vision.ConvNeXt Edifice.Detection.DETR)
behavior = ~w(Edifice.Transformer.DecoderOnly Edifice.Recurrent.GatedDeltaNet
  Edifice.SSM.GatedSSM Edifice.Attention.NSA Edifice.Attention.MoBA
  Edifice.Attention.RetNet Edifice.Recurrent.Reservoir)

label = fn module ->
  cond do
    module == "Edifice.Recurrent" -> "GRU: reference; LSTM: behavior"
    module == "Edifice.Audio.Whisper" -> "Checkpoint; backend limitation"
    module in reference -> "Reference (limited scope)"
    module in checkpoint -> "Checkpoint (limited scope)"
    module in behavior -> "Behavior"
    true -> "Unreviewed"
  end
end

{lines, _} =
  original
  |> String.split("\n")
  |> Enum.map_reduce(false, fn line, architecture_table? ->
    cond do
      String.starts_with?(line, "| Architecture | Module |") ->
        {"| Architecture | Module | Key Feature | Validation evidence |", true}

      architecture_table? and Regex.match?(~r/^\|[-| ]+\|$/, line) ->
        {"|-------------|--------|-------------|---------------------|", true}

      architecture_table? and String.starts_with?(line, "|") ->
        cells = line |> String.split("|") |> Enum.map(&String.trim/1)
        [_, name, module_cell, feature | _] = cells
        [_, module] = Regex.run(~r/`(Edifice\.[\w.]+)`/, module_cell)
        evidence = "[#{label.(module)}](validation.md)"
        {"| #{name} | #{module_cell} | #{feature} | #{evidence} |", true}

      true ->
        {line, false}
    end
  end)

updated = Enum.join(lines, "\n")

if "--check" in System.argv() do
  if updated != original, do: raise("Run elixir scripts/update_architecture_validation.exs")
  IO.puts("Architecture evidence labels are current")
else
  File.write!(index, updated)
end
