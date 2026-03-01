# Loading Pretrained Weights
> How to load HuggingFace SafeTensors checkpoints into Edifice models, write your own key maps for new architectures, and troubleshoot common mismatches.

## What This Guide Covers

Edifice can load pretrained weights from HuggingFace SafeTensors checkpoints into any Axon
model. The system handles the translation between PyTorch parameter naming conventions and
Axon's internal structure, including weight transposition, layout permutation, and key
concatenation.

This guide walks through:

1. **Quick start** -- loading a checkpoint with a built-in key map
2. **How it works** -- the key mapping and transform pipeline
3. **Built-in key maps** -- ViT, Whisper, and ConvNeXt
4. **Writing your own key map** -- for architectures not yet covered
5. **Troubleshooting** -- diagnosing shape mismatches and missing keys

**Prerequisites:** Familiarity with `Axon.build/2`, `Axon.predict/3`, and `Axon.ModelState`.
You should know how to build an Edifice model with `Edifice.build/2` or the architecture's
`build/1` function.

## Quick Start

### 1. Add the safetensors dependency

The `safetensors` package is an optional dependency. Add it to your `mix.exs`:

```elixir
defp deps do
  [
    {:edifice, "~> 0.3.0"},
    {:safetensors, "~> 0.1.3"}
  ]
end
```

Then `mix deps.get`.

### 2. Download a checkpoint

The easiest way is to use the built-in Hub downloader (requires the optional `req` dependency):

```elixir
# Add to mix.exs deps:
{:req, "~> 0.5"}
```

```elixir
alias Edifice.Pretrained.Hub

# Download and cache locally (~/.cache/edifice/)
[path] = Hub.download!("google/vit-base-patch16-224")

# For private/gated models, pass a HuggingFace token
[path] = Hub.download!("meta-llama/Llama-2-7b", token: "hf_...")
```

Alternatively, download manually with `curl`:

```bash
# ViT Base
curl -LO https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.safetensors

# Whisper Base
curl -LO https://huggingface.co/openai/whisper-base/resolve/main/model.safetensors

# ConvNeXt Tiny
curl -LO https://huggingface.co/facebook/convnext-tiny-224/resolve/main/model.safetensors
```

### 3. Load into an Edifice model

```elixir
alias Edifice.Pretrained
alias Edifice.Pretrained.{Hub, KeyMaps}

# Load ViT weights (from Hub download or local path)
[path] = Hub.download!("google/vit-base-patch16-224")
model_state = Pretrained.load(KeyMaps.ViT, path)

# Build the Edifice model
model = Edifice.Vision.ViT.build(
  image_size: 224,
  patch_size: 16,
  embed_dim: 768,
  depth: 12,
  num_heads: 12,
  num_classes: 1000
)

# Run inference
{_init_fn, predict_fn} = Axon.build(model, mode: :inference)
result = predict_fn.(model_state, %{"image" => input_tensor})
```

## How It Works

The loading pipeline has three stages:

```
SafeTensors file
    |
    v
[1] Key Mapping: "vit.encoder.layer.0.attention.output.dense.weight"
                   --> "block_0_attn_proj.kernel"
    |
    v
[2] Tensor Transform: transpose {768, 768} --> {768, 768}
                       (PyTorch [out, in] --> Axon [in, out])
    |
    v
[3] Nesting: "block_0_attn_proj.kernel"
              --> %{"block_0_attn_proj" => %{"kernel" => tensor}}
    |
    v
Axon.ModelState
```

### Stage 1: Key Mapping

Each key in the checkpoint is passed through `map_key/1`, which returns one of:

- **A string** -- the Axon parameter path (e.g., `"block_0_attn_proj.kernel"`)
- **`:skip`** -- intentionally exclude this parameter (e.g., sinusoidal position embeddings)
- **`:unmapped`** -- the key is not recognized (raises in strict mode, warns in non-strict)

### Stage 2: Tensor Transforms

After mapping, the tensor is optionally transformed. The key map provides a list of
`{regex, transform_fn}` pairs via `tensor_transforms/0`. The first regex matching the
**mapped** key determines which transform runs.

Common transforms:
- **Transpose linear weights** -- PyTorch Dense stores `{out, in}`, Axon stores `{in, out}`
- **Permute conv2d weights** -- PyTorch uses OIHW, Axon uses HWIO
- **Reshape embeddings** -- squeeze batch dimensions from `{1, seq, dim}` to `{seq, dim}`

### Stage 3: Nesting

Flat dot-separated keys like `"block_0_attn_proj.kernel"` are nested into the map structure
that `Axon.ModelState` expects: `%{"block_0_attn_proj" => %{"kernel" => tensor}}`.

### Key Concatenation (Optional)

Some architectures store separate source weights that Edifice combines into one parameter.
For example, HuggingFace ViT has separate Q, K, V weight matrices, but Edifice uses a
combined QKV projection.

Key maps can implement the optional `concat_keys/0` callback to define these rules:

```elixir
def concat_keys do
  %{
    "block_0_attn_qkv.kernel" => {[
      "block_0_attn_q.kernel",
      "block_0_attn_k.kernel",
      "block_0_attn_v.kernel"
    ], 1}   # concatenate along axis 1 (output dimension)
  }
end
```

The loader accumulates the source tensors and concatenates them when all parts arrive.

## Built-In Key Maps

### ViT (`Edifice.Pretrained.KeyMaps.ViT`)

For `google/vit-base-patch16-224` and similar ViT checkpoints.

| What | HuggingFace | Edifice |
|------|-------------|---------|
| Patch embedding | `vit.embeddings.patch_embeddings.projection.weight` | `patch_embed_proj.kernel` |
| CLS token | `vit.embeddings.cls_token` | `cls_token_proj.kernel` |
| Position embedding | `vit.embeddings.position_embeddings` | `pos_embed_proj.kernel` |
| Block attention Q/K/V | `vit.encoder.layer.{i}.attention.attention.{q,k,v}_proj.weight` | `block_{i}_attn_qkv.kernel` (concatenated) |
| Block attention output | `vit.encoder.layer.{i}.attention.output.dense.weight` | `block_{i}_attn_proj.kernel` |
| Block MLP | `vit.encoder.layer.{i}.intermediate.dense.weight` | `block_{i}_mlp_fc1.kernel` |
| Final norm | `vit.layernorm.weight` | `final_norm.gamma` |
| Classifier | `classifier.weight` | `classifier.kernel` |

Key details:
- **QKV concatenation**: Separate Q/K/V weights are concatenated along axis 1 for kernels
  and axis 0 for biases. Default is 12 layers; use `ViT.build_concat_keys(num_layers)` for
  other depths.
- **CLS token**: Squeezed from `{1, 1, D}` to `{1, D}`.
- **Position embedding**: Squeezed from `{1, S, D}` to `{S, D}`.
- **Pooler**: `vit.pooler.*` keys are skipped (not used by Edifice).

```elixir
model_state = Edifice.Pretrained.load(KeyMaps.ViT, "model.safetensors")
```

### Whisper (`Edifice.Pretrained.KeyMaps.Whisper`)

For `openai/whisper-base` and similar Whisper checkpoints.

| What | HuggingFace | Edifice |
|------|-------------|---------|
| Encoder conv | `model.encoder.conv1.weight` | `enc_conv1.kernel` |
| Encoder attention | `model.encoder.layers.{i}.self_attn.q_proj.weight` | `enc_block_{i+1}_attn_q.kernel` |
| Encoder norm | `model.encoder.layers.{i}.self_attn_layer_norm.weight` | `enc_block_{i+1}_attn_norm.gamma` |
| Encoder FFN | `model.encoder.layers.{i}.fc1.weight` | `enc_block_{i+1}_ffn_up.kernel` |
| Decoder self-attn | `model.decoder.layers.{i}.self_attn.q_proj.weight` | `dec_block_{i+1}_attn_q.kernel` |
| Decoder cross-attn | `model.decoder.layers.{i}.encoder_attn.q_proj.weight` | `dec_block_{i+1}_cross_attn_q_proj.kernel` |
| Decoder token embed | `model.decoder.embed_tokens.weight` | `dec_token_embed.kernel` |

Key details:
- **Index shift**: HuggingFace uses 0-based indices (`layers.0`), Edifice uses 1-based
  (`enc_block_1`). All indices are shifted by +1 during mapping.
- **Skipped keys**: `model.encoder.embed_positions.weight` (Edifice uses non-trainable
  sinusoidal PE) and `proj_out.weight` (Edifice has its own output projection).
- **Embedding weights**: Not transposed. The transform regex excludes `embed` keys.
- **Conv weights**: Rank-3 conv kernels pass through unchanged (transpose only affects rank-2).

```elixir
model_state = Edifice.Pretrained.load(KeyMaps.Whisper, "model.safetensors")
```

### ConvNeXt (`Edifice.Pretrained.KeyMaps.ConvNeXt`)

For `facebook/convnext-tiny-224` and similar ConvNeXt checkpoints.

| What | HuggingFace | Edifice |
|------|-------------|---------|
| Stem conv | `convnext.embeddings.patch_embeddings.weight` | `stem_conv.kernel` |
| Stem norm | `convnext.embeddings.layernorm.weight` | `stem_norm.gamma` |
| Depthwise conv | `convnext.encoder.stages.{s}.layers.{b}.dwconv.weight` | `stage{s}_block{b}_dw_conv.kernel` |
| Pointwise expand | `convnext.encoder.stages.{s}.layers.{b}.pwconv1.weight` | `stage{s}_block{b}_pw_expand.kernel` |
| Layer scale | `convnext.encoder.stages.{s}.layers.{b}.layer_scale_parameter` | `stage{s}_block{b}_layer_scale.stage{s}_block{b}_gamma` |
| Downsample | `convnext.encoder.stages.{s+1}.downsampling_layer.{0,1}.weight` | `downsample_{s}_{norm,conv}.{gamma,kernel}` |
| Classifier | `classifier.weight` | `classifier.kernel` |

Key details:
- **Conv2d layout**: PyTorch stores OIHW `{out, in, H, W}`, Axon uses HWIO `{H, W, in, out}`.
  All conv kernels are permuted with `Nx.transpose(axes: [2, 3, 1, 0])`.
- **Pointwise convolutions**: HuggingFace uses `nn.Linear` (rank-2 weights), Edifice uses
  `Axon.conv` with 1x1 kernels (rank-4). Linear `{out, in}` is transposed and reshaped to
  `{1, 1, in, out}`.
- **Layer scale**: Reshaped from `{dim}` to `{1, 1, 1, dim}` for broadcasting.
- **Downsample index shift**: HuggingFace places downsampling at the start of stages 1-3.
  Edifice places it between stages. HF `stages.{i+1}.downsampling_layer` maps to Edifice
  `downsample_{i}`.

```elixir
model_state = Edifice.Pretrained.load(KeyMaps.ConvNeXt, "model.safetensors")
```

## Writing Your Own Key Map

To load a new architecture, implement the `Edifice.Pretrained.KeyMap` behaviour.

### Step 1: Inspect the checkpoint

Use `list_keys/1` to see what parameter names the checkpoint contains:

```elixir
Edifice.Pretrained.list_keys("model.safetensors")
#=> ["model.encoder.layers.0.self_attn.k_proj.bias",
#    "model.encoder.layers.0.self_attn.k_proj.weight",
#    ...]
```

### Step 2: Inspect the Edifice model

Build your target model and flatten its parameters to see what Axon expects:

```elixir
model = Edifice.Vision.ViT.build(image_size: 224, patch_size: 16, ...)
{init_fn, _} = Axon.build(model, mode: :inference)
state = init_fn.(input_template, Axon.ModelState.empty())

state.data
|> Edifice.Pretrained.Transform.flatten_params()
|> Map.keys()
|> Enum.sort()
#=> ["block_0_attn_proj.bias", "block_0_attn_proj.kernel", ...]
```

### Step 3: Write map_key/1

Match each checkpoint key to its Axon counterpart. Use pattern matching for fixed keys
and regex for parametric patterns:

```elixir
defmodule MyApp.KeyMaps.MyModel do
  @behaviour Edifice.Pretrained.KeyMap

  @layer_re ~r/^model\.layers\.(\d+)\.(.+)$/

  @impl true
  def map_key("model.embed.weight"), do: "embed.kernel"
  def map_key("model.norm.weight"), do: "final_norm.gamma"

  def map_key(key) do
    case Regex.run(@layer_re, key) do
      [_, idx, rest] -> map_layer(idx, rest)
      nil -> :unmapped
    end
  end

  defp map_layer(i, "self_attn.q_proj.weight"), do: "block_#{i}_attn_q.kernel"
  defp map_layer(i, "self_attn.q_proj.bias"), do: "block_#{i}_attn_q.bias"
  # ... more patterns ...
  defp map_layer(_i, _rest), do: :unmapped
end
```

Tips for `map_key/1`:

- Return `:skip` for keys you intentionally want to ignore (e.g., non-trainable buffers,
  position embeddings you compute differently).
- Return `:unmapped` for keys you don't recognize. In strict mode (default), any unmapped
  key raises an error. Use `strict: false` during development to see what you're missing.
- Use string interpolation with captured regex groups for index-based patterns.
- If the source uses 0-based indices but Edifice uses 1-based, do the arithmetic in
  `map_key`: `idx = String.to_integer(idx_str) + 1`.

### Step 4: Write tensor_transforms/0

Define how tensors should be reshaped after key mapping:

```elixir
@impl true
def tensor_transforms do
  [
    # Transpose all linear weight matrices
    {~r/\.kernel$/, &Edifice.Pretrained.Transform.transpose_linear/1},
    # Leave everything else unchanged
    {~r/\./, &Function.identity/1}
  ]
end
```

The transforms are checked in order -- the first matching regex wins. Common patterns:

| Transform | When to use |
|-----------|-------------|
| `Transform.transpose_linear/1` | Linear/Dense weights: PyTorch `{out, in}` to Axon `{in, out}` |
| `Nx.transpose(t, axes: [2, 3, 1, 0])` | Conv2d weights: PyTorch OIHW to Axon HWIO |
| `Nx.squeeze(t, axes: [0])` | Remove batch dim from `{1, S, D}` to `{S, D}` |
| `Nx.reshape(t, {1, 1, in, out})` | Linear to 1x1 conv kernel |
| `Function.identity/1` | No transform needed |

Important: `transpose_linear/1` only transposes rank-2 tensors. Rank-3 (1D convs) and
rank-1 (biases) pass through unchanged. Use this to your advantage -- a single
`{~r/\.kernel$/, &Transform.transpose_linear/1}` rule handles both linear and conv weights
correctly when convs are rank-3.

If you need a regex that matches kernels but *not* embedding kernels, use a negative
lookbehind: `~r/(?<!embed)\.kernel$/`.

### Step 5: Test it

Write tests using synthetic SafeTensors fixtures:

```elixir
test "loads checkpoint into ModelState" do
  tensors = %{
    "model.embed.weight" => Nx.iota({100, 16}, type: :f32),
    "model.layers.0.self_attn.q_proj.weight" => Nx.iota({16, 16}, type: :f32),
    # ...
  }

  path = write_safetensors_fixture(tensors)
  model_state = Edifice.Pretrained.load(MyApp.KeyMaps.MyModel, path)

  assert %Axon.ModelState{} = model_state
  assert Nx.shape(model_state.data["block_0_attn_q"]["kernel"]) == {16, 16}
end
```

For thorough coverage, also write a round-trip test: init an Edifice model, reverse-map its
params to HuggingFace format, write to SafeTensors, load back, and verify all params match.
See `test/edifice/pretrained/round_trip_test.exs` for examples.

## Load Options

`Edifice.Pretrained.load/3` accepts these options:

| Option | Default | Description |
|--------|---------|-------------|
| `:strict` | `true` | Raise on unmapped checkpoint keys. Set to `false` during development. |
| `:dtype` | `nil` | Cast all tensors to this type (e.g., `:f32`, `:bf16`). `nil` keeps original. |

```elixir
# Non-strict loading (warns instead of raising on unknown keys)
model_state = Pretrained.load(KeyMaps.ViT, path, strict: false)

# Cast all weights to bf16
model_state = Pretrained.load(KeyMaps.ViT, path, dtype: :bf16)
```

## Troubleshooting

### "Strict loading failed: N unmapped key(s)"

The checkpoint contains keys your key map doesn't handle. Either:
- Add `map_key/1` clauses for the missing keys
- Return `:skip` if they're intentionally unused
- Use `strict: false` to skip them with a warning

Run `Pretrained.list_keys(path)` to see all checkpoint keys, then compare with your
`map_key/1` clauses.

### Shape mismatch at inference time

The loaded tensor shape doesn't match what the Axon model expects. Common causes:

1. **Missing transpose** -- PyTorch Dense stores `{out, in}`, Axon stores `{in, out}`.
   Add `Transform.transpose_linear/1` to your `tensor_transforms/0`.

2. **Wrong conv layout** -- PyTorch conv2d uses OIHW, Axon uses HWIO. Use
   `Nx.transpose(tensor, axes: [2, 3, 1, 0])`.

3. **Embedding transpose** -- Embedding weights should NOT be transposed. If your
   `tensor_transforms/0` has a broad `~r/\.kernel$/` rule, it may catch embedding
   kernels too. Use a negative lookbehind: `~r/(?<!embed)\.kernel$/`.

4. **Linear vs conv mismatch** -- Some architectures (ConvNeXt) use `nn.Linear` in
   PyTorch but `Axon.conv` with 1x1 kernels in Edifice. You need to transpose AND
   reshape: `{out, in}` -> transpose -> `{in, out}` -> reshape -> `{1, 1, in, out}`.

5. **Concat axis** -- When concatenating Q/K/V, make sure the axis matches the Axon
   convention. For Axon Dense kernels `{in, out}`, concatenate along axis 1 (the output
   dimension). For biases `{out}`, concatenate along axis 0.

### Missing parameters after loading

Some Edifice layers have parameters that don't exist in the HuggingFace checkpoint:

- **Dense layer biases** for embedding-like layers (CLS token, position embedding) --
  HuggingFace stores these as bare tensors, but Edifice wraps them in Dense layers that
  also have a bias parameter. These biases initialize to zero and generally don't affect
  output.
- **Output projection layers** -- Edifice may define its own output projection (e.g.,
  Whisper's `dec_output_proj`) that has no HuggingFace counterpart. These keep their
  random initialization.

Use `strict: false` to load what's available. The missing parameters will use whatever
values the Axon init function provides.

### LayerNorm parameter names

Axon's `layer_norm` uses `.gamma` and `.beta` for its parameters (not `.scale` and `.bias`,
not `.weight` and `.bias`). Make sure your key map targets these names:

```elixir
# Correct
def map_key("model.norm.weight"), do: "final_norm.gamma"
def map_key("model.norm.bias"), do: "final_norm.beta"

# Wrong -- these won't match Axon's parameter names
def map_key("model.norm.weight"), do: "final_norm.scale"
def map_key("model.norm.bias"), do: "final_norm.bias"
```

## HuggingFace Hub Integration

`Edifice.Pretrained.Hub` downloads checkpoints directly from HuggingFace with local caching.
Requires the optional `req` dependency (`{:req, "~> 0.5"}`).

### Single-file models

```elixir
[path] = Hub.download!("google/vit-base-patch16-224")
model_state = Pretrained.load(KeyMaps.ViT, path)
```

### Sharded models

Large models split weights across multiple files. Hub detects this automatically via
`model.safetensors.index.json`:

```elixir
paths = Hub.download!("bigscience/bloom")
model_state = Pretrained.load_sharded(MyKeyMap, paths)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `:revision` | `"main"` | Git revision (branch, tag, or commit hash) |
| `:cache_dir` | `~/.cache/edifice` | Override local cache directory |
| `:force` | `false` | Re-download even if cached |
| `:token` | `nil` | HuggingFace API token for private/gated models |

### Cache location

```elixir
Hub.cache_path("google/vit-base-patch16-224")
#=> "~/.cache/edifice/google/vit-base-patch16-224"
```

Files are downloaded atomically (to a `.part` temp file, then renamed) so interrupted
downloads won't leave corrupt files in the cache.

## What's Next

- **Architecture key maps**: Add key maps for more architectures by following the pattern above.
- **Model card metadata**: Auto-detect architecture type and build opts from HuggingFace
  `config.json`, enabling `Edifice.Pretrained.from_hub("google/vit-base-patch16-224")`
  without manually specifying a key map.
- **Numerical validation**: Compare Edifice forward pass outputs against PyTorch reference
  outputs to verify correctness beyond just shape matching.
