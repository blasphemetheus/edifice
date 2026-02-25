defmodule Edifice.Export.GGUF do
  @moduledoc """
  GGUF (GPT-Generated Unified Format) exporter for Edifice models.

  Exports Edifice decoder_only models to the GGUF format used by llama.cpp,
  Ollama, and other CPU/GPU inference engines.

  ## GGUF Format Overview

  GGUF is a self-describing binary format:
  1. Magic bytes ('GGUF' = 0x46554747)
  2. Version (uint32, currently 3)
  3. Tensor count (uint64)
  4. Metadata KV count (uint64)
  5. Metadata key-value pairs
  6. Tensor info array
  7. Padding to alignment (32 bytes default)
  8. Tensor data (raw bytes, row-major)

  ## Usage

      # Export a decoder_only model with Q8_0 quantization (default)
      Edifice.Export.GGUF.export(params, config, "model.gguf")

      # Export with F16 precision
      Edifice.Export.GGUF.export(params, config, "model.gguf", quantization: :f16)

      # Export with F32 precision
      Edifice.Export.GGUF.export(params, config, "model.gguf", quantization: :f32)

  ## Supported Quantization Types

  - `:f32` - Full 32-bit floating point (file type 0)
  - `:f16` - Half precision 16-bit floating point (file type 1)
  - `:q8_0` - 8-bit quantization with block-wise scaling (file type 7)

  ## References

  - GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
  - llama.cpp: https://github.com/ggerganov/llama.cpp
  """

  # GGUF magic number: 'GGUF' in little-endian
  @magic 0x46554747
  @version 3
  @default_alignment 32

  # GGUF type codes for metadata values (some are reserved for future use)
  # @type_uint8 0
  # @type_int8 1
  # @type_uint16 2
  # @type_int16 3
  @type_uint32 4
  @type_int32 5
  @type_float32 6
  @type_bool 7
  @type_string 8
  # @type_array 9
  @type_uint64 10
  # @type_int64 11
  # @type_float64 12

  # GGML tensor types
  @ggml_type_f32 0
  @ggml_type_f16 1
  @ggml_type_q8_0 8

  # File types (quantization indicators in metadata)
  @file_type_f32 0
  @file_type_f16 1
  @file_type_q8_0 7

  @doc """
  Export an Edifice :decoder_only model to GGUF format.

  ## Parameters

    - `model_params` - The Axon model state (map of layer names to param tensors)
    - `model_config` - Configuration map with model hyperparameters:
      - `:hidden_size` or `:embed_dim` - Model hidden dimension
      - `:num_heads` - Number of attention heads
      - `:num_kv_heads` - Number of key/value heads (GQA)
      - `:num_layers` - Number of transformer blocks
      - `:vocab_size` - Vocabulary size (optional, for token embedding)
      - `:context_length` - Maximum context length (default: 2048)
      - `:name` - Model name (default: "edifice_model")
    - `output_path` - Path to write the GGUF file
    - `opts` - Options:
      - `:quantization` - `:f32`, `:f16`, or `:q8_0` (default: `:q8_0`)
      - `:architecture` - Architecture name (default: "llama")

  ## Returns

    `:ok` on success, or `{:error, reason}` on failure.
  """
  @spec export(map(), map(), Path.t(), keyword()) :: :ok | {:error, term()}
  def export(model_params, model_config, output_path, opts \\ []) do
    case File.open(output_path, [:write, :binary]) do
      {:ok, io} ->
        try do
          write_gguf(io, model_params, model_config, opts)
          :ok
        after
          File.close(io)
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Write GGUF binary to an IO device.

  This is the main serialization function that writes the complete GGUF
  structure including header, metadata, tensor info, and tensor data.
  """
  @spec write_gguf(IO.device(), map(), map(), keyword()) :: :ok
  def write_gguf(io, params, config, opts) do
    quantization = Keyword.get(opts, :quantization, :q8_0)
    # architecture is passed through opts to encode_metadata
    _architecture = Keyword.get(opts, :architecture, "llama")

    # Map param names and get tensor list
    num_layers = Map.get(config, :num_layers, 4)
    tensors = map_param_names(params, num_layers)

    # Build metadata and tensor info
    metadata_binary = encode_metadata(config, opts)
    {tensor_info_binary, _tensor_infos} = encode_tensor_info(tensors, quantization)

    # Calculate metadata count (number of KV pairs)
    metadata_count = count_metadata_entries(config, opts)

    # Write header
    header = <<
      @magic::little-32,
      @version::little-32,
      length(tensors)::little-64,
      metadata_count::little-64
    >>

    IO.binwrite(io, header)

    # Write metadata
    IO.binwrite(io, metadata_binary)

    # Write tensor info
    IO.binwrite(io, tensor_info_binary)

    # Calculate current position and add padding
    header_size = byte_size(header)
    metadata_size = byte_size(metadata_binary)
    tensor_info_size = byte_size(tensor_info_binary)
    current_pos = header_size + metadata_size + tensor_info_size

    padding_size = padding_to_alignment(current_pos, @default_alignment)
    IO.binwrite(io, :binary.copy(<<0>>, padding_size))

    # Write tensor data
    tensor_data = encode_tensors(tensors, quantization)
    IO.binwrite(io, tensor_data)

    :ok
  end

  @doc """
  Encode metadata key-value section.

  Encodes all required metadata for LLaMA-compatible models including:
  - General architecture info
  - Model dimensions
  - Attention configuration
  - Tokenizer stub info
  """
  @spec encode_metadata(map(), keyword()) :: binary()
  def encode_metadata(config, opts) do
    architecture = Keyword.get(opts, :architecture, "llama")
    quantization = Keyword.get(opts, :quantization, :q8_0)

    hidden_size = Map.get(config, :hidden_size) || Map.get(config, :embed_dim, 256)
    num_heads = Map.get(config, :num_heads, 8)
    num_kv_heads = Map.get(config, :num_kv_heads, num_heads)
    num_layers = Map.get(config, :num_layers, 4)
    context_length = Map.get(config, :context_length, 2048)
    vocab_size = Map.get(config, :vocab_size, 32000)
    name = Map.get(config, :name, "edifice_model")

    # FFN hidden size (SwiGLU uses ~2.667x expansion)
    ffn_hidden_size = Map.get(config, :ffn_hidden_size, round(hidden_size * 2.667))

    # Head dimension for RoPE
    head_dim = div(hidden_size, num_heads)

    file_type =
      case quantization do
        :f32 -> @file_type_f32
        :f16 -> @file_type_f16
        :q8_0 -> @file_type_q8_0
      end

    # Build metadata entries
    entries = [
      # General
      encode_kv("general.architecture", :string, architecture),
      encode_kv("general.name", :string, name),
      encode_kv("general.file_type", :uint32, file_type),

      # Architecture-specific (using arch prefix)
      encode_kv("#{architecture}.context_length", :uint32, context_length),
      encode_kv("#{architecture}.embedding_length", :uint32, hidden_size),
      encode_kv("#{architecture}.feed_forward_length", :uint32, ffn_hidden_size),
      encode_kv("#{architecture}.attention.head_count", :uint32, num_heads),
      encode_kv("#{architecture}.attention.head_count_kv", :uint32, num_kv_heads),
      encode_kv("#{architecture}.block_count", :uint32, num_layers),
      encode_kv("#{architecture}.rope.dimension_count", :uint32, head_dim),
      encode_kv("#{architecture}.vocab_size", :uint32, vocab_size),

      # Tokenizer stub (required for llama.cpp compatibility)
      encode_kv("tokenizer.ggml.model", :string, "llama")
    ]

    IO.iodata_to_binary(entries)
  end

  # Count metadata entries for header
  defp count_metadata_entries(_config, _opts) do
    # Fixed count based on encode_metadata
    12
  end

  @doc """
  Encode tensor info section.

  Returns a tuple of {tensor_info_binary, tensor_info_list} where each
  tensor info contains: name, dimensions, type, and data offset.
  """
  @spec encode_tensor_info([{String.t(), Nx.Tensor.t()}], atom()) ::
          {binary(), [{String.t(), list(), non_neg_integer(), non_neg_integer()}]}
  def encode_tensor_info(tensors, quantization) do
    ggml_type =
      case quantization do
        :f32 -> @ggml_type_f32
        :f16 -> @ggml_type_f16
        :q8_0 -> @ggml_type_q8_0
      end

    {info_binaries, tensor_infos, _offset} =
      Enum.reduce(tensors, {[], [], 0}, fn {name, tensor}, {binaries, infos, offset} ->
        shape = Nx.shape(tensor) |> Tuple.to_list()
        n_dims = length(shape)

        # Calculate data size based on quantization
        data_size = tensor_data_size(tensor, quantization)

        # Encode tensor info entry
        # Format: name (string) + n_dims (uint32) + dims (uint64 array) + type (uint32) + offset (uint64)
        name_binary = encode_string(name)

        dims_binary =
          shape
          |> Enum.map(fn dim -> <<dim::little-64>> end)
          |> IO.iodata_to_binary()

        info_binary =
          <<name_binary::binary, n_dims::little-32, dims_binary::binary, ggml_type::little-32,
            offset::little-64>>

        info = {name, shape, ggml_type, offset}

        # Calculate padded offset for next tensor
        next_offset = offset + data_size
        padded_offset = next_offset + padding_to_alignment(next_offset, @default_alignment)

        {[info_binary | binaries], [info | infos], padded_offset}
      end)

    {IO.iodata_to_binary(Enum.reverse(info_binaries)), Enum.reverse(tensor_infos)}
  end

  @doc """
  Encode tensor data with optional quantization.

  Converts all tensors to the specified format and concatenates them
  with proper alignment padding.
  """
  @spec encode_tensors([{String.t(), Nx.Tensor.t()}], atom()) :: binary()
  def encode_tensors(tensors, quantization) do
    {data_parts, _offset} =
      Enum.reduce(tensors, {[], 0}, fn {_name, tensor}, {parts, offset} ->
        # Convert tensor to target format
        data =
          case quantization do
            :f32 ->
              tensor |> Nx.as_type(:f32) |> Nx.to_binary()

            :f16 ->
              tensor |> Nx.as_type(:f16) |> Nx.to_binary()

            :q8_0 ->
              quantize_q8_0(tensor)
          end

        # Calculate padding
        data_size = byte_size(data)
        next_offset = offset + data_size
        padding_size = padding_to_alignment(next_offset, @default_alignment)
        padding = :binary.copy(<<0>>, padding_size)

        {[data, padding | parts], next_offset + padding_size}
      end)

    IO.iodata_to_binary(Enum.reverse(data_parts))
  end

  @doc """
  Q8_0 quantize a tensor.

  Q8_0 format: Each block of 32 float32 values becomes:
  - 1 float16 scale value
  - 32 int8 quantized values

  The scale is computed as max(abs(block)) / 127.0, and each value
  is quantized as round(value / scale).
  """
  @spec quantize_q8_0(Nx.Tensor.t()) :: binary()
  def quantize_q8_0(tensor) do
    # Flatten and convert to f32
    flat = tensor |> Nx.reshape({:auto}) |> Nx.as_type(:f32)
    total_elements = Nx.size(flat)

    # Pad to multiple of 32
    block_size = 32
    remainder = rem(total_elements, block_size)

    padded =
      if remainder > 0 do
        pad_count = block_size - remainder
        Nx.concatenate([flat, Nx.broadcast(0.0, {pad_count})])
      else
        flat
      end

    padded_size = Nx.size(padded)
    num_blocks = div(padded_size, block_size)

    # Reshape into blocks
    blocks = Nx.reshape(padded, {num_blocks, block_size})

    # Process each block
    blocks
    |> Nx.to_batched(1)
    |> Enum.map(fn block ->
      block = Nx.squeeze(block)
      quantize_q8_0_block(block)
    end)
    |> IO.iodata_to_binary()
  end

  # Quantize a single block of 32 values to Q8_0 format
  defp quantize_q8_0_block(block) do
    # Find max absolute value for scale
    abs_max = block |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    # Compute scale (avoid division by zero)
    scale = if abs_max > 0, do: abs_max / 127.0, else: 1.0

    # Quantize values
    quantized =
      block
      |> Nx.divide(scale)
      |> Nx.round()
      |> Nx.clip(-128, 127)
      |> Nx.as_type(:s8)

    # Pack: f16 scale + 32 int8 values
    scale_f16 = <<scale::float-little-16>>
    values_binary = Nx.to_binary(quantized)

    <<scale_f16::binary, values_binary::binary>>
  end

  @doc """
  Map Edifice decoder_only param names to GGUF tensor names.

  Converts the Axon parameter map structure to GGUF's LLaMA-compatible naming:

  | Edifice Name | GGUF Name |
  |--------------|-----------|
  | input_projection.kernel | token_embd.weight |
  | decoder_block_{N}_attn_norm_gamma | blk.{N-1}.attn_norm.weight |
  | decoder_block_{N}_attn_q_proj.kernel | blk.{N-1}.attn_q.weight |
  | decoder_block_{N}_attn_k_proj.kernel | blk.{N-1}.attn_k.weight |
  | decoder_block_{N}_attn_v_proj.kernel | blk.{N-1}.attn_v.weight |
  | decoder_block_{N}_attn_out_proj.kernel | blk.{N-1}.attn_output.weight |
  | decoder_block_{N}_ffn_norm_gamma | blk.{N-1}.ffn_norm.weight |
  | decoder_block_{N}_ffn_gate.kernel | blk.{N-1}.ffn_gate.weight |
  | decoder_block_{N}_ffn_up.kernel | blk.{N-1}.ffn_up.weight |
  | decoder_block_{N}_ffn_down.kernel | blk.{N-1}.ffn_down.weight |
  | final_norm.gamma/.beta | output_norm.weight |
  """
  @spec map_param_names(map(), pos_integer()) :: [{String.t(), Nx.Tensor.t()}]
  def map_param_names(params, num_layers) do
    # Flatten nested param structure if needed
    flat_params = flatten_params(params)

    # Map known Edifice param patterns to GGUF names
    # Per-layer mappings
    mappings =
      [
        # Token embedding / input projection
        {~r/input_projection\.kernel$/, "token_embd.weight"},
        {~r/input_projection\.bias$/, "token_embd.bias"},

        # Output / LM head (if separate from embedding)
        {~r/lm_head\.kernel$/, "output.weight"},
        {~r/lm_head\.bias$/, "output.bias"},

        # Final norm
        {~r/final_norm\.gamma$/, "output_norm.weight"},
        {~r/final_norm\.beta$/, "output_norm.bias"},
        {~r/final_norm_gamma$/, "output_norm.weight"}
      ] ++
        Enum.flat_map(1..num_layers, fn layer ->
          # GGUF uses 0-indexed layers
          gguf_layer = layer - 1

          [
            # Attention norm
            {~r/decoder_block_#{layer}_attn_norm\.gamma$/, "blk.#{gguf_layer}.attn_norm.weight"},
            {~r/decoder_block_#{layer}_attn_norm_gamma$/, "blk.#{gguf_layer}.attn_norm.weight"},

            # Attention projections
            {~r/decoder_block_#{layer}_attn_q_proj\.kernel$/, "blk.#{gguf_layer}.attn_q.weight"},
            {~r/decoder_block_#{layer}_attn_q_proj\.bias$/, "blk.#{gguf_layer}.attn_q.bias"},
            {~r/decoder_block_#{layer}_attn_k_proj\.kernel$/, "blk.#{gguf_layer}.attn_k.weight"},
            {~r/decoder_block_#{layer}_attn_k_proj\.bias$/, "blk.#{gguf_layer}.attn_k.bias"},
            {~r/decoder_block_#{layer}_attn_v_proj\.kernel$/, "blk.#{gguf_layer}.attn_v.weight"},
            {~r/decoder_block_#{layer}_attn_v_proj\.bias$/, "blk.#{gguf_layer}.attn_v.bias"},
            {~r/decoder_block_#{layer}_attn_out_proj\.kernel$/,
             "blk.#{gguf_layer}.attn_output.weight"},
            {~r/decoder_block_#{layer}_attn_out_proj\.bias$/,
             "blk.#{gguf_layer}.attn_output.bias"},

            # FFN norm
            {~r/decoder_block_#{layer}_ffn_norm\.gamma$/, "blk.#{gguf_layer}.ffn_norm.weight"},
            {~r/decoder_block_#{layer}_ffn_norm_gamma$/, "blk.#{gguf_layer}.ffn_norm.weight"},

            # FFN projections (SwiGLU)
            {~r/decoder_block_#{layer}_ffn_gate\.kernel$/, "blk.#{gguf_layer}.ffn_gate.weight"},
            {~r/decoder_block_#{layer}_ffn_gate\.bias$/, "blk.#{gguf_layer}.ffn_gate.bias"},
            {~r/decoder_block_#{layer}_ffn_up\.kernel$/, "blk.#{gguf_layer}.ffn_up.weight"},
            {~r/decoder_block_#{layer}_ffn_up\.bias$/, "blk.#{gguf_layer}.ffn_up.bias"},
            {~r/decoder_block_#{layer}_ffn_down\.kernel$/, "blk.#{gguf_layer}.ffn_down.weight"},
            {~r/decoder_block_#{layer}_ffn_down\.bias$/, "blk.#{gguf_layer}.ffn_down.bias"}
          ]
        end)

    # Apply mappings
    flat_params
    |> Enum.map(fn {name, tensor} ->
      gguf_name =
        Enum.find_value(mappings, name, fn {pattern, gguf_name} ->
          if Regex.match?(pattern, name), do: gguf_name
        end)

      {gguf_name, tensor}
    end)
    |> Enum.sort_by(fn {name, _} -> name end)
  end

  # Flatten nested Axon.ModelState structure to flat param map
  defp flatten_params(%Axon.ModelState{data: data}) do
    flatten_params(data)
  end

  defp flatten_params(params) when is_map(params) do
    Enum.flat_map(params, fn {key, value} ->
      case value do
        %Nx.Tensor{} = tensor ->
          [{to_string(key), tensor}]

        nested when is_map(nested) ->
          nested
          |> flatten_params()
          |> Enum.map(fn {nested_key, tensor} ->
            {"#{key}.#{nested_key}", tensor}
          end)

        _ ->
          []
      end
    end)
  end

  # Encode a key-value pair for metadata
  defp encode_kv(key, type, value) do
    key_binary = encode_string(key)
    {type_code, value_binary} = encode_value(type, value)
    <<key_binary::binary, type_code::little-32, value_binary::binary>>
  end

  # Encode a GGUF string: uint64 length + UTF-8 bytes
  defp encode_string(str) when is_binary(str) do
    <<byte_size(str)::little-64, str::binary>>
  end

  # Encode typed values
  defp encode_value(:string, value) when is_binary(value) do
    {@type_string, encode_string(value)}
  end

  defp encode_value(:uint32, value) when is_integer(value) do
    {@type_uint32, <<value::little-32>>}
  end

  defp encode_value(:uint64, value) when is_integer(value) do
    {@type_uint64, <<value::little-64>>}
  end

  defp encode_value(:int32, value) when is_integer(value) do
    {@type_int32, <<value::little-signed-32>>}
  end

  defp encode_value(:float32, value) when is_number(value) do
    {@type_float32, <<value::little-float-32>>}
  end

  defp encode_value(:bool, value) when is_boolean(value) do
    {@type_bool, <<if(value, do: 1, else: 0)::8>>}
  end

  # Calculate padding needed to reach alignment
  defp padding_to_alignment(offset, alignment) do
    remainder = rem(offset, alignment)
    if remainder == 0, do: 0, else: alignment - remainder
  end

  # Calculate tensor data size based on quantization
  defp tensor_data_size(tensor, :f32) do
    Nx.size(tensor) * 4
  end

  defp tensor_data_size(tensor, :f16) do
    Nx.size(tensor) * 2
  end

  defp tensor_data_size(tensor, :q8_0) do
    # Q8_0: each block of 32 values = 2 bytes (f16 scale) + 32 bytes (int8s) = 34 bytes
    total = Nx.size(tensor)
    num_blocks = div(total + 31, 32)
    num_blocks * 34
  end
end
