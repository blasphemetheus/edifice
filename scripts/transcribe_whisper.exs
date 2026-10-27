# AI-generated tool; attribution and poem: tools/whisper_reference/README.md.
# Usage: mix run scripts/transcribe_whisper.exs AUDIO [--validate]
Code.require_file("support/whisper_audio_validation.exs", __DIR__)

defmodule WhisperFileTranscription do
  @revision "e37978b90ca9030d5170a5c07aadb050351a65bb"

  def fetch!(directory, name, url) do
    path = Path.join(directory, name)

    unless File.exists?(path) do
      response = Req.get!(url, decode_body: false)
      unless response.status == 200, do: raise("Download failed: #{url}")
      File.write!(path, response.body)
    end

    path
  end

  def run(audio, validate) do
    Nx.default_backend({EXLA.Backend, client: :host})
    Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)
    directory = Path.join(System.tmp_dir!(), "edifice-whisper-transcribe")
    File.mkdir_p!(directory)
    work = Path.join(directory, "run-#{System.pid()}")
    File.mkdir_p!(work)
    base = "https://huggingface.co/openai/whisper-base/resolve/#{@revision}"

    config =
      fetch!(directory, "generation_config.json", base <> "/generation_config.json")
      |> File.read!()
      |> Jason.decode!()

    tokenizer_path = fetch!(directory, "tokenizer.json", base <> "/tokenizer.json")
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_file(tokenizer_path)

    filters =
      fetch!(
        directory,
        "mel_filters.npz",
        "https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/whisper/assets/mel_filters.npz"
      )

    torch =
      System.get_env(
        "LIBTORCH",
        Path.expand(".devenv/state/venv/lib/python3.12/site-packages/torch")
      )

    env = [
      {"LD_LIBRARY_PATH", Path.join(torch, "lib") <> ":" <> System.get_env("LD_LIBRARY_PATH", "")}
    ]

    mel_path = Path.join(work, "input.safetensors")
    command!("whisper-audio", [Path.expand(audio), filters, mel_path], env)
    fixture = Safetensors.read!(mel_path)
    input = fixture["mel_input"]

    if validate do
      independent = WhisperAudioValidation.features(fixture["audio"], fixture["mel_filters"])
      error = Nx.abs(Nx.subtract(independent, input))
      IO.puts("Audio features Nx/LibTorch max_abs=#{Nx.to_number(Nx.reduce_max(error))}")

      unless Nx.to_number(Nx.all_close(independent, input, atol: 1.0e-4, rtol: 1.0e-5)) == 1,
        do: raise("Audio feature implementations disagree")
    end

    {encoder, decoder, loaded} =
      Edifice.Pretrained.from_hub("openai/whisper-base", revision: @revision)

    {_, encode} = Axon.build(encoder, mode: :inference)
    encoded = encode.(loaded, %{"mel_spectrogram" => input})

    prompt = [
      config["decoder_start_token_id"],
      config["lang_to_id"]["<|en|>"],
      config["task_to_id"]["transcribe"],
      config["no_timestamps_token_id"]
    ]

    # No-timestamp English transcription; suppress timestamp and non-speech tokens.
    suppressed =
      Enum.uniq(
        config["suppress_tokens"] ++ Enum.to_list((config["no_timestamps_token_id"] + 1)..51864)
      )

    begin = config["begin_suppress_tokens"]
    {_, predict} = Axon.build(decoder, mode: :inference)

    {tokens, logits} =
      decode(predict, loaded, encoded, prompt, suppressed, begin, config["eos_token_id"])

    {:ok, text} = Tokenizers.Tokenizer.decode(tokenizer, tokens)
    IO.puts("Transcript: #{String.trim(text)}")

    if List.last(tokens) != config["eos_token_id"],
      do: raise("Reached the 128-token limit before end of transcript")

    if validate do
      checkpoint = Path.expand("~/.cache/edifice/openai/whisper-base/model.safetensors")
      reference_encoder = Path.join(work, "reference_encoder.safetensors")

      command!(
        "whisper-reference",
        [checkpoint, mel_path, reference_encoder, "sdpa", "final-only"],
        env
      )

      config_path = Path.join(work, "decode_config.safetensors")

      Safetensors.write!(config_path, %{
        "prompt" => Nx.tensor(prompt, type: :s64),
        "suppress" => Nx.tensor(suppressed, type: :s64),
        "begin_suppress" => Nx.tensor(begin, type: :s64)
      })

      reference_path = Path.join(work, "reference_decoder.safetensors")

      command!(
        "whisper-decode",
        [checkpoint, reference_encoder, config_path, reference_path],
        env
      )

      reference = Safetensors.read!(reference_path)
      expected_tokens = Nx.to_flat_list(reference["tokens"])
      {:ok, expected_text} = Tokenizers.Tokenizer.decode(tokenizer, expected_tokens)
      IO.puts("Reference: #{String.trim(expected_text)}")
      unless tokens == expected_tokens, do: raise("Edifice and LibTorch token sequences differ")
      actual = Nx.stack(logits)
      error = Nx.abs(Nx.subtract(actual, reference["logits"]))

      IO.puts(
        "Decoder logits: max_abs=#{Nx.to_number(Nx.reduce_max(error))} mean_abs=#{Nx.to_number(Nx.mean(error))}"
      )

      unless Nx.to_number(Nx.all_close(actual, reference["logits"], atol: 1.0e-3, rtol: 1.0e-4)) ==
               1,
             do: raise("Decoder logits differ beyond tolerance")

      IO.puts("PASS: complete token sequence and decoder logits match the independent reference")
    end
  end

  defp command!(binary, args, env) do
    {output, status} =
      System.cmd(Path.expand("tools/whisper_reference/target/debug/#{binary}"), args,
        env: env,
        stderr_to_stdout: true
      )

    if status != 0, do: raise("#{binary} failed: #{output}")
    IO.write(output)
  end

  defp decode(predict, loaded, encoded, prompt, suppressed, begin, eos) do
    Enum.reduce_while(length(prompt)..127, {prompt, []}, fn position, {tokens, history} ->
      # Fixed-size input avoids compiling a new graph for each prefix length.
      padded = tokens ++ List.duplicate(eos, 128 - length(tokens))

      output =
        predict.(loaded, %{
          "token_ids" => Nx.tensor([padded], type: :s64),
          "encoder_output" => encoded
        })

      logits = output |> Nx.slice([0, position - 1, 0], [1, 1, 51865]) |> Nx.reshape({51865})
      excluded = if position == length(prompt), do: suppressed ++ begin, else: suppressed

      mask =
        Nx.broadcast(0, {51865})
        |> Nx.indexed_put(
          Nx.tensor(Enum.map(excluded, &[&1])),
          Nx.broadcast(1, {length(excluded)})
        )

      next = Nx.select(mask, Nx.Constants.neg_infinity(), logits) |> Nx.argmax() |> Nx.to_number()
      result = {tokens ++ [next], history ++ [logits]}
      if next == eos, do: {:halt, result}, else: {:cont, result}
    end)
  end
end

case System.argv() do
  [audio] -> WhisperFileTranscription.run(audio, false)
  [audio, "--validate"] -> WhisperFileTranscription.run(audio, true)
  _ -> raise("Usage: mix run scripts/transcribe_whisper.exs AUDIO [--validate]")
end
