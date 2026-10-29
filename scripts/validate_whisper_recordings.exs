# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
# Optional arguments are local audio files. Without arguments, use three public clips.
defmodule WhisperRecordingAssets do
  def fetch!(directory, name, url, sha256) do
    path = Path.join(directory, name)

    unless File.exists?(path) do
      response = Req.get!(url, decode_body: false)
      unless response.status == 200, do: raise("Download failed: #{url}")
      File.write!(path, response.body)
    end

    actual = :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)
    unless actual == sha256, do: raise("Checksum mismatch: #{path}")
    path
  end
end

directory = System.get_env("WHISPER_RECORDINGS_DIR", "/tmp/whisper-recorded-speech")
File.mkdir_p!(directory)

upstream =
  "https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227"

dataset =
  "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/8d141c84e3f84c54cd7bbaa851d24edd0f559734"

filters =
  WhisperRecordingAssets.fetch!(
    directory,
    "mel_filters.npz",
    upstream <> "/whisper/assets/mel_filters.npz",
    "7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c"
  )

recordings =
  case System.argv() do
    [] ->
      for {name, url, checksum} <- [
            {"jfk.flac", upstream <> "/tests/jfk.flac",
             "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"},
            {"asr-1.flac", dataset <> "/1.flac",
             "30885601173f96b0d8ddd020dc959b055c6c1582b85a33e3fcab8c4b08ed94c2"},
            {"asr-2.flac", dataset <> "/2.flac",
             "3fc09ec6d4cc496c530b2019b17bd8fc8ef8a43d6697090971dd1d52cc3d4d89"}
          ] do
        WhisperRecordingAssets.fetch!(directory, name, url, checksum)
      end

    paths ->
      paths
  end

torch =
  System.get_env("LIBTORCH", Path.expand(".devenv/state/venv/lib/python3.12/site-packages/torch"))

library_path = Path.join(torch, "lib") <> ":" <> System.get_env("LD_LIBRARY_PATH", "")
binary = Path.expand("tools/whisper_reference/target/debug/whisper-audio")

fixtures =
  for {path, index} <- Enum.with_index(recordings, 1) do
    output = Path.join(directory, "recording-#{index}.safetensors")

    {message, status} =
      System.cmd(binary, [Path.expand(path), filters, output],
        env: [{"LD_LIBRARY_PATH", library_path}],
        stderr_to_stdout: true
      )

    if status != 0, do: raise("Audio conversion failed: #{message}")
    IO.puts("recording-#{index}: #{Path.basename(path)} #{message}")
    output
  end

System.argv(fixtures)
Code.require_file("validate_whisper_inputs.exs", __DIR__)
