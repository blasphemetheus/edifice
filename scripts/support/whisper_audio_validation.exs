# AI-generated validation; attribution and poem: tools/whisper_reference/README.md.
defmodule WhisperAudioValidation do
  import Nx.Defn
  @pi :math.pi()
  @log10 :math.log(10)

  # Independent Nx implementation checks the Rust/LibTorch frontend from the
  # decoded PCM samples and published filter bank, before encoder inference.
  defn features(audio, filters) do
    audio = Nx.pad(audio, 0, [{0, 480_000 - Nx.axis_size(audio, 0), 0}])

    positions =
      Nx.add(Nx.multiply(Nx.iota({3000, 400}, axis: 0), 160), Nx.iota({3000, 400}, axis: 1)) - 200

    positions = Nx.select(positions < 0, -positions, positions)
    positions = Nx.select(positions >= 480_000, 959_998 - positions, positions)
    window = 0.5 - 0.5 * Nx.cos(2 * @pi * Nx.iota({400}, type: :f32) / 400)
    frames = Nx.take(audio, positions) * window
    power = Nx.abs(Nx.fft(frames)) ** 2
    power = Nx.slice(power, [0, 0], [3000, 201]) |> Nx.transpose()
    log = Nx.log(Nx.max(Nx.dot(filters, power), 1.0e-10)) / @log10
    Nx.new_axis((Nx.max(log, Nx.reduce_max(log) - 8) + 4) / 4, 0)
  end
end
