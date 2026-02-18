defmodule Edifice.Neuromorphic.ANN2SNN do
  @moduledoc """
  ANN-to-SNN Conversion via Rate Coding.

  Provides a bridge between conventional artificial neural networks (ANNs)
  and spiking neural networks (SNNs). The ANN is trained with standard
  backpropagation, then converted to an SNN by replacing ReLU activations
  with integrate-and-fire neurons that encode activation magnitudes as
  spike rates.

  ## Conversion Principle

  A ReLU neuron with activation `a` can be approximated by a spiking neuron
  that fires at rate `a / threshold` over `num_timesteps`. The key insight
  is that the time-averaged spike rate of an IF neuron converges to the
  ReLU activation as timesteps increase.

  ## Architecture

  ```
  ANN Mode:                          SNN Mode:
  Input [batch, input_size]          Input [batch, input_size]
        |                                  |
        v                                  v
  Dense + ReLU                       Dense (same weights)
        |                                  |
        v                                  v
  Dense + ReLU                       IF Neuron (rate-coded)
        |                                  |  (simulate num_timesteps)
        v                                  v
  Output [batch, output_size]        Rate Decode -> Output
  ```

  ## Usage

      # Build the ANN (for training)
      ann = ANN2SNN.build(
        input_size: 256,
        hidden_sizes: [128, 64],
        num_timesteps: 10,
        threshold: 1.0
      )

      # Build the SNN (for inference on neuromorphic hardware)
      snn = ANN2SNN.build_snn(
        input_size: 256,
        hidden_sizes: [128, 64],
        num_timesteps: 10,
        threshold: 1.0
      )

  ## References

  - Diehl et al., "Fast-Classifying, High-Accuracy Spiking Deep Networks
    Through Weight and Threshold Balancing" (IJCNN 2015)
  - Rueckauer et al., "Conversion of Continuous-Valued Deep Networks to
    Efficient Event-Driven Networks for Image Classification" (2017)
  """

  require Axon
  import Nx.Defn

  @default_hidden_sizes [256, 128]
  @default_num_timesteps 10
  @default_threshold 1.0

  @doc """
  Build the ANN version (for training with backpropagation).

  This is a standard feedforward network with ReLU activations. After
  training, the same weights can be used with `build_snn/1` for
  spiking inference.

  ## Options

  - `:input_size` - Input feature dimension (required)
  - `:hidden_sizes` - List of hidden layer sizes (default: [256, 128])
  - `:output_size` - Output dimension (default: last hidden size)
  - `:num_timesteps` - Stored for SNN conversion reference (default: 10)
  - `:threshold` - Spiking threshold for conversion (default: 1.0)

  ## Returns

  An Axon model (standard ANN): `[batch, input_size]` -> `[batch, output_size]`
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:hidden_sizes, [pos_integer()]}
          | {:input_size, pos_integer()}
          | {:output_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    output_size = Keyword.get(opts, :output_size, List.last(hidden_sizes))

    input = Axon.input("input", shape: {nil, input_size})

    # Build standard ANN with ReLU activations
    # Use the same layer names as the SNN for weight sharing
    x =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "dense_#{idx}")
        |> Axon.layer_norm(name: "bn_#{idx}")
        |> Axon.relu()
      end)

    # Output layer (no activation for raw logits)
    if output_size != List.last(hidden_sizes) do
      Axon.dense(x, output_size, name: "output")
    else
      x
    end
  end

  @doc """
  Build the SNN version (for spiking inference).

  Uses the same dense layer structure but replaces ReLU with
  integrate-and-fire neuron simulation. Weights from a trained ANN
  can be directly transferred.

  ## Options

  Same as `build/1`.

  ## Returns

  An Axon model (SNN via rate coding): `[batch, input_size]` -> `[batch, output_size]`
  """
  @spec build_snn(keyword()) :: Axon.t()
  def build_snn(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    output_size = Keyword.get(opts, :output_size, List.last(hidden_sizes))
    num_timesteps = Keyword.get(opts, :num_timesteps, @default_num_timesteps)
    threshold = Keyword.get(opts, :threshold, @default_threshold)

    input = Axon.input("input", shape: {nil, input_size})

    # Same dense layers as ANN (same names for weight sharing)
    # but with IF neuron simulation instead of ReLU
    x =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        projected =
          acc
          |> Axon.dense(size, name: "dense_#{idx}")
          |> Axon.layer_norm(name: "bn_#{idx}")

        # Replace ReLU with IF neuron simulation
        Axon.layer(
          &if_neuron_simulate/2,
          [projected],
          name: "if_neuron_#{idx}",
          num_timesteps: num_timesteps,
          threshold: threshold,
          op_name: :if_neuron
        )
      end)

    if output_size != List.last(hidden_sizes) do
      Axon.dense(x, output_size, name: "output")
    else
      x
    end
  end

  @doc """
  Integrate-and-Fire neuron simulation for rate-coded SNN inference.

  Simulates an IF neuron over multiple timesteps. The input current is
  presented at each timestep, the membrane integrates, and spikes are
  emitted when threshold is exceeded. The output is the average spike
  rate, which approximates the ReLU activation.

  ## Parameters

  - `membrane` - Initial membrane potential (zero)
  - `input_current` - Weighted input current
  - `num_timesteps` - Number of simulation steps
  - `threshold` - Firing threshold

  ## Returns

  Average spike rate (approximates ReLU output).
  """
  @spec if_neuron(Nx.Tensor.t(), Nx.Tensor.t(), pos_integer(), float()) :: Nx.Tensor.t()
  defn if_neuron(membrane, input_current, num_timesteps, threshold) do
    # Simulate IF neuron for num_timesteps
    {_final_membrane, spike_count} =
      while {membrane, spike_count = Nx.broadcast(0.0, Nx.shape(membrane))},
            _i <- 1..num_timesteps do
        new_membrane = membrane + input_current
        spikes = Nx.greater(new_membrane, threshold) |> Nx.as_type(:f32)
        reset_membrane = new_membrane - spikes * threshold
        {reset_membrane, spike_count + spikes}
      end

    # Rate decode: average spikes over time
    spike_count / num_timesteps
  end

  @doc """
  Get the output size of an ANN2SNN model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    Keyword.get(opts, :output_size, List.last(hidden_sizes))
  end

  # IF neuron simulation in a custom Axon layer
  defp if_neuron_simulate(input_current, opts) do
    num_timesteps = opts[:num_timesteps] || @default_num_timesteps
    threshold = opts[:threshold] || @default_threshold

    batch_size = Nx.axis_size(input_current, 0)
    hidden_size = Nx.axis_size(input_current, 1)

    initial_membrane = Nx.broadcast(Nx.tensor(0.0), {batch_size, hidden_size})

    # Simulate over timesteps
    {_final_membrane, spike_sum} =
      Enum.reduce(
        1..num_timesteps,
        {initial_membrane, Nx.broadcast(Nx.tensor(0.0), {batch_size, hidden_size})},
        fn _t, {membrane, acc_spikes} ->
          new_membrane = Nx.add(membrane, input_current)
          spikes = Nx.greater(new_membrane, threshold) |> Nx.as_type(:f32)
          reset_membrane = Nx.subtract(new_membrane, Nx.multiply(spikes, threshold))
          {reset_membrane, Nx.add(acc_spikes, spikes)}
        end
      )

    # Rate decode
    Nx.divide(spike_sum, num_timesteps)
  end
end
