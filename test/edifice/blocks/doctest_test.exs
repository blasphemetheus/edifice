defmodule Edifice.Blocks.DoctestTest do
  use ExUnit.Case, async: true

  # Shared blocks
  doctest Edifice.Blocks.CausalMask
  doctest Edifice.Blocks.CrossAttention
  doctest Edifice.Blocks.FFN
  doctest Edifice.Blocks.RoPE
  doctest Edifice.Blocks.SDPA
  doctest Edifice.Blocks.SinusoidalPE
  doctest Edifice.Blocks.TransformerBlock
end
