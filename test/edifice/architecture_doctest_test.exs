defmodule Edifice.ArchitectureDoctestTest do
  use ExUnit.Case, async: true

  # Representative architectures (one per major family)
  doctest Edifice.Feedforward.MLP
  doctest Edifice.Recurrent
  doctest Edifice.SSM.Mamba
  doctest Edifice.Generative.GAN
  doctest Edifice.Vision.ViT
end
