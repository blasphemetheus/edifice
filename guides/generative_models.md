# Generative Models
> From latent variable models through adversarial training to score-based diffusion and flow matching.

## Overview

Generative models learn to produce new samples that resemble a training distribution. Where
discriminative models learn p(y|x) (label given input), generative models learn p(x) itself --
the full data distribution -- and can draw new samples from it. This capability is fundamental to
tasks like data augmentation, planning through imagination, action generation for control, and
creative synthesis.

Edifice implements 11 generative modules spanning three major paradigms. The first paradigm uses
latent variables: VAE encodes data into a smooth continuous latent space via the reparameterization
trick, while VQ-VAE quantizes to a discrete codebook for sharper reconstructions. The second
paradigm is adversarial: GAN pits a generator against a discriminator in a minimax game, producing
sharp samples but suffering from training instability and mode collapse. The third paradigm --
score-based and flow-based methods -- has become dominant. It includes DDPM (learning to denoise),
DDIM (deterministic fast sampling), DiT (transformer backbone for diffusion), LatentDiffusion
(diffusion in compressed space), ConsistencyModel (single-step generation), ScoreSDE (continuous
SDE framework), FlowMatching (ODE-based optimal transport), and NormalizingFlow (invertible
transforms with exact likelihood).

All generative modules in Edifice return tuples of models (encoder/decoder, generator/discriminator,
or denoiser networks) rather than single models, reflecting their multi-component training procedures.
They are built on Nx/Axon and provide both the model architecture and the associated loss functions
needed for training.

## Conceptual Foundation

The three paradigms differ in how they bridge noise and data:

```
                    Generative Model Paradigms
                    ==========================

Latent Variable (VAE / VQ-VAE):
  Learn an encoder q(z|x) and decoder p(x|z).
  Optimize the Evidence Lower Bound (ELBO):
    log p(x) >= E_q[log p(x|z)] - KL(q(z|x) || p(z))
              = reconstruction     - regularization

Adversarial (GAN):
  Generator G maps noise to data, discriminator D classifies real/fake.
  Minimax game:
    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]

Score / Diffusion / Flow:
  Learn to reverse a noise-adding process.
  Forward:  x_0 -> x_1 -> ... -> x_T ~ N(0, I)     (add noise)
  Reverse:  x_T -> ... -> x_1 -> x_0 ~ p(data)     (remove noise)
  Train a network to predict the noise (or score, or velocity).
```

The ELBO derivation for VAEs reveals a fundamental tradeoff. The reconstruction term
E_q[log p(x|z)] wants the decoder to perfectly reconstruct inputs, pushing the encoder to
be maximally informative. The KL term KL(q(z|x) || p(z)) wants the encoder's posterior to
match a simple prior (usually standard normal), pushing the latent space to be smooth and
regular. The beta parameter in beta-VAE controls this tradeoff.

For diffusion models, the forward process gradually adds Gaussian noise according to a
schedule:

```
Forward process (fixed):
  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

  where alpha_bar_t = product(1 - beta_i) for i = 1..t
  and beta_1 < beta_2 < ... < beta_T is the noise schedule

Reverse process (learned):
  p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 I)

Training objective:
  L = E_{t, x_0, eps}[ ||eps - eps_theta(x_t, t)||^2 ]
  (predict the noise that was added)
```

## Architecture Evolution

```
2013            2017            2020            2021            2023
 |               |               |               |               |
 VAE             WGAN            DDPM            Score SDE       FlowMatching
 (Kingma)       (Arjovsky)      (Ho et al.)     (Song et al.)   (Lipman)
 |               |               |               |               |
 |               |              DDIM             |              ConsistencyModel
 |               |              (fast sample)    |              (single step)
 |               |               |               |
VQ-VAE           |              DiT             NormalizingFlow
(van den Oord)   |              (transformer)   (RealNVP-style)
                 |               |
                 |              LatentDiffusion
                 |              (LDM / Stable Diffusion)

  Latent Variable     Adversarial     Score / Diffusion / Flow
  ================    ============    ========================
  VAE, VQ-VAE         GAN             Diffusion, DDIM, DiT,
                                      LatentDiffusion,
                                      ConsistencyModel,
                                      ScoreSDE, FlowMatching,
                                      NormalizingFlow

  Key transitions:
  VAE -> VQ-VAE:        Discrete codebook avoids posterior collapse, sharper outputs
  DDPM -> DDIM:         Same training, deterministic sampling in ~50 steps (vs ~1000)
  DDPM -> DiT:          Replace U-Net with Transformer + AdaLN-Zero conditioning
  DDPM -> LatentDiff:   Compress first (VAE), then diffuse in latent space
  DDPM -> Consistency:  Learn to map any noise level directly to x_0 (single step)
  Score SDE -> Flow:    Replace SDE with simpler ODE, linear interpolation paths
```

## When to Use What

```
+--------------------+------------------+------------------+-------------------+
| Requirement        | Best Module      | Runner-up        | Why               |
+--------------------+------------------+------------------+-------------------+
| Fast structured    | VAE              | VQ-VAE           | Single forward    |
| latent space       |                  |                  | pass, smooth z    |
+--------------------+------------------+------------------+-------------------+
| Discrete tokens /  | VQ-VAE           | --               | Codebook gives    |
| sharp recon.       |                  |                  | discrete codes    |
+--------------------+------------------+------------------+-------------------+
| Sharpest samples   | GAN              | Diffusion        | Adversarial loss  |
| (small scale)      |                  |                  | avoids blurriness |
+--------------------+------------------+------------------+-------------------+
| Best overall       | Diffusion (DDPM) | DiT              | Stable training,  |
| sample quality     |                  |                  | covers all modes  |
+--------------------+------------------+------------------+-------------------+
| Fast diffusion     | DDIM             | ConsistencyModel | Deterministic     |
| inference          |                  |                  | skip-step sampling|
+--------------------+------------------+------------------+-------------------+
| Single-step        | ConsistencyModel | --               | Maps any noise    |
| generation         |                  |                  | level to x_0      |
+--------------------+------------------+------------------+-------------------+
| Scalable diffusion | DiT              | LatentDiffusion  | Transformer       |
| (large models)     |                  |                  | scales better     |
+--------------------+------------------+------------------+-------------------+
| High-dim data      | LatentDiffusion  | VAE + Diffusion  | Compress first,   |
| efficiency         |                  |                  | diffuse in latent |
+--------------------+------------------+------------------+-------------------+
| Simplest training  | FlowMatching     | Diffusion        | No noise schedule,|
| (modern default)   |                  |                  | linear paths      |
+--------------------+------------------+------------------+-------------------+
| Exact likelihood   | NormalizingFlow  | VAE (ELBO bound) | Invertible =      |
|                    |                  |                  | exact log p(x)    |
+--------------------+------------------+------------------+-------------------+
| Theoretical        | ScoreSDE         | FlowMatching     | Unifying framework|
| framework          |                  |                  | for all diffusion |
+--------------------+------------------+------------------+-------------------+
```

## Key Concepts

### The Reparameterization Trick (VAE)

The VAE encoder outputs parameters of a distribution (mu and log_var) rather than a point
estimate. To generate a latent code, we sample z ~ N(mu, exp(log_var)). The problem: sampling
is not differentiable, so gradients cannot flow through this operation during backpropagation.

The reparameterization trick solves this by rewriting the sampling as a deterministic
function of the parameters plus external noise:

```
z = mu + exp(0.5 * log_var) * eps,     eps ~ N(0, I)

This is equivalent to z ~ N(mu, exp(log_var)), but now:
- eps is sampled independently (no gradient needed)
- z is a deterministic, differentiable function of mu and log_var
- Gradients flow through mu and log_var to the encoder
```

VQ-VAE replaces this continuous sampling with discrete codebook lookup. The encoder output
z_e is quantized to the nearest codebook vector: z_q = codebook[argmin_k ||z_e - e_k||].
Since argmin is not differentiable, VQ-VAE uses the straight-through estimator: gradients
from the decoder pass directly to the encoder as if quantization had not occurred.

### The Diffusion Framework

Diffusion models define a forward process that gradually destroys structure by adding noise,
then learn to reverse this process. The key parameters are the noise schedule
(beta_1, ..., beta_T) and the cumulative product alpha_bar_t:

```
Forward process (no learning required):
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

  As t increases:
  - sqrt(alpha_bar_t) shrinks toward 0  (signal fades)
  - sqrt(1 - alpha_bar_t) grows toward 1 (noise dominates)
  - At t = T: x_T is approximately pure Gaussian noise

Reverse process (the model):
  Given x_t, predict the noise eps that was added, then subtract it:
  x_{t-1} = (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta(x_t, t))
             / sqrt(1 - beta_t) + sigma_t * z

Training is just MSE on the noise prediction:
  L = ||eps - eps_theta(sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps, t)||^2
```

DDIM reformulates the reverse process to be deterministic. By setting the noise coefficient
(controlled by eta) to zero, each reverse step becomes a deterministic function of x_t.
This means DDIM can skip steps -- instead of 1000 sequential denoising steps, it can take
every 20th timestep and still produce valid samples in about 50 steps.

### AdaLN-Zero Conditioning (DiT)

DiT replaces the U-Net backbone traditionally used in diffusion models with a Transformer.
The challenge is conditioning the transformer on the diffusion timestep and optional class
labels. Rather than expensive cross-attention, DiT uses Adaptive Layer Normalization:

```
Standard LayerNorm:
  y = gamma * normalize(x) + beta     (gamma, beta are learned constants)

AdaLN-Zero:
  [gamma, beta, alpha] = MLP(cond)     (modulation from condition vector)
  y = gamma * normalize(x) + beta     (condition-dependent normalization)
  y = alpha * y                        (scaling, initialized to zero)

Condition vector = MLP(sinusoidal_embed(t) + class_embed(c))
```

Initializing alpha to zero means each DiT block starts as the identity function, enabling
stable training of deep models. The conditioning signal (timestep + optional class) modulates
every normalization layer in the network.

### From Diffusion to Flow Matching

Flow Matching simplifies diffusion by replacing the stochastic noise-addition process with
a deterministic optimal transport path:

```
Diffusion (SDE path, complex schedule):
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
  Train to predict noise: ||eps - eps_theta(x_t, t)||^2
  Sample by solving reverse SDE (many steps, stochastic)

Flow Matching (ODE path, no schedule):
  x_t = (1 - t) * x_noise + t * x_data        (linear interpolation)
  velocity = x_data - x_noise                  (constant along path)
  Train to predict velocity: ||v - v_theta(x_t, t)||^2
  Sample by solving forward ODE (fewer steps, deterministic)
```

Flow matching needs no noise schedule, uses simpler math, and typically converges in fewer
sampling steps (10-20 vs 50-100 for DDIM). ScoreSDE provides the theoretical framework that
unifies both perspectives: diffusion as SDE, flow matching as the probability flow ODE of
that same SDE.

### Normalizing Flows: Exact Likelihood via Invertibility

Unlike VAEs (which optimize a lower bound on likelihood) and GANs (which have no explicit
likelihood), normalizing flows provide exact log-likelihood computation. Each coupling layer
is constructed to be trivially invertible with a tractable Jacobian determinant:

```
RealNVP Coupling Layer:
  Input:  (x1, x2)                     (split input in half)
  s, t = NN(x1)                        (neural network on first half)
  y1 = x1                              (pass through unchanged)
  y2 = x2 * exp(s) + t                 (affine transform of second half)
  Output: (y1, y2)

  Inverse: x2 = (y2 - t) * exp(-s)    (trivially invertible)
  Log-det Jacobian: sum(s)             (diagonal Jacobian from affine)

Successive layers alternate which half is transformed.
Total log-probability: log p(x) = log p(z) + sum of log-det-Jacobians
```

The cost of exact likelihood is architectural constraint: every layer must be invertible.
This limits expressiveness compared to unconstrained networks, though stacking many
coupling layers compensates.

## Complexity Comparison

```
+-------------------+-----------+-----------+----------+---------+-----------+
| Module            | Training  | Inference | Training | Sample  | Likelihood|
|                   | Steps/Ep  | Steps     | Loss     | Quality | Available?|
+-------------------+-----------+-----------+----------+---------+-----------+
| VAE               | 1         | 1         | ELBO     | Blurry  | Bound     |
| VQ-VAE            | 1         | 1         | Recon+VQ | Sharper | No        |
| GAN               | 2 (G+D)  | 1         | Adv.     | Sharp   | No        |
| Diffusion (DDPM)  | 1         | ~1000     | MSE      | Best    | Bound     |
| DDIM              | 1         | ~50       | MSE      | Best    | Bound     |
| DiT               | 1         | ~50-250   | MSE      | Best    | Bound     |
| LatentDiffusion   | 2-phase   | ~50+1     | MSE+VAE  | Best    | No        |
| ConsistencyModel  | 1-2       | 1-3       | Consist. | Good    | No        |
| ScoreSDE          | 1         | ~100-1000 | DSM      | Best    | ODE-based |
| FlowMatching      | 1         | ~10-20    | MSE      | Best    | ODE-based |
| NormalizingFlow   | 1         | 1         | NLL      | Decent  | Exact     |
+-------------------+-----------+-----------+----------+---------+-----------+

Notes:
- "Training Steps/Ep" = forward passes per training example
  (GAN needs separate G and D updates; LatentDiffusion has VAE then diffusion phases)
- "Inference Steps" = forward passes to generate one sample
- VAE/VQ-VAE: single pass, fast but limited quality
- Diffusion family: iterative, high quality but slow
- ConsistencyModel: best of both -- near-diffusion quality in 1-3 steps
- FlowMatching: modern sweet spot -- good quality in 10-20 steps
- NormalizingFlow: single pass with exact likelihood, architecturally constrained

Return types:
- VAE, VQ-VAE: {encoder, decoder}
- GAN: {generator, discriminator}
- Diffusion, DDIM, DiT, ConsistencyModel, ScoreSDE, FlowMatching: single denoiser model
- LatentDiffusion: {encoder, decoder, denoiser}
- NormalizingFlow: single invertible model (forward and inverse methods)
```

## Module Reference

- `Edifice.Generative.VAE` -- Variational Autoencoder with reparameterization trick, configurable beta for beta-VAE
- `Edifice.Generative.VQVAE` -- Vector Quantized VAE with discrete codebook, straight-through estimator, and commitment loss
- `Edifice.Generative.GAN` -- Generator/Discriminator framework with standard, WGAN-GP, and conditional variants
- `Edifice.Generative.Diffusion` -- DDPM denoising diffusion with configurable noise schedule and action-conditioned generation
- `Edifice.Generative.DDIM` -- Deterministic diffusion sampling with configurable eta (0 = deterministic, 1 = DDPM) and step skipping
- `Edifice.Generative.DiT` -- Diffusion Transformer with AdaLN-Zero conditioning on timestep and optional class labels
- `Edifice.Generative.LatentDiffusion` -- Two-phase architecture: frozen VAE encoder/decoder + diffusion in compressed latent space
- `Edifice.Generative.ConsistencyModel` -- Single-step generation via learned consistency function with skip connections
- `Edifice.Generative.ScoreSDE` -- Score-based SDE framework supporting VP-SDE (DDPM) and VE-SDE (SMLD) variants
- `Edifice.Generative.FlowMatching` -- Conditional Flow Matching with linear interpolation (optimal transport) paths and ODE sampling
- `Edifice.Generative.NormalizingFlow` -- RealNVP-style affine coupling layers with exact log-likelihood and invertible transforms

## Cross-References

- **Dynamic and Continuous Models** -- NeuralODE connects directly to FlowMatching and ScoreSDE:
  the probability flow ODE in ScoreSDE and the sampling ODE in FlowMatching are both instances
  of neural ODEs. The ODE integration methods (Euler, RK4) used for sampling are the same ones
  used in NeuralODE for continuous dynamics.
- **Building Blocks** -- DiT uses AdaptiveNorm (adaptive layer normalization) for conditioning.
  All modules use standard dense layers and activations from Axon. Diffusion and DDIM use
  sinusoidal timestep embeddings.
- **[Attention Mechanisms](attention_mechanisms.md)** -- DiT uses multi-head self-attention as
  its core processing layer. LatentDiffusion can use attention-based denoisers.
- **[State Space Models](state_space_models.md)** -- The continuous-time ODE perspective of
  SSMs (x'(t) = Ax(t) + Bu(t)) connects to the flow-based view of generative modeling where
  data evolves along learned trajectories.

## Further Reading

1. Kingma, Welling. "Auto-Encoding Variational Bayes." ICLR 2014. arXiv:1312.6114
   -- The VAE paper establishing variational inference for generative models.
2. Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
   arXiv:2006.11239 -- Revived diffusion models with simple training and high quality.
3. Song et al. "Denoising Diffusion Implicit Models." ICLR 2021. arXiv:2010.02502
   -- Deterministic sampling that made diffusion practical (50 steps vs 1000).
4. Lipman et al. "Flow Matching for Generative Modeling." ICLR 2023. arXiv:2210.02747
   -- Simplified diffusion via ODE paths with optimal transport, no noise schedule needed.
5. Song, Dhariwal et al. "Consistency Models." ICML 2023. arXiv:2303.01469
   -- Single-step generation by learning to map any noise level directly to data.
