# diffusion — samplers + multimodal engines

The dependency-free diffusion core shared by the standalone audio, image, and
video engines: the exact noise-schedule and sampler algebra, classifier-free
guidance, and a generic latent-diffusion loop. The `mmengine`, `audio`, `image`,
and `video` packages compose this with the `compute` engine and `libpaccel`.

## Schedules (`schedule.go`)

- **scaled_linear betas**: `betas = linspace(√βstart, √βend, T)²`.
- **alphas_cumprod**: `∏(1 − βᵢ)`; train sigmas `σ = √((1−ᾱ)/ᾱ)`.
- **Karras sigmas** (ρ=7): `σᵢ = (σmax^{1/ρ} + t(σmin^{1/ρ} − σmax^{1/ρ}))^ρ`.

## Samplers

**Euler / Karras, v-prediction** (`euler.go`) — SVD, video, audio:

```
scale_input(x) = x / √(σ²+1)
x0   = v·(−σ/√(σ²+1)) + x/(σ²+1)          # predicted clean sample
x'   = x + ((x − x0)/σ)·(σ_next − σ)       # Euler step; final step (σ_next=0) → x0
```

**LCM, epsilon-prediction** (`lcm.go`) — few-step text-to-image:

```
x0      = (x_t − √(1−ᾱ_t)·ε) / √(ᾱ_t)
x_{prev} = √(ᾱ_prev)·x0                    # jump to next selected timestep; final → x0
```

Both recover `x0` exactly given the correct model output (verified in
`diffusion_test.go`).

## Classifier-free guidance (`pipeline.go`)

```
out = uncond + scale·(cond − uncond)
```

## Pipeline

`Pipeline.Generate(latent, cond, uncond)` runs the loop: scale the latent by the
init sigma, then per step precondition → backbone (twice for CFG) → scheduler
step, and finally decode. The `Backbone` and `Decoder` interfaces are the seams
where the production DiT/UNet and VAE (loaded from a `.paccel`) plug in.

## Engines

- **audio/** — Stable Audio 3 style (T5Gemma → DiT → VAE), Euler v-prediction.
- **image/** — few-step text-to-image (CLIP/T5 → UNet → VAE), LCM.
- **video/** — CogVideoX style (T5 → 3D DiT → 3D VAE), Euler v-prediction.

Each ships a reference backbone/decoder (`mmengine`) on the compute engine so it
runs end-to-end today; the heavy production modules are drop-in via the same
interfaces.
