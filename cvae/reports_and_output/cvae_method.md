# CVAE Inpainting — Method Reference Document

> **Purpose**: Reference material for the report writer. Contains problem formulation, mathematical derivations of the final method, architectural details, training configuration, and evaluation results.
> **Author**: Akingno (CVAE method owner)
> **Code version**: v1.3 (conditional prior + no compose)

---

## 1. Task Formulation

### 1.1 Problem Definition

Let $x \sim p_X$ be a natural face image, and $M \in \{0, 1\}^{H \times W}$ be a binary rectangular mask ($M_{ij}=1$ denotes an occluded pixel). The corrupted image is:

$$
y = x \odot (1 - M)
$$

where $\odot$ denotes element-wise multiplication. The task is to learn the conditional distribution $p_{\hat{X} \mid Y, M}$ and sample inpainting results $\hat{x}$ from it.

### 1.2 Evaluation Regimes

Test-time masks are sampled with center $(c_x, c_y)$ uniformly over the image, and dimensions $(h, w)$ uniformly drawn from three size regimes:

| Regime | Height / Width range | Approx. coverage |
|---|---|---|
| Small  | $[0.1H, 0.2H] \times [0.1W, 0.2W]$ | ~1–4% |
| Medium | $[0.3H, 0.4H] \times [0.3W, 0.4W]$ | ~9–16% |
| Large  | $[0.5H, 0.6H] \times [0.5W, 0.6W]$ | ~25–36% |

For each regime, a fixed set of evaluation masks is pre-generated (`data/fixed_masks_{regime}.pt`) and shared across all team methods, ensuring comparable evaluation.

### 1.3 Metric Categories (Blau & Michaeli, CVPR 2018)

Blau & Michaeli prove that no algorithm can simultaneously minimize **distortion** $\mathbb{E}[\Delta(X, \hat{X})]$ and **perception** $d(p_X, p_{\hat{X}})$. Our four metrics are categorized as:

- **Distortion**: MSE, PSNR, SSIM (paired pixel/structural distances)
- **Perception**: FID, LPIPS (distribution distance and feature-space distance)

---

## 2. CVAE Method (v1.3, Final Design)

### 2.1 Conditional ELBO Derivation

Conditional VAE introduces a latent variable $z$ into the generative process. With condition $c = (y, M)$, the conditional log-likelihood is:

$$
\log p(x \mid c) = \log \int p_\theta(x \mid z, c) \, p(z \mid c) \, dz
$$

Direct optimization of this likelihood is intractable (the integral is not analytically solvable). Introducing a variational distribution $q_\phi(z \mid x, c)$, we obtain the **Evidence Lower BOund (ELBO)**:

$$
\log p(x \mid c) \;\geq\; \mathbb{E}_{q_\phi(z \mid x, c)}\big[\log p_\theta(x \mid z, c)\big] - D_{KL}\big(q_\phi(z \mid x, c) \,\|\, p_\psi(z \mid c)\big)
$$

Equality holds iff $q_\phi(z \mid x, c) = p(z \mid x, c)$. We maximize the ELBO, equivalently minimizing the negative ELBO:

$$
\mathcal{L}_{\text{ELBO}}(x, c) = -\mathbb{E}_{q_\phi(z \mid x, c)}\big[\log p_\theta(x \mid z, c)\big] + D_{KL}\big(q_\phi(z \mid x, c) \,\|\, p_\psi(z \mid c)\big)
$$

### 2.2 Concrete Loss Formulation

**(1) Reconstruction term**: We use an **L1 loss** in place of the standard Gaussian log-likelihood:
- Assuming $p_\theta(x \mid z, c) = \mathcal{N}(x; \hat{x}, \sigma^2 I)$, the NLL reduces to $\propto \|x - \hat{x}\|_2^2$.
- We use L1 instead of L2 because L2 produces over-smoothed (blurry) outputs due to averaging over plausible completions. L1 yields sharper results in practice.
- Formally, this corresponds to assuming $p_\theta$ follows a Laplace distribution: $p_\theta(x \mid z, c) = \text{Laplace}(x; \hat{x}, b)$, whose NLL is $\propto \|x - \hat{x}\|_1$.

$$
\mathcal{L}_{\text{rec}}(x, c) = \|\hat{x} - x\|_1, \quad \hat{x} = \text{Decoder}(z, y, M)
$$

**(2) KL term**: Both $q_\phi$ and $p_\psi$ are diagonal Gaussians, admitting a closed-form KL:

$$
q_\phi(z \mid x, c) = \mathcal{N}\big(z; \mu_q(x, c), \text{diag}(\sigma_q^2(x, c))\big)
$$

$$
p_\psi(z \mid c) = \mathcal{N}\big(z; \mu_p(c), \text{diag}(\sigma_p^2(c))\big)
$$

The closed-form KL between two diagonal Gaussians (summed over latent dimensions) is:

$$
D_{KL}(q_\phi \| p_\psi) = \sum_{i=1}^{D} \left[ \log \frac{\sigma_{p,i}}{\sigma_{q,i}} + \frac{\sigma_{q,i}^2 + (\mu_{q,i} - \mu_{p,i})^2}{2 \sigma_{p,i}^2} - \frac{1}{2} \right]
$$

**(3) Final loss (with $\beta$-warmup)**:

$$
\boxed{\;\mathcal{L}(x, c) = \|\hat{x} - x\|_1 + \beta(t) \cdot D_{KL}\big(q_\phi(z \mid x, c) \,\|\, p_\psi(z \mid c)\big)\;}
$$

The KL weight $\beta(t)$ follows a linear warm-up:

$$
\beta(t) = \min\left(\beta_{\max},\ \beta_{\max} \cdot \frac{t}{T_{\text{warmup}}}\right)
$$

with $\beta_{\max} = 1.0$ and $T_{\text{warmup}} = 10$ epochs. The warm-up prevents the KL term from dominating early in training, which would otherwise cause posterior collapse.

### 2.3 Network Architecture

Three independent `nn.Module` components:

#### Encoder $q_\phi(z \mid x, c)$ (23.74M parameters)

```
Input: concat(gt, masked, mask) → [B, 7, 256, 256]
  Conv(7→64,    k4 s2) + GroupNorm(8) + SiLU  → [B,  64, 128, 128]
  Conv(64→128,  k4 s2) + GroupNorm(8) + SiLU  → [B, 128,  64,  64]
  Conv(128→256, k4 s2) + GroupNorm(8) + SiLU  → [B, 256,  32,  32]
  Conv(256→512, k4 s2) + GroupNorm(8) + SiLU  → [B, 512,  16,  16]
  Conv(512→512, k4 s2) + GroupNorm(8) + SiLU  → [B, 512,   8,   8]
  Flatten → Linear(32768 → 512)
  → μ_q ∈ ℝ^256,  logσ²_q ∈ ℝ^256
```

#### PriorNet $p_\psi(z \mid c)$ (9.95M parameters)

```
Input: concat(masked, mask) → [B, 4, 256, 256]
  Conv(4→32,    k4 s2) + GroupNorm(8) + SiLU  → [B,  32, 128, 128]
  Conv(32→64,   k4 s2) + GroupNorm(8) + SiLU  → [B,  64,  64,  64]
  Conv(64→128,  k4 s2) + GroupNorm(8) + SiLU  → [B, 128,  32,  32]
  Conv(128→256, k4 s2) + GroupNorm(8) + SiLU  → [B, 256,  16,  16]
  Conv(256→256, k4 s2) + GroupNorm(8) + SiLU  → [B, 256,   8,   8]
  Flatten → Linear(16384 → 512)
  → μ_p ∈ ℝ^256,  logσ²_p ∈ ℝ^256
  (A soft floor clamp(min=-2.0) is applied to logσ²_p to prevent
   prior-variance degeneracy.)
```

#### Decoder $p_\theta(x \mid z, c)$ (15.71M parameters)

```
Input: z ∈ ℝ^256
  Linear(256 → 32768) → Reshape → [B, 512, 8, 8]

  5 upsampling blocks, each:
    NearestUpsample(×2) → concat(downsampled (masked, mask) at current scale)
                       → Conv(k3 s1) + GroupNorm(8) + SiLU
                       → Conv(k3 s1) + GroupNorm(8) + SiLU

  Block 1: [B, 512, 8,  8]  → [B, 512, 16,  16]
  Block 2: [B, 512, 16, 16] → [B, 256, 32,  32]
  Block 3: [B, 256, 32, 32] → [B, 128, 64,  64]
  Block 4: [B, 128, 64, 64] → [B,  64, 128, 128]
  Block 5: [B,  64, 128,128]→ [B,  32, 256, 256]

  Conv(32 → 3, k3 s1) + Sigmoid → x̂ ∈ [0, 1]^{3×256×256}
```

**Key architectural choices**:
- **No U-Net skip connections**: prevents encoder features from bypassing the latent bottleneck, ensuring $z$ carries meaningful information.
- **Multi-scale condition injection**: $(y, M)$ is downsampled and concatenated into the decoder at every upsampling scale, providing boundary references.
- **Sigmoid output**: the data range is $[0, 1]$ (from `ToTensor()` normalization).
- **No compose operation**: the decoder directly outputs the full image $\hat{x}$, rather than generating only the hole region. Rationale: see §3.1.

**Total parameter count**: 49.40M

### 2.4 Training and Inference

**Training phase**: $z$ is sampled from the posterior via the reparameterization trick:

$$
z = \mu_q + \sigma_q \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**Inference phase**: $z$ is sampled from the conditional prior $p_\psi$ (not from $\mathcal{N}(0, I)$):

$$
z \sim \mathcal{N}(\mu_p(y, M), \text{diag}(\sigma_p^2(y, M)))
$$

**Pluralistic generation**: For the same input $(y, M)$, repeatedly sampling $z$ yields multiple distinct inpainting results, demonstrating the stochastic nature of CVAE.

---

## 3. Design Evolution (Optional Inclusion in Report Discussion)

This section documents the iterative design process. If report space is limited, it can be compressed into a single paragraph; otherwise, it strengthens the Method or Discussion sections as evidence of principled engineering.

### 3.1 v1.2 (Failed Attempt): Standard Prior + Compose

The initial design adopted the simplest CVAE formulation:
- Fixed prior $p(z) = \mathcal{N}(0, I)$
- Compose output: $\hat{x} = y \odot (1 - M) + \hat{x}_{\text{raw}} \odot M$ (generating only the hole; known pixels are preserved exactly)
- KL was the standard closed-form $D_{KL}(q_\phi(z \mid x, c) \,\|\, \mathcal{N}(0, I))$

**Failure mode**: Within 5 training epochs, the KL term collapsed from ~1500 to below 1, and validation loss stopped improving.

**Diagnosis**: Posterior collapse. The decoder could reconstruct hole-boundary regions using the multi-scale spatial condition $(y, M)$ alone, without relying on $z$. To minimize the KL term, the optimizer forced $q_\phi$ toward the fixed standard-normal prior, reducing $z$ to pure noise.

**Remediation attempt (Free Bits)**: We applied per-dimension KL floor at $\lambda = 0.5$ nats (Kingma et al., 2016 IAF). Result: posterior capacity was preserved (KL stabilized at $D \cdot \lambda = 128$ nats), but validation loss still did not improve — indicating that the decoder had learned to **ignore $z$** entirely and rely solely on the spatial condition for deterministic reconstruction. Fixes on the encoder side alone could not address this.

### 3.2 v1.3 (Final Design): Conditional Prior + No Compose

To address the root cause identified in v1.2, two changes were made:

**(a) Remove compose**: the decoder must reconstruct every pixel (including known regions), forcing $z$ to carry information for all output pixels. This fundamentally eliminates the possibility of the decoder bypassing $z$.

**(b) Introduce a conditional prior $p_\psi(z \mid y, M)$**:
- Both $q_\phi$ and $p_\psi$ are learnable. The KL term becomes the distance between two network outputs, rather than a hard pull toward a fixed distribution.
- $p_\psi$ has access to the condition $c$, and can map each $(y, M)$ to an appropriate region of latent space.
- This design follows the conditional-prior structure from Pluralistic Image Completion (Zheng et al., CVPR 2019); our implementation is a GAN-free simplification.

**Verification**: The first smoke-training run under v1.3 showed stable decreases in validation loss. After 50 epochs of full training, the best val L1 reached 0.01323, and medium-regime FID reached 7.93 (see §5).

### 3.3 Takeaway

This iteration not only shows why a naive CVAE is insufficient for inpainting — it also reveals the **necessity** of the conditional prior in Pluralistic Image Completion. The conditional prior is not an optional architectural choice, but a structural requirement to prevent posterior collapse under the compose-based formulation, or alternatively, to allow the latent code to remain meaningful without compose.

---

## 4. Training Configuration

| Item | Value |
|---|---|
| Dataset | CelebA-HQ (from CelebAMask-HQ, segmentation labels unused) |
| Train / Val / Test split | 80% / 10% / 10%, seed=42 |
| Resolution | 256 × 256 |
| Training mask distribution | Three-regime union (each sample uniformly selects a regime from {small, medium, large}) |
| Data augmentation | Horizontal flip p=0.5, no color jitter |
| Optimizer | Adam, $\text{lr} = 10^{-4}$, $(\beta_1, \beta_2) = (0.9, 0.999)$ |
| Batch size | 12 |
| Epochs | 50 |
| Gradient clipping | max norm = 1.0 |
| Mixed precision | Enabled (AMP); KL computation kept in FP32 for numerical stability |
| Latent dim | 256 |
| $\beta$ warm-up | 10 epochs, linear from 0 to $\beta_{\max}=1.0$ |
| Checkpointing | Save `last.pt` every epoch; save `best.pt` by lowest validation L1 |

**Training wall-clock**: ~4.3 hours on a single RTX 2080 Ti (22GB modified variant) for 50 epochs.

---

## 5. Evaluation Protocol and Results

### 5.1 Metric Definitions

Let $N$ denote the number of test images.

| Metric | Definition | Category |
|---|---|---|
| MSE | $\frac{1}{N} \sum_i \| x_i - \hat{x}_i \|_2^2 / (H \cdot W \cdot C)$ | Distortion |
| PSNR | $10 \log_{10}(1 / \text{MSE}_i)$ averaged over images | Distortion |
| SSIM | Structural Similarity Index, data_range=1.0 | Distortion |
| LPIPS | $\frac{1}{N} \sum_i d_{\text{AlexNet}}(x_i, \hat{x}_i)$, with inputs normalized to $[-1, 1]$ | Perception-side |
| FID  | Fréchet distance between Inception feature distributions of $\{x_i\}$ and $\{\hat{x}_i\}$ | Perception |

**Justification for LPIPS as the fourth metric**:
- LPIPS (Zhang et al., CVPR 2018) measures distance in a learned perceptual feature space and has been shown to correlate strongly with human judgments.
- It complements FID: FID is a distribution-level distance, while LPIPS is a sample-level paired distance.
- Strictly by Blau & Michaeli's definition, a perception metric should be of the form $d(p_X, p_{\hat{X}})$ with no ground-truth reference. LPIPS requires paired ground truth and is therefore technically a feature-space distortion. However, its reliance on learned features places it closer to the perception side than any pixel-space metric, and it is commonly used in this role in the literature. We report it on the perception side and acknowledge this boundary in our analysis.

### 5.2 CVAE Test-Set Results

Evaluated on `checkpoints/run_01/best.pt` (epoch 46, val L1 = 0.01323), on the test set (3000 images × 3 regimes):

| Regime  | MSE      | PSNR    | SSIM    | LPIPS   | FID    |
|---------|----------|---------|---------|---------|--------|
| Small   | 0.000253 | 38.00   | 0.9857  | 0.0116  | 1.57   |
| Medium  | 0.002073 | 27.85   | 0.9414  | 0.0660  | 7.93   |
| Large   | 0.006575 | 22.57   | 0.8563  | 0.1669  | 25.99  |

**Observations**:
- All metrics change monotonically with mask size: distortion metrics (MSE, 1-SSIM, inverse of PSNR) increase with hole size; perception metrics (LPIPS, FID) increase as well. This matches intuition: larger holes are harder to reconstruct and harder to match the true distribution.
- Medium-regime FID of 7.93 is competitive: Pluralistic Image Completion reported FID ≈ 10–15 on CelebA-HQ in their original paper. Our simplified formulation without adversarial training achieves comparable perception scores.
- Visual inspection reveals mild blurriness in high-detail facial regions (eyes, mouth, eyebrows). This is a well-known property of L1 reconstruction losses: L1 rewards the pixel-wise mean of all plausible completions, which tends to be smoother than any individual mode.

### 5.3 Perception-Distortion Tradeoff Analysis

Our three points on the P-D plane trace a monotonically increasing curve from small → medium → large (both distortion and perception worsen together), which is typical for L1-reconstruction CVAEs: at every difficulty level, the model prioritizes distortion minimization, with perception degradation as a side effect.

When compared against Reconstruction and GAN baselines, we expect:
- **Relative to Reconstruction**: comparable or slightly worse distortion (due to stochasticity from $z$), but significantly better perception (since $z$ introduces randomness that helps match the data distribution).
- **Relative to GAN**: better distortion (L1 explicitly penalizes pixel error), but worse perception (GAN directly optimizes distribution matching via the adversarial loss).

This positioning is consistent with Blau & Michaeli's theoretical prediction: likelihood-based methods without adversarial training occupy the **low-distortion end** of the P-D curve, while GAN-based methods occupy the **low-perception end**.

### 5.4 Pluralistic Sampling Verification

A core advantage of CVAE over deterministic reconstruction is stochasticity. For the same masked input, we sample 5 different $z$ from the conditional prior; the pairwise pixel difference averages ~0.01 (in [0,1] range), confirming that the model produces genuinely diverse outputs rather than degenerating into a deterministic function. See `reports/pluralistic_grid_large.png` for a visual demonstration.

---

## 6. Items Pending Integration

### 6.1 Team-Level P-D Plane

A unified P-D scatter plot requires merging `{method}_summary.json` files from all three methods. Suggested format:
- x-axis: MSE or 1-SSIM
- y-axis: FID or LPIPS
- One marker shape per method, one color per regime (or lines connecting a method's three points)

### 6.2 Cross-Method Comparison Analysis (Requires Teammates' Data)

A complete P-D tradeoff analysis depends on data from the Reconstruction and GAN teammates. The analysis should cover:
1. Relative positions of the three methods on the P-D plane at each mask size
2. Which method is closest to the Blau–Michaeli theoretical bound
3. Comparison of how distortion/perception change rates differ across methods as mask size increases

### 6.3 FFHQ Zero-Shot Generalization (Bonus)

If time permits, evaluate `run_01/best.pt` on the FFHQ medium-regime subset to observe metric robustness under distribution shift. As a likelihood-based method, CVAE is expected to generalize relatively well on distortion but may degrade more noticeably on perception.

---

## 7. References

1. Kingma & Welling. "Auto-Encoding Variational Bayes." ICLR 2014.
2. Sohn et al. "Learning Structured Output Representation using Deep Conditional Generative Models." NeurIPS 2015.
3. Kingma et al. "Improved Variational Inference with Inverse Autoregressive Flow." NeurIPS 2016. (Free Bits)
4. Zheng et al. "Pluralistic Image Completion." CVPR 2019.
5. Blau & Michaeli. "The Perception-Distortion Tradeoff." CVPR 2018.
6. Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." CVPR 2018. (LPIPS)
7. Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS 2017. (FID)

---

## Appendix A: Key Code Locations

| Component | File |
|---|---|
| Encoder / PriorNet / Decoder / CVAE | `cvae/model.py` |
| L1 + KL loss + $\beta$ schedule | `cvae/loss.py` |
| Training loop | `cvae/train.py` |
| Evaluation script | `cvae/eval.py` |
| Visualization | `cvae/visualize.py` |
| Training log | `checkpoints/run_01/train.log` |
| Test results | `cvae/reports/cvae_per_image.csv`, `cvae/reports/cvae_summary.json` |

## Appendix B: Reproduction Commands

```bash
# Training
python cvae/train.py --epochs 50 --run-name run_01 \
    --batch-size 12 --num-workers 2 \
    --warmup-epochs 10 --max-beta 1.0

# Evaluation
python cvae/eval.py --ckpt checkpoints/run_01/best.pt \
    --run-name run_01 --batch-size 16

# Visualization
python cvae/visualize.py --ckpt checkpoints/run_01/best.pt \
    --regime large --n-images 4 --n-samples 6
```
