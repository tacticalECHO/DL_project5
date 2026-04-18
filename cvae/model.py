import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 7, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # conv: spatial 256 -> 8, channels in_channels -> 512
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.SiLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.SiLU(inplace=True),
        )

        self.flatten_dim = 512 * 8 * 8
        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class PriorNet(nn.Module):
    def __init__(self, in_channels: int = 4, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # conv: spatial 256 -> 8, channels in_channels -> 256
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU(inplace=True),
        )

        self.flatten_dim = 256 * 8 * 8
        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = logvar.clamp(min=-2.0, max=20.0)
        return mu, logvar


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, cond_ch: int, out_ch: int, use_cond: bool = True):
        super().__init__()
        self.use_cond = use_cond
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        conv1_in = in_ch + cond_ch if use_cond else in_ch
        self.conv1 = nn.Conv2d(conv1_in, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x)
        if self.use_cond:
            assert cond is not None, "cond must be provided when use_cond=True"
            x = torch.cat([x, cond], dim=1)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_ch = 4  # masked(3) + mask(1)

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.fc_act = nn.SiLU(inplace=True)

        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch=512, cond_ch=self.cond_ch, out_ch=512, use_cond=True),
            DecoderBlock(in_ch=512, cond_ch=self.cond_ch, out_ch=256, use_cond=True),
            DecoderBlock(in_ch=256, cond_ch=self.cond_ch, out_ch=128, use_cond=True),
            DecoderBlock(in_ch=128, cond_ch=self.cond_ch, out_ch=64,  use_cond=True),
            DecoderBlock(in_ch=64,  cond_ch=self.cond_ch, out_ch=32,  use_cond=True),
        ])

        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def _get_spatial_cond(self, masked: torch.Tensor, mask: torch.Tensor, target_size: int) -> torch.Tensor:
        if target_size == masked.shape[-1]:
            return torch.cat([masked, mask], dim=1)

        masked_down = nn.functional.interpolate(
            masked, size=target_size, mode="bilinear", align_corners=False
        )
        mask_down = nn.functional.interpolate(
            mask, size=target_size, mode="nearest"
        )
        return torch.cat([masked_down, mask_down], dim=1)

    def forward(self, z, masked, mask):
        h = self.fc(z)
        h = self.fc_act(h)
        h = h.view(-1, 512, 8, 8)

        target_sizes = [16, 32, 64, 128, 256]

        for block, size in zip(self.blocks, target_sizes):
            if block.use_cond:
                cond = self._get_spatial_cond(masked, mask, size)
            else:
                cond = None
            h = block(h, cond)

        x_raw = torch.sigmoid(self.final_conv(h))
        return x_raw


class CVAE(nn.Module):

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels=7, latent_dim=latent_dim)
        self.prior   = PriorNet(in_channels=4, latent_dim=latent_dim)  # ← v1.3
        self.decoder = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        gt: torch.Tensor,
        masked: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

        # Posterior
        enc_input = torch.cat([gt, masked, mask], dim=1)   # [B, 7, H, W]
        mu_q, logvar_q = self.encoder(enc_input)

        # Prior
        prior_input = torch.cat([masked, mask], dim=1)     # [B, 4, H, W]
        mu_p, logvar_p = self.prior(prior_input)

        # Sample z
        if self.training:
            z = self.reparameterize(mu_q, logvar_q)
        else:
            z = self.reparameterize(mu_p, logvar_p)

        x_hat = self.decoder(z, masked, mask)

        return x_hat, (mu_q, logvar_q, mu_p, logvar_p)

    @torch.no_grad()
    def sample(
        self,
        masked: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Pluralistic sampling: multiple z from the conditional prior."""
        self.eval()
        prior_input = torch.cat([masked, mask], dim=1)
        mu_p, logvar_p = self.prior(prior_input)

        results = []
        for _ in range(n_samples):
            z = self.reparameterize(mu_p, logvar_p)
            x_hat = self.decoder(z, masked, mask)
            results.append(x_hat)
        return torch.stack(results, dim=0)


if __name__ == "__main__":
    torch.manual_seed(12138)
    B, latent_dim = 2, 256

    print("Testing Encoder ......")
    encoder = Encoder(in_channels=7, latent_dim=latent_dim)
    fake_input = torch.randn(B, 7, 256, 256)
    mu, logvar = encoder(fake_input)
    assert mu.shape == (B, latent_dim)
    assert logvar.shape == (B, latent_dim)
    print(f"[OK] mu: {tuple(mu.shape)}, logvar: {tuple(logvar.shape)}")
    print(f"[Info] Encoder params: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M")

    print("\nTesting PriorNet ......")
    prior = PriorNet(in_channels=4, latent_dim=latent_dim)
    prior_input = torch.randn(B, 4, 256, 256)
    mu_p, logvar_p = prior(prior_input)
    assert mu_p.shape == (B, latent_dim)
    assert logvar_p.shape == (B, latent_dim)
    print(f"[OK] mu_p: {tuple(mu_p.shape)}, logvar_p: {tuple(logvar_p.shape)}")
    print(f"[Info] PriorNet params: {sum(p.numel() for p in prior.parameters())/1e6:.2f}M")

    print("\nTesting Decoder ......")
    decoder = Decoder(latent_dim=latent_dim)
    z = torch.randn(B, latent_dim)
    masked = torch.rand(B, 3, 256, 256)
    mask = (torch.rand(B, 1, 256, 256) > 0.7).float()
    x_raw = decoder(z, masked, mask)
    assert x_raw.shape == (B, 3, 256, 256)
    assert x_raw.min() >= 0.0 and x_raw.max() <= 1.0
    print(f"[OK] x_raw shape: {tuple(x_raw.shape)}, range: [{x_raw.min():.4f}, {x_raw.max():.4f}]")
    print(f"[Info] Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")

    print("\nTesting CVAE (full forward) ......")
    model = CVAE(latent_dim=latent_dim)
    gt = torch.rand(B, 3, 256, 256)
    mask = (torch.rand(B, 1, 256, 256) > 0.7).float()
    masked = gt * (1 - mask)

    # ---- Train mode ----
    model.train()
    x_hat, (mu_q, logvar_q, mu_p, logvar_p) = model(gt, masked, mask)

    assert x_hat.shape == (B, 3, 256, 256), f"x_hat shape wrong: {x_hat.shape}"
    assert mu_q.shape == (B, latent_dim) and logvar_q.shape == (B, latent_dim)
    assert mu_p.shape == (B, latent_dim) and logvar_p.shape == (B, latent_dim)
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0, "x_hat out of [0,1]"
    print(f"[OK] Train mode:")
    print(f"     x_hat shape: {tuple(x_hat.shape)}, range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")
    print(f"     mu_q / logvar_q / mu_p / logvar_p all shape {(B, latent_dim)}")

    print(f"[OK] No compose (v1.3): decoder outputs full image directly")

    # Gradient flow
    loss = x_hat.sum() + mu_q.sum() + logvar_q.sum() + mu_p.sum() + logvar_p.sum()
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    assert n_with_grad == n_total, f"Grad missing: {n_with_grad}/{n_total}"
    print(f"[OK] Gradient flows to all {n_total} parameter tensors")

    model.eval()
    with torch.no_grad():
        x_hat_eval, (mu_q_e, logvar_q_e, mu_p_e, logvar_p_e) = model(gt, masked, mask)
    assert x_hat_eval.shape == (B, 3, 256, 256)
    print(f"\n[OK] Eval mode (z from prior):")
    print(f"     x_hat shape: {tuple(x_hat_eval.shape)}, range: [{x_hat_eval.min():.4f}, {x_hat_eval.max():.4f}]")

    diff = (x_hat.detach() - x_hat_eval).abs().mean()
    print(f"     Train vs eval mean pixel diff: {diff:.4f} (should be > 0)")

    print("\nTesting CVAE.sample() (pluralistic)")
    n_samples = 3
    samples = model.sample(masked, mask, n_samples=n_samples)
    assert samples.shape == (n_samples, B, 3, 256, 256)
    print(f"[OK] samples shape: {tuple(samples.shape)}")

    diff_01 = (samples[0] - samples[1]).abs().mean()
    diff_02 = (samples[0] - samples[2]).abs().mean()
    print(f"     sample[0] vs [1] diff: {diff_01:.4f}")
    print(f"     sample[0] vs [2] diff: {diff_02:.4f}")
    print(f"     (both should be > 0, proving stochastic diversity)")

    print("\nSummary")
    total = sum(p.numel() for p in model.parameters())
    print(f"CVAE total params: {total/1e6:.2f}M")
    print(f"  Encoder:  {sum(p.numel() for p in model.encoder.parameters())/1e6:.2f}M")
    print(f"  PriorNet: {sum(p.numel() for p in model.prior.parameters())/1e6:.2f}M")
    print(f"  Decoder:  {sum(p.numel() for p in model.decoder.parameters())/1e6:.2f}M")