from typing import Dict
import torch


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_per_sample = kl_per_dim.sum(dim=1)
    return kl_per_sample.mean()


def kl_standard_normal_free_bits(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.5,
) -> torch.Tensor:
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim_batch_mean = kl_per_dim.mean(dim=0)
    kl_floored = kl_per_dim_batch_mean.clamp(min=free_bits)
    kl = kl_floored.sum()
    return kl


def kl_gauss_gauss(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:

    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (
        logvar_p - logvar_q
        + (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
    )
    return kl.sum(dim=1).mean()



def beta_schedule(epoch: int, warmup_epochs: int = 10, max_beta: float = 1.0) -> float:
    if warmup_epochs <= 0:
        return max_beta
    return min(max_beta, max_beta * epoch / warmup_epochs)


def cvae_loss(
    x_hat: torch.Tensor,
    gt: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    l1 = (x_hat - gt).abs().mean()
    kl = kl_gauss_gauss(mu_q, logvar_q, mu_p, logvar_p)
    total = l1 + beta * kl

    log_dict = {
        "l1":    l1.item(),
        "kl":    kl.item(),
        "beta":  beta,
        "total": total.item(),
    }
    return total, log_dict



if __name__ == "__main__":
    torch.manual_seed(12138)
    B, D = 4, 256

    mu = torch.randn(B, D)
    logvar = torch.randn(B, D)
    kl = kl_gauss_gauss(mu, logvar, mu, logvar)
    print(f"KL = {kl.item():.6e}")

    zeros = torch.zeros(B, D)
    kl = kl_gauss_gauss(zeros, zeros, zeros, zeros)
    print(f"KL = {kl.item():.6e}")

    mu_q = torch.randn(B, D) * 0.5
    logvar_q = torch.randn(B, D) * 0.3
    mu_p = torch.zeros(B, D)
    logvar_p = torch.zeros(B, D)  # logvar=0 means sigma=1, i.e. N(0,I)

    kl_general = kl_gauss_gauss(mu_q, logvar_q, mu_p, logvar_p)
    kl_std = kl_standard_normal(mu_q, logvar_q)
    print(f" kl_gauss_gauss   = {kl_general.item():.4f}")
    print(f" kl_standard_normal = {kl_std.item():.4f}")
    diff = (kl_general - kl_std).abs().item()
    print(f" diff = {diff:.2e})")

    mu_q = torch.randn(B, D) * 0.3
    logvar_q = torch.randn(B, D) * 0.2
    mu_p = torch.randn(B, D) * 0.7
    logvar_p = torch.randn(B, D) * 0.4
    kl_qp = kl_gauss_gauss(mu_q, logvar_q, mu_p, logvar_p)
    kl_pq = kl_gauss_gauss(mu_p, logvar_p, mu_q, logvar_q)
    print(f" KL(q||p) = {kl_qp.item():.4f}")
    print(f" KL(p||q) = {kl_pq.item():.4f}")

    for _ in range(5):
        mq = torch.randn(B, D) * 2
        lvq = torch.randn(B, D) * 2
        mp = torch.randn(B, D) * 2
        lvp = torch.randn(B, D) * 2
        kl = kl_gauss_gauss(mq, lvq, mp, lvp)
        assert kl.item() >= -1e-5, f"KL negative: {kl.item()}"
    print("KL is non negative")

    x_hat = torch.rand(B, 3, 256, 256, requires_grad=True)
    gt = torch.rand(B, 3, 256, 256)
    mu_q = torch.randn(B, D, requires_grad=True)
    logvar_q = torch.randn(B, D, requires_grad=True)
    mu_p = torch.randn(B, D, requires_grad=True)
    logvar_p = torch.randn(B, D, requires_grad=True)

    total, log = cvae_loss(x_hat, gt, mu_q, logvar_q, mu_p, logvar_p, beta=0.5)
    print(f"  l1    = {log['l1']:.4f}")
    print(f"  kl    = {log['kl']:.4f}")
    print(f"  beta  = {log['beta']}")
    print(f"  total = {log['total']:.4f}")

    expected = log["l1"] + log["beta"] * log["kl"]
    assert abs(log["total"] - expected) < 1e-5
    print(f" Total == l1 + beta * kl")

    total.backward()
    assert x_hat.grad is not None
    assert mu_q.grad is not None and logvar_q.grad is not None
    assert mu_p.grad is not None and logvar_p.grad is not None
    print(f" Gradient normally flows to x_hat, mu_q/p, logvar_q/p")

    print("\nAll tests passed.")