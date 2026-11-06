# -*- coding: utf-8 -*-
"""Differential-privacy accounting and the private low-rank release
(revised Section 5).

Mechanism (one round t):
    G_t      = sum_k clip_C(G_{t,k})            aggregate of clipped client updates
    V_t      : d2 x r, orthonormal columns, chosen WITHOUT looking at G_t
               (public seed, or post-processing of earlier releases)
    release  : A_t = G_t V_t + N,  N ~ N(0, sigma^2 I_{d1 x r})
    server   : G_hat_t = A_t V_t^T

Adding or removing one client changes G_t by at most C in Frobenius norm,
and ||(G - G') V||_F <= ||G - G'||_F ||V||_2 = ||G - G'||_F because V has
orthonormal columns. The release is therefore a Gaussian mechanism with
L2 sensitivity C, whatever r is. V_t itself is never computed from G_t.

Accounting uses zero-concentrated DP (Bun & Steinke, 2016): a Gaussian
mechanism with sensitivity C and noise sigma is rho = C^2 / (2 sigma^2)-zCDP,
rho adds up over rounds, and rho-zCDP implies
(rho + 2 sqrt(rho ln(1/delta)), delta)-DP. For a single release the analytic
Gaussian mechanism (Balle & Wang, 2018) is also provided.
"""
import math
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------
def eps_from_rho(rho, delta):
    """(eps, delta)-DP implied by rho-zCDP."""
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def rho_from_eps(eps, delta):
    """Largest rho whose zCDP->DP conversion stays within (eps, delta)."""
    L = math.log(1.0 / delta)
    return (math.sqrt(L + eps) - math.sqrt(L)) ** 2


def sigma_for_total_budget(eps_total, delta, rounds, sensitivity=1.0):
    """Per-round Gaussian noise std so that `rounds` releases compose to
    (eps_total, delta)-DP under zCDP accounting."""
    rho_round = rho_from_eps(eps_total, delta) / rounds
    return sensitivity / math.sqrt(2.0 * rho_round)


def analytic_gaussian_sigma(eps, delta, sensitivity=1.0, tol=1e-12):
    """Smallest sigma for which one Gaussian release is (eps, delta)-DP
    (Balle & Wang 2018, Theorem 8). Valid for every eps > 0."""
    def delta_of(sig):
        a = sensitivity / (2.0 * sig)
        b = eps * sig / sensitivity
        return norm.cdf(a - b) - math.exp(eps) * norm.cdf(-a - b)
    lo, hi = 1e-6 * sensitivity, 1.0 * sensitivity
    while delta_of(hi) > delta:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if delta_of(mid) > delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * hi:
            break
    return hi


def classical_gaussian_sigma(eps, delta, sensitivity=1.0):
    """Dwork & Roth (2014) bound. Only valid for eps < 1; kept for comparison."""
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps


# ---------------------------------------------------------------------------
# private low-rank release
# ---------------------------------------------------------------------------
def random_basis(d2, rng):
    """Public, data-independent orthonormal basis of R^{d2} (ordered)."""
    q, _ = np.linalg.qr(rng.standard_normal((d2, d2)))
    return q


def basis_from_release(G_hat_prev, d2, rng):
    """Ordered orthonormal basis of R^{d2} built only from the previous
    round's PUBLIC reconstruction (post-processing, no privacy cost).
    Directions are ordered by the energy the previous release put on them;
    the rest of the space is completed with a random orthonormal complement."""
    if G_hat_prev is None:
        return random_basis(d2, rng)
    _, s, vt = np.linalg.svd(G_hat_prev, full_matrices=False)
    k = int((s > 1e-12 * max(s[0], 1e-300)).sum())
    V = vt[:k].T
    if k < d2:
        R = rng.standard_normal((d2, d2 - k))
        R -= V @ (V.T @ R)
        Q, _ = np.linalg.qr(R)
        V = np.concatenate([V, Q[:, : d2 - k]], axis=1)
    return V


def private_release(G, V, r, sigma, rng):
    """One private round: returns (A_noisy, G_hat)."""
    Vr = V[:, :r]
    A = G @ Vr + rng.normal(0.0, sigma, size=(G.shape[0], r))
    return A, A @ Vr.T


def energies(G, V):
    """e_j = ||G v_j||^2 for the ordered basis columns."""
    return ((G @ V) ** 2).sum(axis=0)


def optimal_rank_threshold(e, d1, sigma, r_max=None):
    """Exact minimiser of the expected loss
        L(r) = ||G||^2 - sum_{j<=r} e_j + d1 r sigma^2
    over r >= 1 (Proposition 1). For non-increasing e_j this is
    #{j : e_j > d1 sigma^2}, clipped to [1, r_max]."""
    e = np.asarray(e, float)
    r_max = r_max or len(e)
    gain = np.cumsum(e[:r_max] - d1 * sigma ** 2)
    return int(np.argmax(gain)) + 1


def expected_loss(e, total_energy, d1, sigma):
    """L(r) for r = 1..len(e)."""
    e = np.asarray(e, float)
    r = np.arange(1, len(e) + 1)
    return total_energy - np.cumsum(e) + d1 * r * sigma ** 2
