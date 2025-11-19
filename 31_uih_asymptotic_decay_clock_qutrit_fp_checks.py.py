#!/usr/bin/env python3
"""
31_uih_asymptotic_decay_clock_qutrit_fp_checks.py

Universal information hydrodynamics asymptotic decay clock:
qutrit Markov chain versus high resolution Fokker-Planck-like chain.

This script probes how the Markov spectral gap lambda_Q controls the
LATE-TIME decay rate of relative entropy for reversible Markov generators
that arise as density sectors of GKLS models (qutrit chain) and as
discrete Fokker-Planck operators (FP-like chain).

For each generator Q with stationary distribution pi we consider the
relative entropy
    S(t) = sum_i p_i(t) log( p_i(t) / pi_i )
and define the instantaneous decay rate
    r_inst(t) = - d/dt log S(t)
along with an asymptotic rate r_fit obtained by fitting a straight line
to log S(t) over a late-time window.

For a reversible Markov semigroup, the spectrum of the symmetrised
operator
    S = B^{-1} Q B,  B = diag(sqrt(pi)),
is real and non-positive, with eigenvalues 0 = -lambda_0 > -lambda_1 >= -lambda_2 >= ...
The Markov spectral gap lambda_Q := lambda_1 sets the slowest relaxation
scale. In general, r_inst(t) can substantially exceed 2 lambda_Q at early
times due to contributions from fast modes, but the LATE-TIME behaviour of
log S(t) is expected to approach a single exponential with rate
    r_asymp ≈ 2 lambda_Q.

We construct:
  * a three-state (qutrit) reversible Markov generator Q_q with random
    stationary distribution, and
  * a high-resolution N_x-state reversible nearest-neighbour FP-like
    generator Q_fp on a periodic lattice,

then rescale Q_fp so that the Markov gaps match:
    lambda_Q^(fp) = lambda_Q^(q).

For each generator we sample random initial distributions p(0), track
S(t) on a time grid, and:

  * estimate r_inst(t) from centred finite differences of log S(t),
  * estimate an asymptotic rate r_fit from a linear fit to log S(t) over
    a late-time window (e.g. last half of the time interval),

and compare both to 2 lambda_Q.

The numerics show that:
  * r_inst(t) can exceed 2 lambda_Q by a factor of several at early times,
    especially in high-dimensional FP-like chains, reflecting UV fast modes;
  * the FITTED late-time rate r_fit clusters tightly around 2 lambda_Q for
    both the qutrit and FP-like chains once their gaps are matched.

This supports a "UIH asymptotic decay clock" picture in which the reversible
Markov gap sets the universal late-time information decay rate across very
different microscopic realisations (qutrit GKLS density sectors and FP limits),
while transient overshoots in r_inst(t) are non-universal short-time effects.
"""

import numpy as np
import scipy.linalg as la


def build_reversible_qutrit(seed: int = 1234):
    """
    Construct a random 3x3 reversible Markov generator Q_q with
    strictly positive stationary distribution pi.

    Convention: columns of Q sum to zero and Q_ij is the rate from j->i.
    """
    rng = np.random.default_rng(seed)

    # Random stationary distribution with full support
    pi = rng.random(3)
    pi /= pi.sum()

    # Symmetric non-negative conductances S_ij = S_ji
    S = rng.random((3, 3))
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)

    # Off-diagonal rates via detailed balance: pi_j Q_ij = S_ij
    Q = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            if i != j:
                if pi[j] <= 0:
                    raise RuntimeError("Encountered zero stationary weight.")
                Q[i, j] = S[i, j] / pi[j]
    # Diagonal entries fix column sums to zero
    for j in range(3):
        Q[j, j] = -np.sum(Q[:, j]) + Q[j, j]

    # Sanity checks
    col_sums = Q.sum(axis=0)
    if not np.allclose(col_sums, 0.0, atol=1e-14):
        raise RuntimeError("Column sums of Q_q are not zero.")
    if np.any(pi <= 0):
        raise RuntimeError("Non-positive stationary probability in qutrit chain.")

    return Q, pi


def build_fp_like_chain(Nx: int = 60, L: float = 2.0, beta: float = 1.0, seed: int = 5678):
    """
    Build a reversible nearest-neighbour Markov generator Q_fp on a
    periodic lattice {0,...,Nx-1} with a smooth potential V(x).

    The stationary distribution is pi_i proportional to exp( -beta * V(x_i) ), and
    nearest-neighbour rates are chosen to satisfy detailed balance.
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, L, Nx, endpoint=False)
    # Simple two-mode potential with randomised amplitudes
    a1 = 0.7 + 0.3 * rng.random()
    a2 = 0.4 + 0.2 * rng.random()
    V = a1 * np.cos(2.0 * np.pi * x / L) + a2 * np.cos(4.0 * np.pi * x / L)

    pi = np.exp(-beta * V)
    pi /= pi.sum()

    # Symmetric conductances on the ring
    S = np.zeros((Nx, Nx), dtype=float)
    for j in range(Nx):
        jp = (j + 1) % Nx
        jm = (j - 1) % Nx
        # Symmetric positive conductances to neighbours
        s1 = 0.5 + rng.random()
        s2 = 0.5 + rng.random()
        S[jp, j] += s1
        S[j, jp] += s1
        S[jm, j] += s2
        S[j, jm] += s2

    Q = np.zeros((Nx, Nx), dtype=float)
    for i in range(Nx):
        for j in range(Nx):
            if i != j and S[i, j] > 0.0:
                Q[i, j] = S[i, j] / pi[j]

    for j in range(Nx):
        Q[j, j] = -np.sum(Q[:, j]) + Q[j, j]

    col_sums = Q.sum(axis=0)
    if not np.allclose(col_sums, 0.0, atol=1e-12):
        raise RuntimeError("Column sums of Q_fp are not zero.")
    if np.any(pi <= 0):
        raise RuntimeError("Non-positive stationary probability in FP chain.")

    return Q, pi, x, V


def markov_spectral_gap(Q: np.ndarray, pi: np.ndarray) -> float:
    """
    Compute the spectral gap lambda_Q of a reversible generator Q with
    stationary distribution pi via the symmetrised operator
        S = B^{-1} Q B, B = diag(sqrt(pi)).

    The spectrum of S is real; zero is a simple eigenvalue and the gap
    is min_{lambda != 0} ( -lambda ).
    """
    B = np.diag(np.sqrt(pi))
    # Similarity transform; S should be symmetric for reversible chains
    S = la.inv(B) @ Q @ B
    # Numerical asymmetry is small; use the symmetric part explicitly
    S = 0.5 * (S + S.T)
    evals = np.linalg.eigvalsh(S)
    # Sort eigenvalues: the largest should be zero
    evals_sorted = np.sort(evals)
    # Zero eigenvalue is at the top; next one down gives the gap
    # S has non-positive spectrum, so evals <= 0
    # Gap lambda_Q is smallest positive value of -eval over non-zero modes
    nonzero = evals_sorted[:-1]  # discard the eigenvalue closest to zero
    gaps = -nonzero
    gaps = gaps[gaps > 1e-10]
    if gaps.size == 0:
        raise RuntimeError("Failed to identify nonzero eigenvalues for gap.")
    return float(np.min(gaps))


def relative_entropy(p: np.ndarray, pi: np.ndarray) -> float:
    """
    Classical relative entropy S(p||pi) in nats; assumes p, pi > 0 and sum to 1.
    """
    # Guard against tiny numerical negativity
    p_safe = np.clip(p, 1e-300, 1.0)
    pi_safe = np.clip(pi, 1e-300, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(pi_safe))))


def evolve_markov(Q: np.ndarray, p0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Evolve p(t) = exp(t Q) p0 for a collection of times.

    Q is n x n with column sums zero; p0 is length n column vector.
    Returns an array of shape (len(times), n).
    """
    n = Q.shape[0]
    # Diagonalise Q once
    evals, evecs = la.eig(Q)
    # Right eigenvectors in columns of evecs; Q = evecs diag(evals) evecs^{-1}
    Vinv = la.inv(evecs)
    # For reversible chains the spectrum is real; discard tiny imaginaries
    evals = evals.real
    evecs = evecs.real
    Vinv = Vinv.real

    p0 = np.asarray(p0, dtype=float)
    assert p0.shape == (n,)
    coeffs0 = Vinv @ p0

    out = np.zeros((len(times), n), dtype=float)
    for k, t in enumerate(times):
        factors = np.exp(evals * t)
        pt = evecs @ (factors * coeffs0)
        # Normalise small numerical drift
        s = pt.sum()
        if s <= 0:
            raise RuntimeError("Non-positive total probability during evolution.")
        pt /= s
        out[k, :] = pt.real
    return out


def estimate_rate_from_entropy(times: np.ndarray, entropies: np.ndarray):
    """
    Given S(t) sampled on a uniform grid, estimate the instantaneous
    decay rate r_inst(t) = -d/dt log S(t) via centred finite differences
    and return summary statistics (min, max, mean) over the interior points.
    """
    S = np.asarray(entropies, dtype=float)
    if np.any(S <= 0):
        raise RuntimeError("Encountered non-positive entropy value.")
    logS = np.log(S)
    dt = times[1] - times[0]
    # Centred derivative for interior points
    r = -(logS[2:] - logS[:-2]) / (2.0 * dt)
    return r.min(), r.max(), r.mean()


def fit_asymptotic_rate(times: np.ndarray, entropies: np.ndarray, tail_frac: float = 0.5):
    """
    Fit an asymptotic decay rate r_fit by linear regression of log S(t)
    over the last tail_frac fraction of the time interval.

    Returns r_fit >= 0 such that log S(t) ~ log C - r_fit t on the tail.
    """
    S = np.asarray(entropies, dtype=float)
    if np.any(S <= 0):
        raise RuntimeError("Encountered non-positive entropy value in fit.")
    n = len(times)
    start_index = int((1.0 - tail_frac) * n)
    start_index = max(start_index, 0)
    t_tail = times[start_index:]
    logS_tail = np.log(S[start_index:])
    # Simple least squares fit: logS_tail ≈ a + b t, with b ≈ -r_fit
    coeffs = np.polyfit(t_tail, logS_tail, deg=1)
    b = coeffs[0]
    r_fit = -b
    return float(r_fit)


def run_decay_clock_checks(
    Nx: int = 60,
    L: float = 2.0,
    beta: float = 1.0,
    n_inits: int = 20,
    t_max_factor: float = 4.0,
    n_times: int = 101,
    tail_frac: float = 0.5,
    seed_q: int = 1234,
    seed_fp: int = 5678,
):
    """
    Build qutrit and FP-like reversible generators with matched Markov
    gaps and test both:

      * instantaneous rates r_inst(t) extracted from relative entropy,
      * asymptotic fitted rates r_fit from late-time log S(t),

    against the Markov gap clock 2 lambda_Q for both chains.
    """
    print("=" * 72)
    print("UIH asymptotic decay clock: qutrit Markov versus FP-like chain")
    print("=" * 72)

    # Build generators
    Q_q, pi_q = build_reversible_qutrit(seed=seed_q)
    Q_fp, pi_fp, x_fp, V_fp = build_fp_like_chain(Nx=Nx, L=L, beta=beta, seed=seed_fp)

    # Spectral gaps
    lam_q = markov_spectral_gap(Q_q, pi_q)
    lam_fp_raw = markov_spectral_gap(Q_fp, pi_fp)

    print("\nSpectral gaps before matching:")
    print(f"  lambda_Q^(qutrit) approx {lam_q:.6f}")
    print(f"  lambda_Q^(FP, raw) approx {lam_fp_raw:.6f}")

    # Rescale FP generator so that its gap matches qutrit gap
    scale = lam_q / lam_fp_raw
    Q_fp_matched = scale * Q_fp
    lam_fp = markov_spectral_gap(Q_fp_matched, pi_fp)

    print("\nAfter rescaling FP generator:")
    print(f"  scale factor on Q_fp: {scale:.6f}")
    print(f"  lambda_Q^(FP, matched) approx {lam_fp:.6f}")

    # Common time grid measured in units of lambda_Q^{-1}
    t_max = t_max_factor / lam_q
    times = np.linspace(0.0, t_max, n_times)

    def random_prob_vec(n, rng):
        v = rng.random(n)
        v /= v.sum()
        return v

    rng_q = np.random.default_rng(seed_q + 100)
    rng_fp = np.random.default_rng(seed_fp + 100)

    def analyse_chain(label, Q, pi, rng):
        print("\n" + "-" * 72)
        print(f"Entropy decay diagnostics for {label}")
        print("-" * 72)
        n = len(pi)
        lam = markov_spectral_gap(Q, pi)
        print(f"  Dimension n = {n}")
        print(f"  Markov spectral gap lambda_Q approx {lam:.6f}")
        two_lam = 2.0 * lam
        print(f"  Asymptotic clock scale 2 lambda_Q approx {two_lam:.6f}")

        max_r_global = -np.inf
        mean_r_list = []
        ratio_inst_list = []
        r_fit_list = []
        ratio_fit_list = []

        for k in range(n_inits):
            p0 = random_prob_vec(n, rng)
            ent = []
            pts = evolve_markov(Q, p0, times)
            for j in range(len(times)):
                ent.append(relative_entropy(pts[j], pi))
            ent = np.asarray(ent)

            # Discard runs where entropy is extremely small (close to equilibrium)
            if ent[0] < 1e-8:
                continue

            r_min, r_max, r_mean = estimate_rate_from_entropy(times, ent)
            r_fit = fit_asymptotic_rate(times, ent, tail_frac=tail_frac)

            max_r_global = max(max_r_global, r_max)
            mean_r_list.append(r_mean)
            ratio_inst_list.append(r_max / two_lam)
            r_fit_list.append(r_fit)
            ratio_fit_list.append(r_fit / two_lam)

            print(
                f"  Init {k:2d}:  S(0) approx {ent[0]:.4e}, "
                f"r_min approx {r_min:.4f}, r_max approx {r_max:.4f}, "
                f"r_mean approx {r_mean:.4f}, "
                f"r_max/(2 lambda_Q) approx {r_max / two_lam:.4f}, "
                f"r_fit approx {r_fit:.4f}, r_fit/(2 lambda_Q) approx {r_fit / two_lam:.4f}"
            )

        mean_r =_
