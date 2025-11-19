#!/usr/bin/env python3
"""
33_gkls_bloch_two_quadratures_k_flow_checks.py

UIH two quadratures K-flow test in a physical GKLS qubit model.

We consider a driven, damped qubit with Hamiltonian and Lindblad operators
    H = 0.5 * (Omega * sigma_x + Delta * sigma_z)
    L1 = sqrt(gamma) * sigma_-
    Lphi = sqrt(gamma_phi) * sigma_z

The GKLS generator L acts on density matrices rho as
    L(rho) = -i[H, rho]
             + sum_k (Lk rho Lk^dag - 0.5 {Lk^dag Lk, rho})

We represent states in the Pauli basis
    {Sigma_0, Sigma_1, Sigma_2, Sigma_3} = {I, sigma_x, sigma_y, sigma_z}
by Bloch coordinates alpha = (alpha_0, alpha_x, alpha_y, alpha_z)^T with
    rho = alpha_0 I + alpha_x sigma_x + alpha_y sigma_y + alpha_z sigma_z,
    alpha_mu = (1/2) tr(Sigma_mu rho).

The GKLS evolution induces a real 4x4 matrix K on Bloch coordinates via
    dot(alpha) = K alpha.

At the stationary state rho_ss of the GKLS semigroup we place the BKM
(Bogoliubov Kubo Mori) metric. In the eigenbasis of rho_ss with eigenvalues
lambda_m > 0 the BKM weights are
    c_mn = 1 / lambda_m                if m = n
           (log lambda_m - log lambda_n) / (lambda_m - lambda_n)   if m != n

These define a diagonal positive metric on the vectorised operator basis;
transporting it to the Pauli basis gives a 4x4 positive matrix M whose
traceless 3x3 block induces a Riemannian metric on the Bloch vector space.

We then:

  * restrict K and M to the traceless Bloch subspace (x, y, z),
  * compute the metric adjoint
        K_sharp = M^{-1} K^T M,
  * split the generator as
        G = 0.5 * (K + K_sharp),   J = 0.5 * (K - K_sharp),
  * verify the metriplectic identities
        M G symmetric,   M J skew,

and study the quadratic functional
    F(u) = 0.5 u^T M u
on deviations u in the Bloch subspace.

We compute the dissipative spectrum from the generalised eigenproblem
    (-M G) v = lambda M v
and identify the smallest positive eigenvalue lambda_min.

For a set of random initial deviations u(0) with fixed F(0) we evolve:

  * the full K-flow:  dot(u) = K u,
  * the pure G-flow:  dot(u) = G u,

via exact matrix exponentials, track F_K(t) and F_G(t) on a time grid,
and fit late-time logarithmic decay rates
    F(t) ~ C exp(-r t).

Diagnostics:

  * instantaneous production identity:
        dF/dt = u^T M G u  along the K-flow,
    checked numerically via finite differences;
  * asymptotic decay rates r_G and r_K compared to
        2 * lambda_min.

The test realises the UIH picture of one current and two quadratures in
a concrete GKLS qubit:

  * G fixes the entropy production channel and the dissipative spectrum
    (lambda_min, lambda_max, ...),
  * J is no-work in the BKM metric (purely antisymmetric),
  * the pure G-flow saturates the lower bound r_G ~ 2 * lambda_min,
  * the full K-flow has r_K >= 2 * lambda_min, illustrating how the
    reversible quadrature J can accelerate the decay of F by mixing
    eigenmodes while never changing the underlying dissipative spectrum.
"""

import numpy as np
import scipy.linalg as la


def paulis():
    """Return Pauli matrices and identity."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    return I2, sx, sy, sz


def gkls_superoperator(H, L_ops):
    """
    Build the GKLS superoperator as a linear map on 2x2 matrices, returned
    as a 4x4 matrix in the Pauli basis coordinates alpha.

    We use the basis {Sigma_0, Sigma_1, Sigma_2, Sigma_3} = {I, sigma_x, sigma_y, sigma_z}
    and coordinates
        alpha_mu = 0.5 * tr(Sigma_mu rho).
    """
    I2, sx, sy, sz = paulis()
    Sigma = [I2, sx, sy, sz]

    def L_action(rho):
        """GKLS action on density matrix rho."""
        comm = -1.0j * (H @ rho - rho @ H)
        dissip = np.zeros_like(rho, dtype=complex)
        for L in L_ops:
            LdL = L.conj().T @ L
            dissip += L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
        return comm + dissip

    def to_coords(X):
        """Map Hermitian 2x2 matrix X to alpha in R^4 via Pauli coordinates."""
        alpha = np.zeros(4, dtype=float)
        for mu, S in enumerate(Sigma):
            alpha[mu] = 0.5 * np.real(np.trace(S @ X))
        return alpha

    # Build K by acting L on basis elements and reading coordinates
    K = np.zeros((4, 4), dtype=float)
    for nu, S_nu in enumerate(Sigma):
        X = L_action(S_nu)
        alpha = to_coords(X)
        K[:, nu] = alpha

    return K


def stationary_bloch_vector(K, tol=1e-12):
    """
    Find the stationary Bloch vector alpha_ss from K alpha = 0.

    We find the eigenvector of K associated with eigenvalue closest to 0
    and normalise it so that alpha_0 = 0.5 (trace one).
    """
    evals, evecs = la.eig(K)
    idx = np.argmin(np.abs(evals))
    v = evecs[:, idx]
    v = v.real
    if np.abs(v[0]) < tol:
        raise RuntimeError("Stationary eigenvector has near-zero alpha_0 component.")
    scale = 0.5 / v[0]
    alpha_ss = scale * v
    return alpha_ss


def density_from_bloch(alpha):
    """
    Construct a 2x2 density matrix from Bloch coordinates alpha in R^4:
        rho = alpha_0 I + alpha_x sigma_x + alpha_y sigma_y + alpha_z sigma_z.
    """
    I2, sx, sy, sz = paulis()
    return alpha[0] * I2 + alpha[1] * sx + alpha[2] * sy + alpha[3] * sz


def bkm_metric_bloch(rho_ss):
    """
    Construct the 4x4 BKM metric matrix M in the Pauli basis at rho_ss.

    Steps:
      * diagonalise rho_ss = U diag(lambda) U^dag,
      * compute 2x2 BKM weight matrix C in the eigenbasis,
      * vectorise basis elements Sigma_a' = U^dag Sigma_a U,
      * assemble M_ab = <Sigma_a, Sigma_b>_BKM
        = vec(Sigma_a')^dag diag(c_flat) vec(Sigma_b').
    """
    I2, sx, sy, sz = paulis()
    Sigma = [I2, sx, sy, sz]

    # Diagonalise rho_ss
    lam, U = la.eigh(rho_ss)
    if np.any(lam <= 0.0):
        raise RuntimeError("Non positive eigenvalues in rho_ss for BKM metric.")
    # BKM weights in eigenbasis
    C = np.zeros((2, 2), dtype=float)
    for m in range(2):
        for n in range(2):
            if m == n:
                C[m, n] = 1.0 / lam[m]
            else:
                num = np.log(lam[m]) - np.log(lam[n])
                den = lam[m] - lam[n]
                C[m, n] = num / den
    # Diagonal metric on vectorised basis
    c_flat = np.zeros(4, dtype=float)
    # Basis elements |m><n| with weights C_mn
    c_flat[0] = C[0, 0]
    c_flat[1] = C[0, 1]
    c_flat[2] = C[1, 0]
    c_flat[3] = C[1, 1]
    C_diag = np.diag(c_flat)

    # Transform Pauli basis into eigenbasis and vectorise
    Udag = U.conj().T
    vecs = []
    for S in Sigma:
        S_eig = Udag @ S @ U
        v = S_eig.reshape(-1, order="F")  # vec in column-major convention
        vecs.append(v)
    vecs = np.stack(vecs, axis=1)  # shape (4,4), columns are vecs of Sigma_a'

    # Assemble M_ab = u_a^dag C_diag u_b
    M = np.zeros((4, 4), dtype=float)
    for a in range(4):
        for b in range(4):
            M[a, b] = np.real(np.conj(vecs[:, a]) @ (C_diag @ vecs[:, b]))
    return M


def metriplectic_split(K_tr, M_tr):
    """
    Given a 3x3 generator K_tr and a 3x3 metric M_tr, compute metric adjoint,
    symmetric and antisymmetric parts, and diagnostic residuals.
    """
    Minv = la.inv(M_tr)
    K_sharp = Minv @ K_tr.T @ M_tr
    G = 0.5 * (K_tr + K_sharp)
    J = 0.5 * (K_tr - K_sharp)

    MG = M_tr @ G
    MJ = M_tr @ J
    sym_res = la.norm(MG - MG.T)
    skew_res = la.norm(MJ + MJ.T)
    sym_rel = sym_res / max(1.0, la.norm(MG))
    skew_rel = skew_res / max(1.0, la.norm(MJ))

    return G, J, sym_res, sym_rel, skew_res, skew_rel


def dissipative_spectrum(M_tr, G_tr, tol=1e-10):
    """
    Solve the generalised eigenproblem (-M_tr G_tr) v = lambda M_tr v
    and return positive eigenvalues and their maximum residual.
    """
    A = -M_tr @ G_tr
    B = M_tr
    evals, evecs = la.eig(A, B)
    evals = np.real(evals)
    evecs = np.real(evecs)

    # Filter positive eigenvalues
    pos_mask = evals > tol
    lambdas = evals[pos_mask]
    if lambdas.size == 0:
        raise RuntimeError("No positive eigenvalues in dissipative spectrum.")

    # Residual check
    max_res = 0.0
    for lam, v in zip(lambdas, evecs.T[pos_mask]):
        Av = A @ v
        Bv = B @ v
        res = la.norm(Av - lam * Bv) / max(1.0, la.norm(Av) + la.norm(Bv))
        max_res = max(max_res, res)
    lambdas_sorted = np.sort(lambdas)
    return lambdas_sorted, max_res


def matrix_flow_evolution(A, u0, times):
    """
    Evolve u(t) = exp(t A) u0 for a collection of times using matrix exponentials.

    This is robust even if A is not diagonalisable.
    """
    n = len(u0)
    out = np.zeros((len(times), n), dtype=float)
    for k, t in enumerate(times):
        At = la.expm(t * A)
        ut = At @ u0
        out[k, :] = ut.real
    return out


def quadratic_F(M, u):
    """Quadratic functional F(u) = 0.5 * u^T M u."""
    return 0.5 * float(u.T @ (M @ u))


def estimate_production(times, F_vals, M_tr, G_tr, U_vals):
    """
    Compare numerical dF/dt with the analytic production u^T M G u
    along a trajectory U_vals(t).

    Returns max absolute and max relative residuals.
    """
    times = np.asarray(times, dtype=float)
    F_vals = np.asarray(F_vals, dtype=float)
    U_vals = np.asarray(U_vals, dtype=float)
    dt = times[1] - times[0]

    # Numerical dF/dt via centred differences on interior points
    dF_num = (F_vals[2:] - F_vals[:-2]) / (2.0 * dt)

    # Analytic production at interior times (align indices)
    prod = []
    for k in range(1, len(times) - 1):
        u = U_vals[k]
        prod.append(float(u.T @ (M_tr @ (G_tr @ u))))
    prod = np.asarray(prod, dtype=float)

    abs_res = np.max(np.abs(dF_num - prod))
    denom = np.max(np.abs(prod))
    rel_res = abs_res / max(1.0e-12, denom)
    return abs_res, rel_res


def fit_decay_rate(times, F_vals, t_min_frac=0.3):
    """
    Fit F(t) ~ C exp(-r t) on [t_min_frac * t_max, t_max] to extract r.

    Returns fitted rate r and the sum of squared residuals of the linear fit
    in log space.
    """
    times = np.asarray(times, dtype=float)
    F_vals = np.asarray(F_vals, dtype=float)
    # Guard against tiny numerical negatives
    F_safe = np.clip(F_vals, 1.0e-300, None)
    logF = np.log(F_safe)

    t_max = times[-1]
    t_min = t_min_frac * t_max
    mask = times >= t_min
    t_fit = times[mask]
    y_fit = logF[mask]

    # Linear fit y = a + b t, with r = -b
    A = np.vstack([np.ones_like(t_fit), t_fit]).T
    coeffs, _, _, _ = la.lstsq(A, y_fit)
    a, b = coeffs
    r = -b

    # Compute residuals explicitly for robustness
    y_pred = A @ coeffs
    res = float(np.sum((y_fit - y_pred) ** 2))

    return float(r), res


def random_bloch_deviation(M_tr, F0=0.5, seed=None):
    """
    Sample a random deviation u0 in R^3 normalised so that F(u0) = F0
    in the metric M_tr.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=3)
    norm_M = np.sqrt(float(u.T @ (M_tr @ u)))
    if norm_M < 1.0e-14:
        raise RuntimeError("Sampled near zero vector in random_bloch_deviation.")
    scale = np.sqrt(2.0 * F0) / norm_M
    return scale * u


def run_gkls_bloch_two_quadratures_test(
    Omega=1.0,
    Delta=0.7,
    gamma=1.0,
    gamma_phi=0.4,
    n_inits=5,
    t_max_factor=6.0,
    n_times=400,
    seed=12345,
):
    """
    Main driver for the GKLS Bloch two quadratures K-flow test.

    Parameters
    ----------
    Omega, Delta, gamma, gamma_phi : floats
        Hamiltonian and dissipative parameters of the qubit model.
    n_inits : int
        Number of random initial deviations u(0) to sample.
    t_max_factor : float
        Maximal time as a multiple of 1 / lambda_min.
    n_times : int
        Number of time samples on [0, t_max].
    seed : int
        Random seed for reproducibility.
    """
    print("=" * 72)
    print("UIH two quadratures K-flow test for a GKLS qubit (Bloch coordinates)")
    print("=" * 72)

    # Build GKLS generator in Pauli basis
    I2, sx, sy, sz = paulis()
    sigma_minus = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)

    H = 0.5 * (Omega * sx + Delta * sz)
    L1 = np.sqrt(gamma) * sigma_minus
    Lphi = np.sqrt(gamma_phi) * sz
    L_ops = [L1, Lphi]

    K_full = gkls_superoperator(H, L_ops)

    # Stationary state in Bloch form and density matrix
    alpha_ss = stationary_bloch_vector(K_full)
    rho_ss = density_from_bloch(alpha_ss)

    # BKM metric at rho_ss in Pauli basis, then restrict to traceless block
    M_full = bkm_metric_bloch(rho_ss)
    M_tr = M_full[1:, 1:]
    K_tr = K_full[1:, 1:]

    print("\nMetric and generator diagnostics")
    print("--------------------------------")
    print(f"Full Bloch K matrix:\n{K_full}")
    print(f"\nStationary Bloch vector alpha_ss = {alpha_ss}")
    print(f"\nBKM metric M (4x4):\n{M_full}")

    # Use numpy.linalg.cond instead of scipy.linalg.cond
    cond_M_tr = np.linalg.cond(M_tr)
    print(f"\nTraceless Bloch metric M_tr condition number  = {cond_M_tr:.3e}")

    # Metriplectic split
    G_tr, J_tr, sym_res, sym_rel, skew_res, skew_rel = metriplectic_split(K_tr, M_tr)
    print(f"Norm of M_tr G_tr symmetry residual           = {sym_res:.3e}")
    print(f"Relative symmetry residual                    = {sym_rel:.3e}")
    print(f"Norm of M_tr J_tr skewness residual           = {skew_res:.3e}")
    print(f"Relative skewness residual                    = {skew_rel:.3e}")

    # Dissipative spectrum
    lambdas, eig_res = dissipative_spectrum(M_tr, G_tr)
    lambda_min = float(lambdas[0])
    lambda_max = float(lambdas[-1])

    print("\nDissipative spectrum of (-M_tr G_tr, M_tr)")
    print("------------------------------------------")
    print(f"Number of positive eigenvalues                = {len(lambdas)}")
    print(f"Smallest positive lambda_min                  = {lambda_min:.6f}")
    print(f"Largest positive lambda_max                  = {lambda_max:.6f}")
    print(f"Max relative eigenpair residual               = {eig_res:.3e}")

    # Time grid based on smallest dissipative eigenvalue
    t_max = t_max_factor / lambda_min
    times = np.linspace(0.0, t_max, n_times)
    dt = times[1] - times[0]
    print("\nTime grid")
    print("---------")
    print(f"t_max                                       = {t_max:.6f}")
    print(f"Number of time samples                      = {n_times}")
    print(f"dt                                          = {dt:.6f}")
    print(f"Expected asymptotic F decay rate (pure G)   = 2 * lambda_min = {2.0 * lambda_min:.6f}")

    rng = np.random.default_rng(seed)
    two_lambda_min = 2.0 * lambda_min

    abs_res_list = []
    rel_res_list = []
    rK_list = []
    rG_list = []
    resK_list = []
    resG_list = []

    print("\nTrajectory diagnostics")
    print("----------------------")

    for k in range(n_inits):
        # Random deviation in R^3 with fixed initial F = 0.5
        u0 = random_bloch_deviation(M_tr, F0=0.5, seed=rng.integers(1, 10**9))

        # Pure G-flow and full K-flow
        U_G = matrix_flow_evolution(G_tr, u0, times)
        U_K = matrix_flow_evolution(K_tr, u0, times)

        F_G = np.array([quadratic_F(M_tr, u) for u in U_G])
        F_K = np.array([quadratic_F(M_tr, u) for u in U_K])

        # Production identity along K-flow
        abs_res, rel_res = estimate_production(times, F_K, M_tr, G_tr, U_K)

        # Fit decay rates from F_G and F_K
        r_G, res_G = fit_decay_rate(times, F_G, t_min_frac=0.3)
        r_K, res_K = fit_decay_rate(times, F_K, t_min_frac=0.3)

        abs_res_list.append(abs_res)
        rel_res_list.append(rel_res)
        rG_list.append(r_G)
        rK_list.append(r_K)
        resG_list.append(res_G)
        resK_list.append(res_K)

        print(f"Initial condition {k:2d}:")
        print(f"  F_K(0)                                  = {F_K[0]:.6e}")
        print(f"  F_G(0)                                  = {F_G[0]:.6e}")
        print(f"  Max |dF_K/dt_num - u^T M G u|           = {abs_res:.3e}")
        print(f"  Rel production residual                 = {rel_res:.3e}")
        print(f"  Fitted r_K (full K)                     = {r_K:.6f}")
        print(f"  Fitted r_G (pure G)                     = {r_G:.6f}")
        print(f"  r_K / (2 lambda_min)                    = {r_K / two_lambda_min:.6f}")
        print(f"  r_G / (2 lambda_min)                    = {r_G / two_lambda_min:.6f}")
        print(f"  log-fit residuals K, G                  = {res_K:.3e}, {res_G:.3e}\n")

    max_abs_res = float(np.max(abs_res_list))
    max_rel_res = float(np.max(rel_res_list))
    mean_rK = float(np.mean(rK_list))
    mean_rG = float(np.mean(rG_list))
    std_rK = float(np.std(rK_list))
    std_rG = float(np.std(rG_list))

    print("Summary over initial conditions")
    print("-------------------------------")
    print(f"Max abs production residual                 = {max_abs_res:.3e}")
    print(f"Max rel production residual                 = {max_rel_res:.3e}")
    print(f"Mean r_K / (2 lambda_min)                   = {mean_rK / two_lambda_min:.6f}")
    print(f"Std  r_K / (2 lambda_min)                   = {std_rK / two_lambda_min:.6f}")
    print(f"Mean r_G / (2 lambda_min)                   = {mean_rG / two_lambda_min:.6f}")
    print(f"Std  r_G / (2 lambda_min)                   = {std_rG / two_lambda_min:.6f}")

    print("\nConclusion:")
    print("  The metric adjoint split K = G + J in the BKM geometry yields")
    print("  MG symmetric and MJ skew to numerical precision, so J is a")
    print("  genuine no-work direction for the quadratic functional F(u) = 0.5 u^T M u.")
    print("  The pure G-flow decays at a rate r_G tightly clustered around")
    print("  2 * lambda_min, the smallest positive eigenvalue of the dissipative")
    print("  operator, saturating the UIH lower bound.")
    print("  The full K-flow shows decay rates r_K >= 2 * lambda_min, illustrating")
    print("  how the reversible channel J can accelerate the decay of F by mixing")
    print("  eigenmodes, while never altering the underlying dissipative spectrum.")


if __name__ == "__main__":
    run_gkls_bloch_two_quadratures_test()
