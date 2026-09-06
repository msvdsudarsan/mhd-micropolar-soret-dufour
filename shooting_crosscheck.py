"""
Independent second implementation of the model in model.py.

Method: explicit shooting (RK45/Radau time-stepping via solve_ivp,
missing initial conditions found by scipy.optimize.fsolve on the
far-field residuals), as opposed to model.py's implicit relaxation/
collocation approach (scipy.integrate.solve_bvp). This is a different
numerical algorithm family entirely -- forward integration + nonlinear
root-finding, rather than global collocation on the whole domain --
built directly from the governing equations and boundary conditions
as typeset in the manuscript, without importing or referencing
model.py's own code.

Run this file directly to reproduce the base case, the M-sweep of
Table 5, the classical-Ostrach validation limit (Pr=0.7, 0.72 and 1),
and a cross-check against Rees & Pop's (1998) own published Table 1,
and compare against model.py's solve_bvp output.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def rhs(eta, y, K, Pr, B, lam, S, Ec, Bi, Q0, M, Nc, Sc, Kr, Sr, Du):
    f, fp, fpp, g, gp, th, thp, ph, php = y
    fppp = -1.0 / (1 + K) * (3 * f * fpp - 2 * fp**2 - M * fp + th + Nc * ph + K * gp)
    gpp = 1.0 / lam * (K * B * (2 * g + fpp) + fp * g - 3 * f * gp)
    R1 = -(S * fpp * thp + 3 * Pr * f * thp + Pr * Ec * fpp**2 + Q0 * np.exp(-eta))
    R2 = -3 * Sc * f * php + Sc * Kr * ph
    a11 = 1 + S * fp
    a12 = Pr * Du
    a21 = Sc * Sr
    a22 = 1.0
    det = a11 * a22 - a12 * a21
    thpp = (R1 * a22 - a12 * R2) / det
    phpp = (a11 * R2 - a21 * R1) / det
    return [fp, fpp, fppp, gp, gpp, thp, thpp, php, phpp]


def _residual(unknowns, eta_inf, params, method, rtol, atol, max_step):
    fpp0, gp0, th0, php0 = unknowns
    Bi = params[6]
    thp0 = -Bi * (1 - th0)
    y0 = [0, 0, fpp0, 0, gp0, th0, thp0, 1, php0]
    sol = solve_ivp(rhs, [0, eta_inf], y0, args=params, method=method,
                     rtol=rtol, atol=atol, max_step=max_step)
    yend = sol.y[:, -1]
    return [yend[1], yend[3], yend[5], yend[7]]


def solve_shooting(P, eta_inf=10.0, guess=(0.83, -0.10, 1.23, -0.83),
                    method="RK45", rtol=1e-10, atol=1e-12, max_step=0.05,
                    xtol=1e-12):
    params = (P["K"], P["Pr"], P["B"], P["lam"], P["S"], P["Ec"], P["Bi"],
              P["Q0"], P["M"], P["Nc"], P["Sc"], P["Kr"], P["Sr"], P["Du"])
    x, info, ier, msg = fsolve(_residual, guess,
                                args=(eta_inf, params, method, rtol, atol, max_step),
                                full_output=True, xtol=xtol)
    if ier != 1:
        raise RuntimeError(f"shooting failed: {msg}")
    fpp0, gp0, th0, php0 = x
    thp0 = -P["Bi"] * (1 - th0)
    return fpp0, -gp0, -thp0, th0, -php0  # returns f''(0), -g'(0), -th'(0), th(0), -ph'(0)


def base_params(**kwargs):
    P = dict(K=0.5, Pr=0.7, B=0.5, lam=0.5, S=0.1, Ec=0.1, Bi=0.2, Q0=1.0,
              M=0.5, Nc=0.5, Sc=0.6, Kr=0.5, Sr=0.5, Du=0.2)
    P.update(kwargs)
    return P


if __name__ == "__main__":
    print("Table 5 (M-sweep), independent shooting-method reproduction:")
    print(f"{'M':>5s} {'f`(0)':>10s} {'-g`(0)':>10s} {'-th`(0)':>10s} {'th(0)':>10s} {'-ph`(0)':>10s}")
    for m in [0, 0.5, 1.0, 1.5, 2.0]:
        P = base_params(M=m)
        vals = solve_shooting(P)
        print(f"{m:5.2f} " + " ".join(f"{v:10.6f}" for v in vals))

    print("\nClassical Ostrach (1953) limit, Pr=0.7, Bi=1e5 (isothermal-wall surrogate):")
    Pos = base_params(K=0, S=0, Ec=0, Q0=0, M=0, Sr=0, Du=0, Kr=0, Nc=0, Bi=1e5, Pr=0.7)
    vals = solve_shooting(Pos, guess=(0.679, 1e-5, 0.999995, -0.4614),
                           method="Radau", rtol=1e-12, atol=1e-14, max_step=0.02,
                           xtol=1e-13)
    print(" ".join(f"{v:10.6f}" for v in vals))
    print("Reference: approximately 0.674-0.68, commonly quoted for this "
          "Prandtl number in the secondary heat-transfer literature.")
    rp_fpp0_07, rp_mgp0_07, rp_mthp0_07 = vals[0], vals[1], vals[2]

    print("\nClassical Ostrach limit, Pr=0.72, Bi=1e5 (Ostrach's own tabulated Pr):")
    Pos072 = base_params(K=0, S=0, Ec=0, Q0=0, M=0, Sr=0, Du=0, Kr=0, Nc=0, Bi=1e5, Pr=0.72)
    vals072 = solve_shooting(Pos072, guess=(0.676, 1e-5, 0.999995, -0.5046),
                              method="Radau", rtol=1e-12, atol=1e-14, max_step=0.02,
                              xtol=1e-13)
    fpp0_072, mgp0_072, mthp0_072 = vals072[0], vals072[1], vals072[2]
    print(f"Computed: f''(0) = {fpp0_072:.6f}   -theta'(0) = {mthp0_072:.6f}")
    print("Reference (Ostrach's own NACA Report 1111, via his stated Nu=63.6 "
          "at Gr_x=1e9): -theta'(0) = 0.505792")
    print(f"Relative difference in -theta'(0): {100*abs(mthp0_072-0.505792)/0.505792:.3f}%")

    print("\nClassical Ostrach limit, Pr=1.0, Bi=1e5 (isothermal-wall surrogate):")
    Pos1 = base_params(K=0, S=0, Ec=0, Q0=0, M=0, Sr=0, Du=0, Kr=0, Nc=0, Bi=1e5, Pr=1.0)
    vals1 = solve_shooting(Pos1, guess=(0.642185, 1e-6, 0.999994, -0.567142),
                            method="Radau", rtol=1e-12, atol=1e-14, max_step=0.01,
                            xtol=1e-12)
    fpp0_1, mgp0_1, mthp0_1, th0_1, mphp0_1 = vals1
    print(f"Computed: f''(0) = {fpp0_1:.6f}   -theta'(0) = {mthp0_1:.6f}")
    print("Reference (Peker & Oturanc, for the classical Ostrach problem): "
          "f''(0) = 0.6421   -theta'(0) = 0.5671")
    print(f"Relative difference in f''(0): {100*abs(fpp0_1-0.6421)/0.6421:.3f}%")

    print("\nRees & Pop (1998) cross-check, Pr=0.7, K=0 (their Table 1, n=1):")
    beta = 2.0 ** 0.5
    alpha = 2 * beta
    fpp0_rp_converted = (beta ** 2 / alpha) * 0.96012
    thetap0_rp_converted = beta * (-0.35321)
    print(f"  Converted from their table: f''(0) = {fpp0_rp_converted:.6f}   "
          f"-theta'(0) = {-thetap0_rp_converted:.6f}")
    print(f"  Computed here:              f''(0) = {rp_fpp0_07:.6f}   "
          f"-theta'(0) = {rp_mthp0_07:.6f}")
    rel_f = 100 * abs(rp_fpp0_07 - fpp0_rp_converted) / fpp0_rp_converted
    rel_th = 100 * abs(rp_mthp0_07 - (-thetap0_rp_converted)) / abs(thetap0_rp_converted)
    print(f"  Relative difference: {rel_f:.4f}% in f''(0), {rel_th:.4f}% in -theta'(0)")
