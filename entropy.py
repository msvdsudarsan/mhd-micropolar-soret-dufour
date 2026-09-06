"""
Entropy generation and Bejan number analysis for the MHD micropolar
Soret-Dufour boundary-layer model (model.py).

The dimensionless local entropy generation number is derived in the
manuscript (Section 4.9) from the classical Bejan-type local volumetric
entropy generation rate, extended for combined heat and mass transfer with
Soret-Dufour cross-diffusion, and deliberately WITHOUT a separate magnetic
(Joule-heating) entropy term -- consistent with the model's own energy
equation (Eq. theta), which contains no Ohmic source. The friction term
uses the same Newtonian viscous-dissipation combination, nu*(du/dy)^2, that
already appears in the model's own dimensional energy equation, rather than
a micropolar-enhanced (mu+kappa) version, for the same consistency reason.

The result (verified symbolically and numerically against the raw
dimensional definition in derive_entropy_v2.py / derive_mass_cross.py):

    Ns(eta) = theta'(eta)^2
              + (Pr*Ec/Omega) * f''(eta)^2
              + (L*Omega_C/Omega^2) * phi'(eta)^2
              + (L/Omega) * phi'(eta)*theta'(eta)

    Be(eta) = Ns_heat(eta) / Ns(eta)          (Bejan number)

New dimensionless groups (not present in the base model):
    Omega   = (Tf - T_inf)/T_inf     dimensionless wall-to-ambient temperature difference
    Omega_C = (Cw - C_inf)/C_inf     dimensionless wall-to-ambient concentration difference
    L       = R*Dm*(Cw-C_inf)/k      diffusive entropy-generation parameter
"""
import numpy as np
from model import Params, solve_case, wall_gradients, with_param


def entropy_profile(sol, P: Params, eta_grid, Omega=1.0, Omega_C=1.0, L=0.5):
    """Return (Ns_heat, Ns_fric, Ns_mass, Ns_cross, Ns_total, Be) arrays
    evaluated at the points of eta_grid, for a solved BVP solution `sol`."""
    y = sol.sol(eta_grid)
    fpp = y[2]
    thp = y[6]
    php = y[8]

    Ns_heat = thp**2
    Ns_fric = (P.Pr * P.Ec / Omega) * fpp**2
    Ns_mass = (L * Omega_C / Omega**2) * php**2
    Ns_cross = (L / Omega) * php * thp
    Ns_total = Ns_heat + Ns_fric + Ns_mass + Ns_cross
    Be = Ns_heat / Ns_total
    return Ns_heat, Ns_fric, Ns_mass, Ns_cross, Ns_total, Be


def wall_entropy_and_bejan(P: Params, eta_inf=10.0, n_init=400,
                            Omega=1.0, Omega_C=1.0, L=0.5):
    """Convenience: solve the base case and return wall-value (eta=0)
    entropy-generation components and the wall Bejan number."""
    sol = solve_case(P, eta_inf, n_init)
    Ns_heat, Ns_fric, Ns_mass, Ns_cross, Ns_total, Be = entropy_profile(
        sol, P, np.array([0.0]), Omega, Omega_C, L)
    return (Ns_heat[0], Ns_fric[0], Ns_mass[0], Ns_cross[0], Ns_total[0], Be[0])


if __name__ == "__main__":
    P = Params()
    print("Wall-value entropy-generation decomposition at the base case")
    print("(Omega=1.0, Omega_C=1.0, L=0.5):")
    Nh, Nf, Nm, Nc, Nt, Be = wall_entropy_and_bejan(P)
    print(f"  Ns_heat(0)  = {Nh:.6f}")
    print(f"  Ns_fric(0)  = {Nf:.6f}")
    print(f"  Ns_mass(0)  = {Nm:.6f}")
    print(f"  Ns_cross(0) = {Nc:.6f}")
    print(f"  Ns_total(0) = {Nt:.6f}")
    print(f"  Be(0)       = {Be:.6f}")
