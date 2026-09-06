"""
Similarity-equation model for:

  "Coupled Heat and Mass Transfer in MHD Micropolar Fluid Flow past a
   Vertical Plate with Thermal Dispersion, Soret-Dufour Effects and
   Chemical Reaction" (Subbarao & Madhyannapu)

This is the primary collocation implementation used to generate every
table and figure in the manuscript. It is cross-checked in Table 2
against shooting_crosscheck.py, an algorithmically independent
shooting-method implementation (forward integration + nonlinear
root-find, rather than global collocation).

Energy-equation coefficient (Section 2.1 / Appendix A of the manuscript).
The coefficient multiplying theta'' is (1+S*fp), obtained from an
independent symbolic re-derivation of the similarity reduction and
matching the base study's own stated equation (Rao & Umamaheswara and
Koteswara Rao 2017, Eq. 2.11) exactly.

State vector:
    y[0]=f   y[1]=f'   y[2]=f''
    y[3]=g   y[4]=g'
    y[5]=th  y[6]=th'
    y[7]=ph  y[8]=ph'

Governing similarity equations (Eqs. 10-13 of the manuscript):
    (1+K) f''' + 3 f f'' - 2 f'^2 - M f' + K g' + theta + Nc*phi = 0
    lambda g'' - K B (2g + f'') - f' g + 3 f g' = 0
    (1+S f') th'' + S th th' f'' + 3 Pr f th' + Pr Ec f''^2 + Pr Du ph'' + Q0 e^{-eta} = 0
    ph'' + 3 Sc f ph' - Sc Kr ph + Sc Sr th'' = 0

Boundary conditions (Eq. 14):
    eta=0:    f=0, f'=0, g=0, th'(0) = -Bi(1-th(0)), ph=1
    eta->inf: f'=0, g=0, th=0, ph=0
"""

from dataclasses import dataclass, replace
import numpy as np
from scipy.integrate import solve_bvp


@dataclass
class Params:
    K: float = 0.5     # micropolar material parameter
    Pr: float = 0.7    # Prandtl number
    B: float = 0.5     # material parameter
    lam: float = 0.5   # spin-gradient-viscosity ratio
    S: float = 0.1     # thermal dispersion parameter
    Ec: float = 0.1    # Eckert number
    Bi: float = 0.2    # Biot number (convective BC)
    Q0: float = 1.0    # internal heat-generation constant (was "C" in the .m file)
    M: float = 0.5     # magnetic parameter
    Nc: float = 0.5    # concentration buoyancy-ratio parameter
    Sc: float = 0.6    # Schmidt number
    Kr: float = 0.5    # chemical-reaction parameter
    Sr: float = 0.5    # Soret number
    Du: float = 0.2    # Dufour number


def odes(eta, y, P: Params):
    """Right-hand side of the 9th-order first-order system, vectorised
    over the mesh (eta and each row of y are length-n arrays)."""
    f, fp, fpp, g, gp, th, thp, ph, php = y

    dydx = np.empty_like(y)
    dydx[0] = fp
    dydx[1] = fpp
    dydx[2] = -1.0 / (1 + P.K) * (
        3 * f * fpp - 2 * fp**2 - P.M * fp + th + P.Nc * ph + P.K * gp
    )

    dydx[3] = gp
    dydx[4] = 1.0 / P.lam * (P.K * P.B * (2 * g + fpp) + fp * g - 3 * f * gp)

    dydx[5] = thp
    dydx[7] = php

    # theta'' and phi'' solved jointly, Eqs. (18)-(19) of the manuscript.
    # The dispersion cross-term S*f''*theta' follows from the product-rule
    # expansion of d/dy(alpha_y dT/dy) derived in Appendix A.
    R1 = -(P.S * fpp * thp + 3 * P.Pr * f * thp + P.Pr * P.Ec * fpp**2
           + P.Q0 * np.exp(-eta))
    R2 = -3 * P.Sc * f * php + P.Sc * P.Kr * ph

    a11 = (1 + P.S * fp)
    a12 = P.Pr * P.Du
    a21 = P.Sc * P.Sr
    a22 = 1.0

    detA = a11 * a22 - a12 * a21   # = Delta of Eq. (19); > 0 for every
                                    # case in this paper, see Sec. 3 for
                                    # the explicit bound on Pr*Sc*Sr*Du.
    dydx[6] = (R1 * a22 - a12 * R2) / detA   # theta''
    dydx[8] = (a11 * R2 - a21 * R1) / detA   # phi''

    return dydx


def bcs(ya, yb, P: Params):
    return np.array([
        ya[0],                          # f(0) = 0
        ya[1],                          # f'(0) = 0
        ya[3],                          # g(0) = 0
        ya[6] + P.Bi * (1 - ya[5]),     # theta'(0) = -Bi(1-theta(0))
        ya[7] - 1,                      # phi(0) = 1
        yb[1],                          # f'(inf) = 0
        yb[3],                          # g(inf) = 0
        yb[5],                          # theta(inf) = 0
        yb[7],                          # phi(inf) = 0
    ])


def initial_guess(x):
    """Smooth decaying guess, same functional form as guess() in the
    .m file (f' ~ e^-eta, g ~ eta*e^-eta, theta/phi ~ e^-eta)."""
    y0 = np.zeros((9, x.size))
    y0[0] = 1 - np.exp(-x)
    y0[1] = np.exp(-x)
    y0[2] = -np.exp(-x)
    y0[3] = x * np.exp(-x)
    y0[4] = np.exp(-x) - x * np.exp(-x)
    y0[5] = np.exp(-x)
    y0[6] = -np.exp(-x)
    y0[7] = np.exp(-x)
    y0[8] = -np.exp(-x)
    return y0


def solve_case(P: Params, eta_inf: float = 10.0, n_init: int = 400,
               tol: float = 1e-8, max_nodes: int = 50000):
    """Solve one parameter set. n_init seeds the mesh; solve_bvp's own
    adaptive refinement decides how far to grow it, so the converged
    node count won't match the shooting method's own internal steps --
    only the converged wall gradients are expected to agree (Table 2,
    Section 3.1)."""
    x = np.linspace(0, eta_inf, n_init)
    y_guess = initial_guess(x)
    sol = solve_bvp(lambda eta, y: odes(eta, y, P),
                     lambda ya, yb: bcs(ya, yb, P),
                     x, y_guess, tol=tol, max_nodes=max_nodes, verbose=0)
    if not sol.success:
        raise RuntimeError(f"solve_bvp did not converge: {sol.message}")
    return sol


def wall_gradients(sol):
    """f''(0), -g'(0), -theta'(0), theta(0), -phi'(0) -- skin friction,
    wall couple stress, local Nusselt indicator, wall temperature, and
    local Sherwood indicator. -g'(0) is read directly off the state
    vector per Eq. (20); no extra scaling is applied anywhere here."""
    y0 = sol.sol(0.0)
    fpp0 = y0[2]
    mgp0 = -y0[4]
    mthp0 = -y0[6]
    theta0 = y0[5]
    mphp0 = -y0[8]
    return fpp0, mgp0, mthp0, theta0, mphp0


def with_param(P: Params, **kwargs) -> Params:
    """P2 = with_param(P, M=1.0) instead of copy+mutate by hand."""
    return replace(P, **kwargs)
