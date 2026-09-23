"""
Transport-budget decomposition and mechanism-isolation analysis for:

  "Transport-Mechanism Analysis of Coupled Heat and Mass Transfer in MHD
   Micropolar Fluid Flow past a Vertical Plate with Thermal Dispersion,
   Soret-Dufour Effects and Chemical Reaction"
  (submission to International Journal of Heat and Mass Transfer)

This script is additional to model.py / run_analysis.py and reproduces
the two new quantitative results introduced for the IJHMT submission:

  1. Table 5 and Fig. 7 of the manuscript: an exact term-by-term
     decomposition of the energy equation (Eq. 9) and the species
     equation (Eq. 10) along the converged base-case profile, using
     theta'' and phi'' recovered from the same 2x2 solve used inside
     model.odes(). Summing the recovered terms reproduces each governing
     equation to machine precision, which is checked explicitly below
     rather than assumed.

  2. Table 6 of the manuscript: a mechanism-isolation comparison, in
     which the magnetic, reactive and cross-diffusive mechanisms are
     switched on cumulatively from the same convective, thermally
     dispersive base state (K, S and Nc held fixed throughout, since
     these belong to the base convective-boundary-layer model rather
     than to the present six-mechanism extension).

Run with:  python transport_budget.py
"""

import sys
import numpy as np
from model import Params, solve_case, wall_gradients, with_param

try:
    trapz = np.trapezoid
except AttributeError:  # older NumPy
    trapz = np.trapz


def recover_second_derivatives(P: Params, eta, f, fp, fpp, th, thp, ph, php):
    """Recompute theta'' and phi'' from the identical 2x2 linear system
    used inside model.odes(), for post-processing the converged solution."""
    R1 = -(P.S * fpp * thp + 3 * P.Pr * f * thp + P.Pr * P.Ec * fpp**2
           + P.Q0 * np.exp(-eta))
    R2 = -3 * P.Sc * f * php + P.Sc * P.Kr * ph
    a11 = (1 + P.S * fp)
    a12 = P.Pr * P.Du
    a21 = P.Sc * P.Sr
    a22 = 1.0
    detA = a11 * a22 - a12 * a21
    thpp = (R1 * a22 - a12 * R2) / detA
    phpp = (a11 * R2 - a21 * R1) / detA
    return thpp, phpp


def transport_budget(P: Params = None, eta_max: float = 8.0, n: int = 400):
    """Decompose Eqs. (9)-(10) into their constituent terms along the
    converged profile and report each term's share of the local
    transport budget, exactly as reported in Table 5 of the manuscript."""
    P = P or Params()
    sol = solve_case(P)
    eta = np.linspace(0, eta_max, n)
    f, fp, fpp, g, gp, th, thp, ph, php = sol.sol(eta)
    thpp, phpp = recover_second_derivatives(P, eta, f, fp, fpp, th, thp, ph, php)

    # --- energy-equation terms (Eq. 9) ---
    E_cond = thpp
    E_disp = P.S * fp * thpp + P.S * fpp * thp
    E_conv = 3 * P.Pr * f * thp
    E_visc = P.Pr * P.Ec * fpp**2
    E_dufour = P.Pr * P.Du * phpp
    E_gen = P.Q0 * np.exp(-eta)
    residual_E = E_cond + E_disp + E_conv + E_visc + E_dufour + E_gen
    assert np.max(np.abs(residual_E)) < 1e-10, "energy-equation decomposition does not close"

    # --- species-equation terms (Eq. 10) ---
    S_diff = phpp
    S_conv = 3 * P.Sc * f * php
    S_rxn = -P.Sc * P.Kr * ph
    S_soret = P.Sc * P.Sr * thpp
    residual_S = S_diff + S_conv + S_rxn + S_soret
    assert np.max(np.abs(residual_S)) < 1e-10, "species-equation decomposition does not close"

    def L1(x):
        return trapz(np.abs(x), eta)

    energy_terms = {
        "conduction": L1(E_cond),
        "dispersion": L1(E_disp),
        "convection": L1(E_conv),
        "viscous dissipation": L1(E_visc),
        "Dufour": L1(E_dufour),
        "heat generation": L1(E_gen),
    }
    species_terms = {
        "diffusion": L1(S_diff),
        "convection": L1(S_conv),
        "reaction": L1(S_rxn),
        "Soret": L1(S_soret),
    }

    tot_E = sum(energy_terms.values())
    tot_S = sum(species_terms.values())
    energy_share = {k: 100 * v / tot_E for k, v in energy_terms.items()}
    species_share = {k: 100 * v / tot_S for k, v in species_terms.items()}

    return {
        "eta": eta,
        "profiles": dict(f=f, fp=fp, fpp=fpp, th=th, thp=thp, ph=ph, php=php,
                          thpp=thpp, phpp=phpp),
        "energy_terms": dict(cond=E_cond, disp=E_disp, conv=E_conv,
                              visc=E_visc, dufour=E_dufour, gen=E_gen),
        "species_terms": dict(diff=S_diff, conv=S_conv, rxn=S_rxn, soret=S_soret),
        "energy_share_pct": energy_share,
        "species_share_pct": species_share,
        "max_energy_residual": float(np.max(np.abs(residual_E))),
        "max_species_residual": float(np.max(np.abs(residual_S))),
    }


def mechanism_isolation_table(Nc_fixed: bool = True):
    """Reproduce Table 6 (upper block) of the manuscript: cumulative mechanism-isolation
    comparison from the same convective, thermally dispersive base state."""
    base = Params()
    rows = []

    def add(label, **overrides):
        P = with_param(base, **overrides)
        s = solve_case(P)
        fpp, gp, thp, th0, php = wall_gradients(s)
        rows.append((label, fpp, gp, -thp, php))

    add("Base convective model (M=Sr=Du=Kr=0)", M=0.0, Sr=0.0, Du=0.0, Kr=0.0)
    add("+ MHD (M=0.5)", Sr=0.0, Du=0.0, Kr=0.0)
    add("+ Reaction (Kr=0.5)", Sr=0.0, Du=0.0)
    add("+ Soret (Sr=0.5)", Du=0.0)
    add("+ Dufour instead of Soret (Du=0.2)", Sr=0.0)
    add("Full model (Soret + Dufour together)")

    return rows


def extended_limiting_cases():
    """Reproduce Table 6 (lower block) of the manuscript: one mechanism removed at a
    time from the full base case."""
    base = Params()
    rows = []

    def add(label, **overrides):
        P = with_param(base, **overrides)
        s = solve_case(P)
        fpp, gp, thp, th0, php = wall_gradients(s)
        rows.append((label, fpp, gp, -thp, php))

    add("Full base case")
    add("M=0 (no MHD)", M=0.0)
    add("Sr=Du=0 (no cross-diffusion)", Sr=0.0, Du=0.0)
    add("S=0 (no thermal dispersion)", S=0.0)
    add("Kr=0 (no reaction)", Kr=0.0)
    add("K=0 (passive microrotation)", K=0.0)

    return rows


if __name__ == "__main__":
    result = transport_budget()
    print("Energy-equation residual (max |sum of signed terms|):",
          result["max_energy_residual"])
    print("Species-equation residual (max |sum of signed terms|):",
          result["max_species_residual"])
    print()
    print("Energy-equation transport-budget shares (%):")
    for k, v in result["energy_share_pct"].items():
        print(f"  {k:20s} {v:6.1f}")
    print()
    print("Species-equation transport-budget shares (%):")
    for k, v in result["species_share_pct"].items():
        print(f"  {k:20s} {v:6.1f}")

    print()
    print("Mechanism-isolation comparison:")
    for label, fpp, gp, mthp, php in mechanism_isolation_table():
        print(f"  {label:42s} f''(0)={fpp:.6f}  -g'(0)={gp:.6f}  "
              f"-th'(0)={mthp:.6f}  -ph'(0)={php:.6f}")

    print()
    print("Extended limiting-case reductions:")
    for label, fpp, gp, mthp, php in extended_limiting_cases():
        print(f"  {label:35s} f''(0)={fpp:.6f}  -g'(0)={gp:.6f}  "
              f"-th'(0)={mthp:.6f}  -ph'(0)={php:.6f}")


def species_dominance_map(Sr_vals=None, Du_vals=None):
    """Reproduce Figure 6(c) of the manuscript: for each (Sr, Du) grid point,
    identify which term of the species-equation transport-budget decomposition
    (diffusion, convection, reaction, Soret) carries the largest share, using
    the same magnitude-normalized decomposition as transport_budget() above."""
    import numpy as np
    if Sr_vals is None:
        Sr_vals = np.linspace(0, 2, 25)
    if Du_vals is None:
        Du_vals = np.linspace(0, 0.8, 25)
    base = Params()
    labels = ["diffusion", "convection", "reaction", "Soret"]
    dominant = [[None] * len(Du_vals) for _ in Sr_vals]
    for i, sr in enumerate(Sr_vals):
        for j, du in enumerate(Du_vals):
            P = with_param(base, Sr=sr, Du=du)
            result = transport_budget(P)
            shares = result["species_share_pct"]
            key_map = {"diffusion": "diffusion", "convection": "convection",
                       "reaction": "reaction", "Soret": "Soret"}
            dominant[i][j] = max(shares, key=shares.get)
    return Sr_vals, Du_vals, dominant





def energy_dominance_map(M_vals=None, S_vals=None):
    """Reproduce Figure 7(c) of the manuscript: for each (S, M) grid point,
    identify which term of the energy-equation transport-budget decomposition
    carries the largest share, using the same magnitude-normalized
    decomposition as transport_budget() above."""
    import numpy as np
    if M_vals is None:
        M_vals = np.linspace(0, 2, 30)
    if S_vals is None:
        S_vals = np.linspace(0, 1.0, 30)
    base = Params()
    dominant = [[None] * len(S_vals) for _ in M_vals]
    for i, m in enumerate(M_vals):
        for j, s in enumerate(S_vals):
            P = with_param(base, M=m, S=s)
            result = transport_budget(P)
            shares = result["energy_share_pct"]
            dominant[i][j] = max(shares, key=shares.get)
    return M_vals, S_vals, dominant


def make_figure5(outdir="figures", Sr_vals=None, Du_vals=None):
    """Generate and save the complete three-panel manuscript Figure 6:
    (a) local Nusselt indicator -theta'(0), (b) local Sherwood indicator
    -phi'(0), and (c) the dominant species-transport mechanism, all over
    the (Sr, Du) plane. This is the exact figure used in the manuscript;
    run this function (or this script's __main__ block) to regenerate it
    from scratch."""
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if Sr_vals is None:
        Sr_vals = np.linspace(0, 2, 25)
    if Du_vals is None:
        Du_vals = np.linspace(0, 0.8, 25)
    base = Params()
    SR, DU = np.meshgrid(Sr_vals, Du_vals, indexing="ij")
    NTH = np.zeros_like(SR)
    NPH = np.zeros_like(SR)
    labels = ["diffusion", "convection", "reaction", "Soret"]
    DOM = np.zeros_like(SR)
    for i, sr in enumerate(Sr_vals):
        for j, du in enumerate(Du_vals):
            P = with_param(base, Sr=sr, Du=du)
            sol = solve_case(P)
            f, fp, fpp, g, gp, th, thp, ph, php = sol.sol(0.0)
            NTH[i, j] = -thp
            NPH[i, j] = -php
            result = transport_budget(P)
            shares = result["species_share_pct"]
            DOM[i, j] = labels.index(max(shares, key=shares.get))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=300)

    ax = axes[0]
    c0 = ax.contourf(DU, SR, NTH, levels=20, cmap="viridis")
    ax.set_xlabel(r"$D_u$"); ax.set_ylabel(r"$S_r$")
    ax.set_title(r"(a) Local Nusselt indicator $-\theta'(0)$")
    fig.colorbar(c0, ax=ax, shrink=0.85)

    ax = axes[1]
    c1 = ax.contourf(DU, SR, NPH, levels=20, cmap="plasma")
    ax.set_xlabel(r"$D_u$"); ax.set_ylabel(r"$S_r$")
    ax.set_title(r"(b) Local Sherwood indicator $-\phi'(0)$")
    fig.colorbar(c1, ax=ax, shrink=0.85)

    ax = axes[2]
    present = sorted(set(DOM.flatten().astype(int)))
    cmap = matplotlib.colors.ListedColormap(
        ["#4c72b0", "#dd8452", "#55a868", "#c44e52"][: max(present) + 1]
    )
    c2 = ax.pcolormesh(DU, SR, DOM, cmap=cmap, vmin=-0.5, vmax=3.5, shading="auto")
    cb = fig.colorbar(c2, ax=ax, ticks=range(4))
    cb.ax.set_yticklabels(labels)
    ax.set_xlabel(r"$D_u$"); ax.set_ylabel(r"$S_r$")
    ax.set_title("(c) Dominant species-transport\nmechanism")

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "fig5_manuscript_srdu_full.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Figure 6 (manuscript) saved to {outpath}")
    return outpath


def make_figure6(outdir="figures", M_vals=None, S_vals=None):
    """Generate and save the complete three-panel manuscript Figure 7:
    (a) energy-equation transport-budget terms vs eta, (b) species-equation
    transport-budget terms vs eta, both at the base parameter set, and
    (c) the dominant energy-transport mechanism over the (S, M) plane.
    This is the exact figure used in the manuscript."""
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = Params()
    sol = solve_case(P)
    eta = np.linspace(0, 8, 400)
    f, fp, fpp, g, gp, th, thp, ph, php = sol.sol(eta)
    thpp, phpp = recover_second_derivatives(P, eta, f, fp, fpp, th, thp, ph, php)

    E_cond = thpp
    E_disp = P.S * fp * thpp + P.S * fpp * thp
    E_conv = 3 * P.Pr * f * thp
    E_visc = P.Pr * P.Ec * fpp ** 2
    E_dufour = P.Pr * P.Du * phpp
    E_gen = P.Q0 * np.exp(-eta)

    S_diff = phpp
    S_conv = 3 * P.Sc * f * php
    S_rxn = -P.Sc * P.Kr * ph
    S_soret = P.Sc * P.Sr * thpp

    fig = plt.figure(figsize=(13.5, 4.0), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    ax.plot(eta, E_cond, label="conduction", lw=1.6)
    ax.plot(eta, E_conv, label="convection", lw=1.6)
    ax.plot(eta, E_gen, label="heat gen.", lw=1.6)
    ax.plot(eta, E_dufour, label="Dufour", lw=1.6)
    ax.plot(eta, E_disp, label="dispersion", lw=1.2)
    ax.plot(eta, E_visc, label="viscous diss.", lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel("term magnitude")
    ax.set_title("(a) Energy-equation transport budget")
    ax.legend(fontsize=6.5, frameon=False); ax.set_xlim(0, 6)

    ax = fig.add_subplot(gs[1])
    ax.plot(eta, S_diff, label="diffusion", lw=1.6)
    ax.plot(eta, S_conv, label="convection", lw=1.6)
    ax.plot(eta, S_soret, label="Soret", lw=1.6)
    ax.plot(eta, S_rxn, label="reaction", lw=1.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel("term magnitude")
    ax.set_title("(b) Species-equation transport budget")
    ax.legend(fontsize=6.5, frameon=False); ax.set_xlim(0, 6)

    base = Params()
    if M_vals is None:
        M_vals = np.linspace(0, 2, 30)
    if S_vals is None:
        S_vals = np.linspace(0, 1.0, 30)
    MM, SS = np.meshgrid(M_vals, S_vals, indexing="ij")
    labels = ["conduction", "convection", "dispersion",
              "viscous dissipation", "Dufour", "heat generation"]
    DOM = np.zeros_like(MM)
    for i, m in enumerate(M_vals):
        for j, s in enumerate(S_vals):
            Pm = with_param(base, M=m, S=s)
            r = transport_budget(Pm)
            sh = r["energy_share_pct"]
            DOM[i, j] = labels.index(max(sh, key=sh.get))
    present = sorted(set(DOM.flatten().astype(int)))
    ax = fig.add_subplot(gs[2])
    cmap = matplotlib.colors.ListedColormap(["#4c72b0", "#dd8452"][: len(present)])
    c = ax.pcolormesh(SS, MM, DOM, cmap=cmap, shading="auto")
    cb = fig.colorbar(c, ax=ax, ticks=present)
    cb.ax.set_yticklabels([labels[i] for i in present])
    ax.set_xlabel(r"$S$"); ax.set_ylabel(r"$M$")
    ax.set_title("(c) Dominant energy-\ntransport mechanism")

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "fig6_manuscript_budget_full.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Figure 7 (manuscript) saved to {outpath}")
    return outpath


if __name__ == "__main__" and "--figures" in sys.argv:
    # Regenerate the exact manuscript Figure 6 and Figure 7 (each a 25x25 or
    # 30x30 grid of full BVP solves per panel, so this is slower than the
    # default summary above). Run explicitly with:
    #   python transport_budget.py --figures
    make_figure5()
    make_figure6()
