"""
Generates the two entropy-generation / Bejan-number figures (fig9, fig10)
for Section 4.9 of the manuscript. Run after run_analysis.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Params, solve_case, with_param
from entropy import entropy_profile

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
ETA_INF = 10.0
N_MESH = 400
OMEGA, OMEGA_C, L = 1.0, 1.0, 0.5


def fig9_decomposition():
    """Figure 9: entropy-generation decomposition Ns_heat/fric/mass/cross/total
    vs eta at the base case."""
    P = Params()
    sol = solve_case(P, ETA_INF, N_MESH)
    eta = np.linspace(0, 6, 400)
    Nh, Nf, Nm, Ncr, Nt, Be = entropy_profile(sol, P, eta, OMEGA, OMEGA_C, L)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(eta, Nt, 'k-', lw=2.2, label=r"$N_s$ (total)")
    ax.plot(eta, Nh, '--', label=r"heat transfer")
    ax.plot(eta, Nf, '--', label=r"fluid friction")
    ax.plot(eta, Nm, '--', label=r"mass transfer")
    ax.plot(eta, Ncr, ':', label=r"Soret--Dufour cross term")
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$N_s(\eta)$")
    ax.set_title("Entropy-generation decomposition at the base parameter set")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig9_entropy_decomposition.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig9_entropy_decomposition.png"), dpi=300)
    plt.close(fig)
    print("fig9 saved. Ns_total range:", Nt.min(), Nt.max(), " any negative:", (Nt < 0).any())


def fig10_bejan_M():
    """Figure 10: Bejan number Be(eta) for M = 0, 0.5, 1.0, 2.0 at the base case."""
    P = Params()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    eta = np.linspace(0.02, 6, 400)  # avoid eta=0 only if needed; Be well-defined at 0 too
    for M in [0.0, 0.5, 1.0, 2.0]:
        sol = solve_case(with_param(P, M=M), ETA_INF, N_MESH)
        _, _, _, _, Nt, Be = entropy_profile(sol, with_param(P, M=M), eta, OMEGA, OMEGA_C, L)
        ax.plot(eta, Be, label=f"$M={M}$")
    ax.axhline(0.5, color='gray', lw=0.6, ls=':')
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$Be(\eta)$")
    ax.set_ylim(0, 1)
    ax.set_title("Bejan number profile for varying magnetic parameter $M$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig10_bejan_M.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig10_bejan_M.png"), dpi=300)
    plt.close(fig)
    print("fig10 saved.")


if __name__ == "__main__":
    fig9_decomposition()
    fig10_bejan_M()
