# MHD Micropolar Soret-Dufour BVP -- Python solver + independent cross-check

Python implementation of the solver behind *"Nonlinear Coupled Heat and
Mass Transfer in MHD Micropolar Fluid Flow past a Vertical Plate with
Thermal Dispersion, Soret-Dufour Effects and Chemical Reaction"*
(Madhyannapu, Lanka, Sajja, Suresh & Subbarao). All numerical results
in the manuscript are produced by pure-Python/SciPy code.

## Authors

1. Sri Venkata Durga Sudarsan Madhyannapu (corresponding author) --
   Department of Mathematics, Dr. RVR NRI Institute of Technology
   (Deemed to be University), Pothavarapadu, Agiripalli, Andhra
   Pradesh 521212, India. ORCID: 0009-0001-2126-6428
2. Swetha Lanka -- Department of Mathematics, Sir C. R. Reddy College
   of Engineering (Autonomous), Eluru, Andhra Pradesh 534007, India.
   ORCID: 0000-0001-5542-9486
3. Venkata Subrahmanyam Sajja -- Department of Mathematics, Koneru
   Lakshmaiah Education Foundation, Guntur, Andhra Pradesh, India.
   ORCID: 0000-0003-3926-7426
4. Moganti Satya Suresh -- Department of Mathematics, Sri Vasavi
   Engineering College, Tadepalligudem, West Godavari, Andhra Pradesh
   534101, India. ORCID: 0009-0004-5942-5748
5. Kankipati Subbarao -- Department of Mathematics, Dr. RVR NRI
   Institute of Technology (Deemed to be University), Pothavarapadu,
   Agiripalli, Andhra Pradesh 521212, India. ORCID: 0009-0000-3953-2950

## The energy-equation coefficient

The term multiplying theta'' in the similarity-reduced energy equation
(Eq. 12) is

```
(1 + S*fp) * theta'' + ...
```

This coefficient follows from the similarity reduction of the
governing equations and matches the base study's own stated energy
equation (Rao & Koteswara Rao 2017, Eq. 2.11) exactly. It is also the
coefficient that makes the model's doubly-reduced limit
(K=S=Ec=Q0=M=Sr=Du=Kr=Nc=0, Bi -> infinity) reduce exactly, with no
rescaling, to the classical Ostrach (1953) free-convection equations.

| Check | Result |
|---|---|
| Table 5 (M-sweep, 5 rows x 5 columns) | exact match, 6 dp |
| Table 7 (Sr/Du/Kr/Sc one-at-a-time sweeps) | exact match, 6 dp |
| Table 8 (25-point joint Sr,Du grid, both theta'(0) and phi'(0)) | exact match, 6 dp |
| Section 4.8 synergy index (Eq. 22): corner, grid-average, smallest point | 53.9% / 13.7% / 1.2%, all exact |
| Table 3 (mesh 200/400/800 nodes x eta_inf = 8/10/12/15) | exact match |
| Table 4 (both limiting-case reductions) | exact match |
| Table 6 (Bi-sweep, theta(0)) | exact match |
| Classical Ostrach (1953) limit, Pr=0.7 | f''(0)=0.678908, agrees with the value commonly quoted in the secondary literature (~0.674-0.68) to ~0.5% |
| Classical Ostrach limit, Pr=0.72 (his own tabulated Pr) | -theta'(0)=0.504629, agrees with a value read directly from Ostrach's 1953 text (Nu=63.6 at Gr_x=1e9) to ~0.23% |
| Classical Ostrach limit, Pr=1.0 | f''(0)=0.642185, agrees with the benchmark value reported by Peker & Oturanc (arXiv:1212.1706) for the classical Ostrach problem (f''(0)=0.6421) to ~0.01% |
| Rees & Pop (1998) cross-check, Pr=0.7, K=0 | f''(0)=0.678908, -theta'(0)=0.499508, converted from their own Table 1 (n=1) via an exact rescaling: agrees to ~0.0001%/0.0013% -- the tightest external check in this repository |
| Table 9 (K-sensitivity, K=0..2.0) | exact match, 6 dp; K=0 row confirms -g'(0)=0 (passive microrotation) to floating-point precision |
| Regularity map (Supplementary Fig. S6): Delta_min vs analytical bound | exact match to machine precision (0.0 discrepancy) across all 25 (Sr,Du) grid points |
| Sensitivity ranking (Supplementary Table S2, Section 4.11) | exact match, 6 dp; dominant parameter per output confirmed (K for f''/g', Bi for theta', Sc for phi') |
| Entropy-generation second-law check (Section 4.9) | Ns >= 0 confirmed across the domain and across the full M-sweep, minimum O(1e-9) |

## Two independent implementations

Two algorithmically distinct Python codes are included, both built
directly from the manuscript's equations:

- **`model.py` + `run_analysis.py`** -- the primary implementation.
  Solves the 9th-order BVP by global collocation (`scipy.integrate.
  solve_bvp`, a Lobatto IIIa scheme with residual control and adaptive
  mesh refinement). This is what generates every table and figure in
  the manuscript.
- **`shooting_crosscheck.py`** -- an independent verification.
  Solves the same equations and boundary conditions by a different
  method: explicit forward integration (`scipy.integrate.solve_ivp`,
  adaptive Runge-Kutta/Radau) combined with a nonlinear root-find
  (`scipy.optimize.fsolve`) on the missing initial slopes `f''(0)`,
  `g'(0)`, `theta(0)`, `phi'(0)`. It reproduces the base case and the
  full M-sweep of Table 6 to six decimal places, and independently
  reproduces both classical-Ostrach validation limits (Pr=0.7 and the
  tighter, exactly benchmarked Pr=1 case). It has not been
  run against every other table in the manuscript (the joint Sr-Du
  grid, mesh/domain-independence study, etc.) -- those rely on the
  collocation implementation's own residual control and mesh/domain
  independence checks, described in the manuscript's Section 3.

## Contents

| File | Purpose |
|---|---|
| `model.py` | The ODE system, boundary conditions, `solve_case()`. |
| `run_analysis.py` | Reproduces Tables 5, 6, 7, 8, 9 and the synergy index, all four Ostrach/Rees-Pop external checks, the entropy-generation second-law check, the regularity-map verification, the sensitivity ranking, the trade-off analysis, and the main manuscript figures (Fig. 1) plus Supplementary Figures S1-S7. |
| `shooting_crosscheck.py` | The independent shooting-method cross-check described above. Run directly: `python shooting_crosscheck.py`. |
| `entropy.py` | The entropy-generation number Ns(eta) and Bejan number Be(eta) (Section 4.9), derived from the same solved profiles as the rest of the paper. |
| `make_entropy_figures.py` | Generates manuscript Figs. 2-3 (entropy-generation decomposition and Bejan-number vs M) from `entropy.py`. Run after `run_analysis.py`. |

## Correspondence between code output filenames and the final manuscript

The output filenames inside `figures/` (and a few internal function
names in `run_analysis.py`) were set early in development and were not
renamed when the manuscript's own figure/table numbering was finalised
during peer preparation. The mapping to the **final** manuscript and
Supplementary Material is:

| Code output / function | Final manuscript item |
|---|---|
| `fig8_SrDu_contour.{pdf,png}` | Manuscript **Fig. 1** (joint Soret\u2013Dufour sweep) |
| `fig9_entropy_decomposition.{pdf,png}` (from `make_entropy_figures.py`) | Manuscript **Fig. 2** |
| `fig10_bejan_M.{pdf,png}` (from `make_entropy_figures.py`) | Manuscript **Fig. 3** |
| `fig1_base_case.{pdf,png}` | Supplementary **Fig. S8** |
| `fig11_regularity_map.{pdf,png}` | Supplementary **Fig. S6** |
| `fig12_tradeoff.{pdf,png}` | Supplementary **Fig. S7** |
| other `plot_family(...)` outputs (Sr, Du, Kr, Sc, S, M parameter sweeps) | Supplementary **Figs. S1\u2013S5, S9** |
| `table5_M_sweep()` | Manuscript **Table 5** |
| `table7_single_sweeps()` | Manuscript **Table 7** |
| `table8_srdu_grid()` | Manuscript **Table 8** |
| `table10_K_sensitivity()` | Manuscript **Table 9** (function name retained from an earlier draft numbering) |
| `table_sensitivity()` | Supplementary **Table S2** (9\u00d74 elasticity matrix) |
| `table_extended_crosscheck()` | Supplementary **Table S1** |

No numerical results changed between draft and final numbering \u2014
only the manuscript's own figure/table labels were reorganised when
material was moved to the Supplementary document.

## Setup

Requires Python 3.9+ (tested on 3.12) with SciPy, NumPy and Matplotlib
(tested against scipy 1.17, numpy 2.4, matplotlib 3.10; see
requirements.txt for exact pins). No GPU or special hardware is
needed -- everything here runs comfortably on a standard laptop CPU.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_analysis.py          # figures -> ./figures/ (PDF + 300 dpi PNG), tables -> stdout
python shooting_crosscheck.py   # independent cross-check -> stdout
python make_entropy_figures.py  # manuscript Figs. 2-3 (entropy generation, Bejan number) -> ./figures/
```

Approximate runtime on a standard laptop CPU (single core, no
parallelism used): `run_analysis.py` completes in under 15 seconds.
`shooting_crosscheck.py` took under 40 seconds in our own testing, but
be aware this can vary considerably across environments: the three
stiff `Bi=1e5` isothermal-wall validation blocks (the Pr=0.7, Pr=0.72
and Pr=1 Ostrach checks, and the Rees & Pop cross-check) rely on
`scipy.optimize.fsolve` finding a root for a shooting problem that is
numerically stiff, and how many iterations that takes can depend on
the installed SciPy/NumPy/BLAS versions. If the script seems to hang,
it is most likely stuck in one of these four blocks rather than in
the M-sweep or base-case sections, which are not stiff and complete
quickly. Neither script requires a GPU, a cluster, or any non-default
SciPy build; both are still no more than a few minutes even in a
slower environment.

## Notes on the model

- State vector: `y1..y9 = f, f', f'', g, g', theta, theta', phi, phi'`.
- theta'' and phi'' come out of the same 2x2 linear solve described in
  Eqs. (18)-(19) of the manuscript -- Cramer's rule, not a decoupling
  approximation.
- `solve_bvp` (collocation) and the shooting method use fundamentally
  different algorithms, so their agreement to six decimal places is a
  genuine independent cross-check, not two runs of the same solver.
- `Q0` is the heat-generation constant, matching the manuscript's own
  notation (`C` is reserved for species concentration there, so reusing
  it for the heat-generation constant would be ambiguous).

## External validation

Beyond the internal two-implementation cross-check, the doubly reduced
form of this model (all extension mechanisms switched off, isothermal-
wall limit via Bi -> infinity) matches the classical Ostrach (1953)
free-convection equations exactly in form, and matches Rees & Pop's
(1998) K=0 reduction of the same classical problem exactly as well,
under a single explicit rescaling that reconciles both the momentum
and energy equations simultaneously.

Numerically, four independent checks are available, from loosest to
tightest:
- ~0.5% against the range commonly quoted in the secondary literature
  at Pr=0.7;
- ~0.23% against a value read directly from Ostrach's own 1953 text
  at his tabulated Pr=0.72;
- ~0.01% against a benchmark value reported by Peker & Oturanc
  (arXiv:1212.1706) at Pr=1;
- ~0.0001-0.0013% against Rees & Pop's (1998) own Table 1 at Pr=0.7,
  converted into this paper's normalization -- the tightest of the
  four, and obtained entirely from a second paper's own published
  digits with no fitting.

All four are confirmed independently by both the collocation and
shooting implementations -- see `ostrach_validation()` and
`rees_pop_validation()` in run_analysis.py, and the corresponding
blocks in shooting_crosscheck.py -- and the manuscript's Conclusion
gives the full derivation and discussion.
