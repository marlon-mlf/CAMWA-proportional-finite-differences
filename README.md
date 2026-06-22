# CAMWA Proportional Finite Differences

This repository contains the supplementary Python code for the manuscript:

**Logarithmic-Scale Finite Difference Schemes for Positive Parabolic Problems on Geometric Grids**

Authors:

- José D. Zúniga-Varela
- Marlon M. López-Flores
- William Campillay-Llanos

The code reproduces the numerical tables and convergence figure reported in the manuscript. The computations are deterministic and use no random sampling.

## Files

- `proportional_heat_benchmark.py`: deterministic Python script for the explicit proportional heat benchmark, backward Euler check, Crank–Nicolson reference check, source-driven check, variable-coefficient logarithmic diffusion check, additive-x diagnostic, same-node nonuniform-x diagnostic, two-dimensional logarithmic-grid check, and smooth-in-x diagnostic.
- `regression_reference.json`: reference numerical output used by the `--check` option.
- `requirements.txt`: minimal dependency list.
- `output/`: generated CSV files, LaTeX table fragments, environment metadata, checksums, and the regenerated convergence figure.
- `LICENSE`: MIT License for the supplementary Python code.
- `CITATION.cff`: citation metadata for the software release.

## Environment

Python 3.9 or later is required.

The script uses:

- NumPy
- Matplotlib

No random sampling is used, so no random seed is required.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the results

To regenerate all CSV files, LaTeX table fragments, checksums, metadata, and the convergence figure, run:

```bash
python3 proportional_heat_benchmark.py --outdir output
```

To run the regression check against the reference output, run:

```bash
python3 proportional_heat_benchmark.py --check --outdir output_check
```

The regression check compares the computed values with `regression_reference.json` using a fixed numerical tolerance.

## Output structure

After running the script, the output directory contains:

- `output/csv/*.csv`: machine-readable numerical tables.
- `output/latex_tables/*.tex`: LaTeX table-row fragments.
- `output/figures/convergence_loglog.pdf`: regenerated convergence figure.
- `output/computed_tables.json`: complete numerical output.
- `output/checksums.txt`: SHA256 checksum information.
- `output/environment.json`: Python, NumPy, and platform metadata.

The generated outputs reproduce Tables 3–11 and Figure 2 of the manuscript up to displayed rounding.

## License

The supplementary Python code in this repository is released under the MIT License. See the `LICENSE` file for details.

This license applies to the software code in this repository. The manuscript text remains under the publication terms of the journal.

## Citation

Please cite the archived Zenodo release.

Zúniga-Varela, J. D., López-Flores, M. M., & Campillay-Llanos, W. (2026). Supplementary code for logarithmic-scale finite difference schemes for positive parabolic problems on geometric grids, version 1.0.0. Zenodo. https://doi.org/10.5281/zenodo.20789782

GitHub release: https://github.com/marlon-mlf/CAMWA-proportional-finite-differences/releases/tag/v1.0.0

## Contact

For questions about the code or reproducibility package, contact:

William Campillay-Llanos  
wcampillay@uct.cl
