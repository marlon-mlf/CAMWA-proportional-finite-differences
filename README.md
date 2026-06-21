# Supplementary reproducibility package

This directory contains the deterministic source code and regression files used to reproduce the numerical verification in the manuscript.

## Files

- `proportional_heat_benchmark.py`: deterministic Python script for the explicit proportional heat benchmark, backward Euler check, Crank-Nicolson reference check, source-driven check, variable-coefficient logarithmic diffusion consistency check, additive-x diagnostic, same-node nonuniform-x diagnostic, two-dimensional logarithmic-grid check, and smooth-in-x diagnostic.
- `regression_reference.json`: reference numerical output used by the `--check` option.
- `requirements.txt`: minimal dependency list.
- `output/`: generated CSV files, LaTeX table rows, environment metadata, checksums, and the regenerated convergence figure.

## Environment

Python 3.9 or later is required. The script uses NumPy and Matplotlib. It uses no random sampling, so no random seed is required.

## Commands

From the root directory of the article package, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Supplementary/requirements.txt
python3 Supplementary/proportional_heat_benchmark.py --outdir Supplementary/output
python3 Supplementary/proportional_heat_benchmark.py --check --outdir Supplementary/output_check
```

The first command sequence regenerates CSV files, LaTeX table fragments, checksums, and the convergence figure. The second run checks the computed values against `regression_reference.json` with a fixed numerical tolerance.

## Output structure

- `Supplementary/output/csv/*.csv`: machine-readable tables.
- `Supplementary/output/latex_tables/*.tex`: LaTeX table-row fragments.
- `Supplementary/output/figures/convergence_loglog.pdf`: regenerated convergence figure.
- `Supplementary/output/computed_tables.json`: complete numerical output.
- `Supplementary/output/checksums.txt`: SHA256 checksum for the computed JSON output.
- `Supplementary/output/environment.json`: Python, NumPy, and platform metadata.

## License

The supplementary Python code in this repository is released under the MIT License. See the `LICENSE` file for details. This license applies to the software code in this repository. The manuscript text remains under the publication terms of the journal.
