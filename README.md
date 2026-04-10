# Quantum Amplitude Estimation for Catastrophe Insurance Tail-Risk Pricing

Empirical convergence and NISQ noise analysis comparing Grover-amplified quantum amplitude estimation (QAE) against classical Monte Carlo baselines for excess-of-loss pricing on heavy-tailed catastrophe distributions.

## Setup

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running the experiments

```bash
python run_pipeline.py
```

This runs all seven experiments (synthetic Pareto/lognormal data and real NOAA Storm Events data), writes results to `results/`, and copies plot images into `full_paper/` for LaTeX compilation. On a laptop the full pipeline takes approximately 30--60 minutes; the NOAA data download requires internet access on first run.

## Project structure

```
fairy_queen/              Core library
  data_pipeline.py          Data acquisition, lognormal fitting, NOAA download
  quantum_circuits.py       State preparation, oracle construction, discretisation
  experiment1.py            Noiseless convergence scaling
  experiment2.py            NISQ noise degradation
  experiment3.py            Tail-specific excess loss
  experiment4.py            Validation on real NOAA data
  experiment5.py            Budget-matched comparison with strong classical baselines
  experiment6.py            Qubit/bin sweep (discretisation vs estimation error)
  experiment7.py            Empirical PMF (no parametric fit)
  results.py                Plot generation and summary metrics
  logging_config.py         Logging setup
run_pipeline.py           Main entry point for all experiments
full_paper/paper.tex      Paper source
tests/                    Unit tests
```

## Supplementary experiment scripts

Several supplementary scripts were used to generate additional results reported in the paper (bootstrap convergence slopes, quasi-Monte Carlo baselines, log-spaced binning analysis, extended n=8 qubit experiments). These are not included in the repository to keep it focused on the core pipeline, but are available from the authors on request.

## Data

Synthetic data is generated deterministically (seed 42). NOAA Storm Events files are downloaded automatically on first run and cached in `data/cache/`. The exact file versions used in the paper are listed in Table 1 of the manuscript; a pinned manifest is written to `data/cache/noaa_manifest.json`.

## Tests

```bash
pytest
```

## License

See the paper for citation details.
