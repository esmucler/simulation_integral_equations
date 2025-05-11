# Simulation Integral Equations

This repository contains scripts to replicate the simulations described in the paper *"On the asymptotic validity of confidence sets for linear functionals of solutions to integral equations"*.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/simulation_integral_equations.git
   cd simulation_integral_equations
   ```

2. Install dependencies using [Poetry](https://python-poetry.org/):
   ```bash
   poetry install
   ```

## Usage

Run the `generate_plots.py` script to perform simulations and generate plots:

```bash
python generate_plots.py
```

This will produce two plots:
- `coverage_vs_sample_size.png`: Coverage vs. sample size for the two methods.
- `median_length_vs_sample_size.png`: Median length vs. sample size for the two methods.