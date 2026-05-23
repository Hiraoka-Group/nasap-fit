# Getting Started

## Installation

Install from PyPI:

```bash
pip install nasap-fit
```

Requires Python 3.12 or later on Linux or macOS.

## Configuration File

`nasap-fit` reads a YAML file that describes your reaction network and experimental data.
A minimal `config.yaml` looks like this:

```yaml
QASAPDataFile: data/concentration.csv   # path to QASAP experimental data
reactionDataFile: data/reactions.csv    # path to reaction network definition
species: 10                             # total number of chemical species
constantSize: 3                         # number of distinct rate constant types

trackedSpecies:
  M9L6:                                 # label used in QASAPDataFile header
    index: 9                            # zero-based species index
    100%concentration: 1.0              # reference concentration for normalisation

initConc:
  0: 9.0                                # species 0 has initial concentration 9.0
```

### Required fields

| Field | Type | Description |
|---|---|---|
| `QASAPDataFile` | string | Path to the CSV file with experimental concentration data |
| `reactionDataFile` | string | Path to the CSV file defining the reaction network |
| `species` | int | Total number of chemical species |
| `constantSize` | int | Number of distinct rate constant types (unique `kind` values) |
| `trackedSpecies` | mapping | Species to compare against QASAP data |
| `initConc` | mapping | Initial concentrations keyed by species index |

### Optional fields

| Field | Default | Description |
|---|---|---|
| `tolAbsError` | C++ default | Absolute tolerance for the ODE solver (CVODE) |
| `tolRelError` | C++ default | Relative tolerance for the ODE solver |
| `scalar` | C++ default | Differential Evolution scaling factor |
| `crossOver` | C++ default | Differential Evolution crossover probability |
| `upperLim` | C++ default | Upper bound for DE initial population |
| `lowerLim` | C++ default | Lower bound for DE initial population |
| `cvodeMaxNumSteps` | C++ default | Maximum steps for CVODE per time interval |
| `logLevel` | `normal` | Verbosity: `quiet`, `normal`, or `verbose` |

## Basic Workflow

### 1. Load the engine

```python
from nasap_fit import NasapFit

engine = NasapFit.from_yaml("config.yaml")
```

### 2. Global search with Differential Evolution

```python
population = engine.run_de(
    pop_size=128,
    termination_condition={"timeLimit": 60.0},
    seed=1,
)
# Sort by error (ascending)
population.sort(key=lambda r: r.error)
best = population[0]
print(f"Best SSR: {best.error:.4f}")
print(f"Best constants: {best.constants}")
```

### 3. Local refinement with Levenberg-Marquardt

```python
refined = engine.run_lm(
    best.constants,
    termination_condition={"maxIter": 300, "xtol": 1e-8},
)
print(f"Refined SSR: {refined.error:.4f}")
```

### 4. Compute NRMSE

```python
nrmse = engine.calc_nrmse(refined.error)
print(f"NRMSE: {nrmse:.4f}")
```

### 5. Simulate concentration profiles

```python
result = engine.simulate(
    t=[0.0, 1.0, 10.0, 100.0, 1000.0],
    constants=refined.constants,
    reaction_ids=list(range(engine.reaction_count)),
)

# Concentration of species 9 at each time point
conc_species9 = result.y[9]
print(conc_species9)
```

## Termination Conditions

Pass a dictionary to `termination_condition` with one or more of these keys:

| Key | Type | Description |
|---|---|---|
| `maxIter` | int | Stop after this many generations |
| `timeLimit` | float | Stop after this many seconds of wall time |
| `xtol` | float | Stop when the parameter-space neighbourhood is smaller than this (LM only) |
| `ftolAbs` | float | Stop when absolute improvement in SSR is below this threshold |
| `ftolRel` | float | Stop when relative improvement in SSR is below this threshold |
| `targetError` | float | Stop when SSR falls below this value |
| `stall` | int | Stop when improvement stalls for this many generations |

!!! note
    When `ftolAbs` or `ftolRel` is set without `stall`, `stall` is automatically
    set to `1` with a warning.
