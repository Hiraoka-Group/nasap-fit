# nasap-fit

A high-performance Python library for fitting reaction rate constants in chemical kinetics analysis (NASAP), powered by a C++ backend using SUNDIALS and SuiteSparse.

## Installation

```bash
pip install nasap-fit
```

Requires Python 3.12+ on Linux or macOS.

## Quick Start

```python
from nasap_fit import NasapFit

engine = NasapFit.from_yaml("config.yaml")

# Global search with Differential Evolution
population = engine.run_de(
    pop_size=128,
    terminationCondition={"timeLimit": 60.0},
    seed=1,
)
best = min(population, key=lambda r: r.error)

# Local refinement with Levenberg-Marquardt
refined = engine.run_lm(
    best.constants,
    terminationCondition={"maxIter": 300, "xtol": 1e-8},
)

# Simulate concentration and reaction progress
result = engine.simulate(
    t=[1.0, 10.0, 100.0],
    constant=refined.constants,
    reaction_ids=[0, 1],
)
```

## License

MIT © Hiraoka Group
