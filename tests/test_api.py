"""Integration tests for the nasap_fit public API.

Requires the C++ binding (_core) to be built.  Skipped automatically when
_core is still a MagicMock (i.e. not compiled yet).

Synthetic test data
-------------------
We use a minimal 2-species reversible reaction A ⇌ B:

    reaction CSV:
        init=0 (A), product=1 (B), kind=kf   →  A → B
        init=1 (B), product=0 (A), kind=kr   →  B → A

The C++ engine solves (per reactionNetwork.cpp, using speciesData_[species]=1.0
for unimolecular reactions):

    dx[A]/dt = -kf*x[A] + kr*x[B]
    dx[B]/dt = +kf*x[A] - kr*x[B]

We reproduce this ODE in scipy/Radau and write the resulting concentrations as
QASAP CSV.  The SSR at the true constants should be near zero and strictly
smaller than at any obviously-wrong constants.

Rate constants are sorted alphabetically in std::map, so index 0 → kf, 1 → kr.
"""
import math
from unittest.mock import MagicMock

import pytest

import nasap_fit._core as _core_mod

_CORE_IS_REAL = not isinstance(_core_mod, MagicMock)
pytestmark = pytest.mark.skipif(
    not _CORE_IS_REAL,
    reason="_core not built — skipping API integration tests",
)

_KF_TRUE = 0.3
_KR_TRUE = 0.1
_K_TRUE = [_KF_TRUE, _KR_TRUE]  # index 0 = kf, index 1 = kr (alphabetical)
_FULL_CONC = 2.0e-3
_T_EVAL = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


@pytest.fixture(scope="module")
def minimal_engine(tmp_path_factory):
    """Return a NasapFit engine built from synthetic A ⇌ B QASAP data."""
    pytest.importorskip("scipy")
    from scipy.integrate import solve_ivp

    def rhs(t, x):
        return [
            -_KF_TRUE * x[0] + _KR_TRUE * x[1],
             _KF_TRUE * x[0] - _KR_TRUE * x[1],
        ]

    sol = solve_ivp(
        rhs, [0.0, 50.0], [_FULL_CONC, 0.0],
        t_eval=_T_EVAL, method="Radau", rtol=1e-10, atol=1e-13,
    )

    tmp = tmp_path_factory.mktemp("minimal_system")

    (tmp / "reaction.csv").write_text(
        "init_assem_id,entering_assem_id,product_assem_id,"
        "leaving_assem_id,duplicate_count,kind\n"
        "0,,1,,1,kf\n"
        "1,,0,,1,kr\n",
        encoding="utf-8",
    )

    # Track only species B (index 1).
    # QASAP is stored as percentage values, while the objective compares
    # concentrations: y_sim - (QASAP/100 * fullConc).
    lines = ["time,B"]
    for t, y_b in zip(sol.t, sol.y[1]):
        pct = max(float(y_b), 0.0) / _FULL_CONC * 100.0
        lines.append(f"{t},{pct}")
    (tmp / "qasap.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (tmp / "config.yaml").write_text(
        "QASAPDataFile: qasap.csv\n"
        "reactionDataFile: reaction.csv\n"
        "species: 2\n"
        "constantSize: 2\n"
        "trackedSpecies:\n"
        "  B:\n"
        "    index: 1\n"
        f"    '100%concentration': {_FULL_CONC}\n"
        "initConc:\n"
        f"  0: {_FULL_CONC}\n"
        "tolAbsError: 1.0e-10\n"
        "tolRelError: 1.0e-7\n"
        "logLevel: quiet\n",
        encoding="utf-8",
    )

    from nasap_fit import NasapFit
    return NasapFit.from_yaml(tmp / "config.yaml")


# ---------------------------------------------------------------------------
# Module-level API (no engine instance required)
# ---------------------------------------------------------------------------

def test_default_config_returns_object():
    from nasap_fit import default_config
    cfg = default_config()
    assert cfg is not None
    assert hasattr(cfg, "species")


def test_config_from_yaml_creates_config(minimal_engine, tmp_path_factory):
    # Smoke test: engine was constructed from YAML without error.
    # Verify the config object is accessible.
    assert minimal_engine is not None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_reaction_count(minimal_engine):
    assert minimal_engine.reaction_count == 2


def test_rate_constant_index_keys(minimal_engine):
    idx = minimal_engine.rate_constant_index
    assert set(idx.keys()) == {"kf", "kr"}
    assert idx["kf"] == 0  # alphabetical order: kf < kr
    assert idx["kr"] == 1


# ---------------------------------------------------------------------------
# calc_error
# ---------------------------------------------------------------------------

def test_calc_error_returns_finite_nonneg(minimal_engine):
    err = minimal_engine.calc_error(_K_TRUE)
    assert math.isfinite(err)
    assert err >= 0.0


def test_calc_error_true_k_less_than_bad_k(minimal_engine):
    # Error at true constants must be lower than at an obviously-wrong set.
    err_true = minimal_engine.calc_error(_K_TRUE)
    err_bad = minimal_engine.calc_error([10.0, 10.0])
    assert err_true < err_bad


def test_calc_error_is_sum_of_squared_concentration_residuals(minimal_engine):
    bad_constants = [10.0, 10.0]
    simulated = minimal_engine.simulate(
        t=_T_EVAL, constants=bad_constants, reaction_ids=[]
    )
    expected = 0.0
    for t, state in zip(_T_EVAL, simulated.y):
        observed_b = (
            _FULL_CONC
            * _KF_TRUE
            / (_KF_TRUE + _KR_TRUE)
            * (1.0 - math.exp(-(_KF_TRUE + _KR_TRUE) * t))
        )
        expected += (state[1] - observed_b) ** 2
    assert minimal_engine.calc_error(bad_constants) == pytest.approx(
        expected, rel=1e-5, abs=1e-14
    )


def test_calc_error_wrong_length_raises(minimal_engine):
    with pytest.raises(ValueError):
        minimal_engine.calc_error([0.3])  # should be length 2


def test_calc_error_nonpositive_raises(minimal_engine):
    with pytest.raises(ValueError):
        minimal_engine.calc_error([0.3, 0.0])


# ---------------------------------------------------------------------------
# calc_rmse
# ---------------------------------------------------------------------------

def test_calc_rmse_nonneg(minimal_engine):
    err = minimal_engine.calc_error(_K_TRUE)
    rmse = minimal_engine.calc_rmse(err)
    assert math.isfinite(rmse)
    assert rmse >= 0.0


def test_calc_rmse_is_root_mean_squared_concentration_error(minimal_engine):
    err = minimal_engine.calc_error([10.0, 10.0])
    expected = math.sqrt(err / len(_T_EVAL))
    assert minimal_engine.calc_rmse(err) == pytest.approx(expected)


def test_calc_rmse_negative_raises(minimal_engine):
    with pytest.raises(ValueError):
        minimal_engine.calc_rmse(-1.0)


def test_calc_rmse_inf_raises(minimal_engine):
    with pytest.raises(ValueError):
        minimal_engine.calc_rmse(math.inf)


# ---------------------------------------------------------------------------
# run_de
# ---------------------------------------------------------------------------

def test_run_de_returns_nonempty_list(minimal_engine):
    pop = minimal_engine.run_de(
        pop_size=6,
        termination_condition={"maxIter": 3},
        seed=42,
    )
    assert len(pop) == 6


def test_run_de_population_sorted_by_error(minimal_engine):
    pop = minimal_engine.run_de(
        pop_size=6,
        termination_condition={"maxIter": 3},
        seed=42,
    )
    errors = [ind.error for ind in pop]
    assert errors == sorted(errors)


def test_run_de_individual_has_correct_constants_length(minimal_engine):
    pop = minimal_engine.run_de(
        pop_size=6,
        termination_condition={"maxIter": 3},
        seed=42,
    )
    assert all(len(ind.constants) == 2 for ind in pop)


def test_run_de_individual_errors_are_finite_nonneg(minimal_engine):
    pop = minimal_engine.run_de(
        pop_size=6,
        termination_condition={"maxIter": 3},
        seed=42,
    )
    assert all(math.isfinite(ind.error) and ind.error >= 0.0 for ind in pop)


def test_run_de_empty_termination_raises(minimal_engine):
    with pytest.raises(ValueError):
        minimal_engine.run_de(pop_size=6, termination_condition={})


# ---------------------------------------------------------------------------
# run_de_from_population
# ---------------------------------------------------------------------------

def test_run_de_from_population_returns_sorted_list(minimal_engine):
    init_pop = [
        [0.1, 0.1],
        [0.3, 0.1],
        [1.0, 1.0],
    ]
    pop = minimal_engine.run_de_from_population(
        init_pop, termination_condition={"maxIter": 2}, seed=0
    )
    errors = [ind.error for ind in pop]
    assert errors == sorted(errors)


def test_run_de_from_population_too_few_rows_raises(minimal_engine):
    with pytest.raises(ValueError):
        minimal_engine.run_de_from_population(
            [[0.3, 0.1], [1.0, 1.0]],
            termination_condition={"maxIter": 1},
        )


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

def test_simulate_time_points_preserved(minimal_engine):
    t_in = [1.0, 5.0, 10.0]
    result = minimal_engine.simulate(t=t_in, constants=_K_TRUE, reaction_ids=[])
    assert list(result.t) == t_in


def test_simulate_species_concentrations_nonneg(minimal_engine):
    result = minimal_engine.simulate(
        t=[1.0, 5.0, 10.0], constants=_K_TRUE, reaction_ids=[]
    )
    for species_series in result.y:
        assert all(c >= -1e-9 for c in species_series)


def test_simulate_concentration_sum_is_conserved(minimal_engine):
    # Total concentration (A+B) is conserved in a closed reversible reaction.
    result = minimal_engine.simulate(
        t=[1.0, 10.0, 50.0], constants=_K_TRUE, reaction_ids=[]
    )
    # y[time_index][species_index]
    totals = [result.y[i][0] + result.y[i][1] for i in range(3)]
    for total in totals:
        assert abs(total - _FULL_CONC) < 1e-8


def test_simulate_reaction_progress_structure(minimal_engine):
    result = minimal_engine.simulate(
        t=[1.0, 5.0], constants=_K_TRUE, reaction_ids=[0, 1]
    )
    rp = result.reactionProgress
    assert list(rp.reaction_ids) == [0, 1]
    assert len(rp.J) == 2
    assert len(rp.reaction_labels) == 2


def test_simulate_negative_time_raises(minimal_engine):
    with pytest.raises(ValueError):
        minimal_engine.simulate(t=[-1.0], constants=_K_TRUE, reaction_ids=[])


def test_simulate_invalid_reaction_id_raises(minimal_engine):
    with pytest.raises(ValueError):
        minimal_engine.simulate(t=[1.0], constants=_K_TRUE, reaction_ids=[99])


def test_simulate_reaction_progress_mass_balance(minimal_engine):
    # For A ⇌ B starting at [A]=_FULL_CONC, [B]=0:
    #   d[B]/dt = kf*[A] - kr*[B]  =>  J_forward(t) - J_reverse(t) = [B](t) - [B](0) = [B](t)
    # This identity holds exactly for any t.
    t_test = [1.0, 5.0, 10.0, 50.0]
    result = minimal_engine.simulate(t=t_test, constants=_K_TRUE, reaction_ids=[0, 1])
    for i in range(len(t_test)):
        B_conc = result.y[i][1]
        J_forward = result.reactionProgress.J[i][0]
        J_reverse = result.reactionProgress.J[i][1]
        assert abs((J_forward - J_reverse) - B_conc) < 1e-6, (
            f"t={t_test[i]}: J_forward={J_forward}, J_reverse={J_reverse}, "
            f"B_conc={B_conc}, diff={abs((J_forward - J_reverse) - B_conc)}"
        )


# ---------------------------------------------------------------------------
# run_lm
# ---------------------------------------------------------------------------

def test_run_lm_result_has_constants_and_error(minimal_engine):
    result = minimal_engine.run_lm(_K_TRUE, termination_condition={"maxIter": 5})
    assert hasattr(result, "constants")
    assert hasattr(result, "error")
    assert len(result.constants) == 2


def test_run_lm_does_not_increase_error(minimal_engine):
    # LM starting from a perturbed point should not worsen the error.
    initial = [0.5, 0.5]
    initial_error = minimal_engine.calc_error(initial)
    result = minimal_engine.run_lm(initial, termination_condition={"maxIter": 30})
    assert result.error <= initial_error + 1e-10


def test_run_lm_error_is_finite_nonneg(minimal_engine):
    result = minimal_engine.run_lm(_K_TRUE, termination_condition={"maxIter": 5})
    assert math.isfinite(result.error)
    assert result.error >= 0.0


# ---------------------------------------------------------------------------
# run_lm_batch
# ---------------------------------------------------------------------------

def test_run_lm_batch_returns_list_of_correct_length(minimal_engine):
    initial_batch = [_K_TRUE, [0.5, 0.5]]
    results = minimal_engine.run_lm_batch(
        initial_batch, termination_condition={"maxIter": 5}
    )
    assert len(results) == 2


def test_run_lm_batch_each_result_has_constants(minimal_engine):
    results = minimal_engine.run_lm_batch(
        [_K_TRUE, [0.5, 0.5]], termination_condition={"maxIter": 5}
    )
    assert all(len(r.constants) == 2 for r in results)


# ---------------------------------------------------------------------------
# gauss_newton_hessian
# ---------------------------------------------------------------------------

def test_gauss_newton_hessian_shape(minimal_engine):
    H = minimal_engine.gauss_newton_hessian(_K_TRUE)
    assert len(H) == 2
    assert all(len(row) == 2 for row in H)


def test_gauss_newton_hessian_uses_concentration_residual_scaling(minimal_engine):
    H = minimal_engine.gauss_newton_hessian(_K_TRUE)
    assert max(abs(value) for row in H for value in row) < 1e-3


def test_gauss_newton_hessian_values_finite(minimal_engine):
    H = minimal_engine.gauss_newton_hessian(_K_TRUE)
    assert all(math.isfinite(H[i][j]) for i in range(2) for j in range(2))
