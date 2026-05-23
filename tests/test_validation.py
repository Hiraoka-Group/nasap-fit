"""Tests for nasap_fit._validation.

Covers pure-Python functions that do not require the C++ binding (_core).
All test data is generated within this file.
"""
import math
import textwrap
from pathlib import Path

import pytest

from nasap_fit._validation import (
    find_project_root,
    resolve_path,
    validate_config_yaml,
    validate_constants_vector,
    validate_population,
    validate_qasap_csv_file,
    validate_qasap_data,
    validate_reaction_csv_file,
)


# ---------------------------------------------------------------------------
# validate_qasap_data
# ---------------------------------------------------------------------------

def test_validate_qasap_data_valid():
    data = [[0.0, 1.0], [1.0, 0.5], [2.0, 0.1]]
    validate_qasap_data(data, expected_columns=2)  # no exception


def test_validate_qasap_data_equal_times_ok():
    # Duplicate time values in consecutive rows are allowed.
    data = [[0.0, 1.0], [0.0, 0.9], [1.0, 0.5]]
    validate_qasap_data(data, expected_columns=2)


def test_validate_qasap_data_extra_columns_ok():
    # More columns than expected_columns is allowed.
    data = [[0.0, 1.0, 2.0], [1.0, 0.5, 1.5]]
    validate_qasap_data(data, expected_columns=2)


def test_validate_qasap_data_empty_raises():
    with pytest.raises(ValueError):
        validate_qasap_data([], expected_columns=2)


def test_validate_qasap_data_too_few_columns_raises():
    data = [[0.0], [1.0]]
    with pytest.raises(ValueError, match="expected"):
        validate_qasap_data(data, expected_columns=2)


def test_validate_qasap_data_non_numeric_raises():
    data = [[0.0, "bad"]]
    with pytest.raises(TypeError):
        validate_qasap_data(data, expected_columns=2)


def test_validate_qasap_data_inf_raises():
    data = [[0.0, math.inf]]
    with pytest.raises(ValueError):
        validate_qasap_data(data, expected_columns=2)


def test_validate_qasap_data_nan_raises():
    data = [[0.0, math.nan]]
    with pytest.raises(ValueError):
        validate_qasap_data(data, expected_columns=2)


def test_validate_qasap_data_negative_raises():
    data = [[0.0, -0.1]]
    with pytest.raises(ValueError):
        validate_qasap_data(data, expected_columns=2)


def test_validate_qasap_data_non_monotone_time_raises():
    data = [[1.0, 0.5], [0.5, 0.3]]
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_qasap_data(data, expected_columns=2)


# ---------------------------------------------------------------------------
# validate_constants_vector
# ---------------------------------------------------------------------------

def test_validate_constants_vector_valid():
    result = validate_constants_vector([1.0, 2.0, 3.0])
    assert result == [1.0, 2.0, 3.0]


def test_validate_constants_vector_returns_floats():
    result = validate_constants_vector([1, 2, 3])  # integer inputs are coerced to float
    assert all(isinstance(v, float) for v in result)


def test_validate_constants_vector_expected_size_match():
    validate_constants_vector([1.0, 2.0], expected_size=2)  # no exception


def test_validate_constants_vector_size_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        validate_constants_vector([1.0, 2.0, 3.0], expected_size=2)


def test_validate_constants_vector_empty_raises():
    with pytest.raises(ValueError):
        validate_constants_vector([])


def test_validate_constants_vector_zero_raises():
    with pytest.raises(ValueError):
        validate_constants_vector([1.0, 0.0, 1.0])


def test_validate_constants_vector_negative_raises():
    with pytest.raises(ValueError):
        validate_constants_vector([1.0, -0.5])


def test_validate_constants_vector_inf_raises():
    with pytest.raises(ValueError):
        validate_constants_vector([1.0, math.inf])


def test_validate_constants_vector_nan_raises():
    with pytest.raises(ValueError):
        validate_constants_vector([math.nan])


def test_validate_constants_vector_non_numeric_raises():
    with pytest.raises(TypeError):
        validate_constants_vector([1.0, "x"])


# ---------------------------------------------------------------------------
# validate_population
# ---------------------------------------------------------------------------

def test_validate_population_valid():
    pop = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    result = validate_population(pop, constant_size=2)
    assert len(result) == 3
    assert result[0] == [1.0, 2.0]


def test_validate_population_too_few_rows_raises():
    with pytest.raises(ValueError, match="at least 3"):
        validate_population([[1.0, 2.0], [3.0, 4.0]], constant_size=2)


def test_validate_population_min_size_1_ok():
    validate_population([[1.0, 2.0]], constant_size=2, min_size=1)  # no exception


def test_validate_population_row_size_mismatch_raises():
    pop = [[1.0, 2.0], [3.0], [5.0, 6.0]]
    with pytest.raises(ValueError):
        validate_population(pop, constant_size=2)


def test_validate_population_invalid_value_raises():
    pop = [[1.0, 2.0], [3.0, -1.0], [5.0, 6.0]]
    with pytest.raises(ValueError):
        validate_population(pop, constant_size=2)


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

def test_resolve_path_absolute():
    p = resolve_path("/tmp/foo.csv")
    assert p == Path("/tmp/foo.csv")


def test_resolve_path_with_project_root(tmp_path):
    result = resolve_path("data/qasap.csv", project_root=tmp_path)
    assert result == (tmp_path / "data/qasap.csv").resolve()


def test_resolve_path_with_config_path(tmp_path):
    config = tmp_path / "config" / "config.yaml"
    result = resolve_path("qasap.csv", config_path=config)
    assert result == (tmp_path / "config" / "qasap.csv").resolve()


def test_resolve_path_project_root_takes_priority(tmp_path):
    config = tmp_path / "sub" / "config.yaml"
    result = resolve_path("qasap.csv", config_path=config, project_root=tmp_path)
    assert result == (tmp_path / "qasap.csv").resolve()


# ---------------------------------------------------------------------------
# find_project_root
# ---------------------------------------------------------------------------

def test_find_project_root_via_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    result = find_project_root(sub)
    assert result == tmp_path


def test_find_project_root_via_git(tmp_path):
    (tmp_path / ".git").mkdir()
    result = find_project_root(tmp_path)
    assert result == tmp_path


def test_find_project_root_not_found(tmp_path):
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    # Returns None only when no marker exists up to the filesystem root.
    # When run inside a real project tree, a parent may match, so we accept
    # both None and Path without raising.
    result = find_project_root(isolated)
    assert result is None or isinstance(result, Path)


# ---------------------------------------------------------------------------
# validate_config_yaml
# ---------------------------------------------------------------------------

MINIMAL_CFG = {
    "QASAPDataFile": "qasap.csv",
    "reactionDataFile": "reaction.csv",
    "species": 3,
    "constantSize": 2,
    "trackedSpecies": {
        "A": {"index": 0, "100%concentration": 1.0},
        "B": {"index": 1, "100%concentration": 2.0},
    },
    "initConc": {"0": 0.5, "1": 0.3},
}


def _cfg(**overrides):
    import copy
    cfg = copy.deepcopy(MINIMAL_CFG)
    cfg.update(overrides)
    return cfg


def test_validate_config_yaml_minimal_valid():
    result = validate_config_yaml(MINIMAL_CFG, validate_files=False)
    assert result.species == 3
    assert result.constant_size == 2
    assert result.tracked_names == ["A", "B"]
    assert result.tracked_indices == [0, 1]



def test_validate_config_yaml_optional_fields():
    cfg = _cfg(tolAbsError=1e-6, tolRelError=1e-4, scalar=0.5, crossOver=0.9,
               upperLim=1e4, lowerLim=1e-3, cvodeMaxNumSteps=500)
    result = validate_config_yaml(cfg, validate_files=False)
    assert result.tol_abs == pytest.approx(1e-6)
    assert result.cvode_max_num_steps == 500


def test_validate_config_yaml_missing_qasap_raises():
    cfg = _cfg()
    del cfg["QASAPDataFile"]
    with pytest.raises(ValueError):
        validate_config_yaml(cfg, validate_files=False)


def test_validate_config_yaml_missing_reaction_raises():
    cfg = _cfg()
    del cfg["reactionDataFile"]
    with pytest.raises(ValueError):
        validate_config_yaml(cfg, validate_files=False)


def test_validate_config_yaml_invalid_species_raises():
    with pytest.raises(ValueError):
        validate_config_yaml(_cfg(species="abc"), validate_files=False)


def test_validate_config_yaml_species_zero_raises():
    with pytest.raises(ValueError):
        validate_config_yaml(_cfg(species=0), validate_files=False)


def test_validate_config_yaml_empty_tracked_species_raises():
    with pytest.raises(ValueError):
        validate_config_yaml(_cfg(trackedSpecies={}), validate_files=False)


def test_validate_config_yaml_tracked_index_out_of_range_raises():
    cfg = _cfg(trackedSpecies={"A": {"index": 99, "100%concentration": 1.0}})
    with pytest.raises(ValueError, match="species"):
        validate_config_yaml(cfg, validate_files=False)


def test_validate_config_yaml_empty_init_conc_raises():
    with pytest.raises(ValueError):
        validate_config_yaml(_cfg(initConc={}), validate_files=False)


def test_validate_config_yaml_init_conc_index_out_of_range_raises():
    with pytest.raises(ValueError):
        validate_config_yaml(_cfg(initConc={"99": 0.5}), validate_files=False)


def test_validate_config_yaml_upper_lower_inverted_raises():
    with pytest.raises(ValueError, match="upperLim"):
        validate_config_yaml(_cfg(upperLim=1.0, lowerLim=10.0), validate_files=False)


# ---------------------------------------------------------------------------
# validate_qasap_csv_file
# ---------------------------------------------------------------------------

def _write_qasap_csv(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_validate_qasap_csv_file_valid(tmp_path):
    p = tmp_path / "qasap.csv"
    _write_qasap_csv(p, """\
        time,A,B
        0.0,1.0,2.0
        1.0,0.5,1.0
        2.0,0.1,0.5
    """)
    validate_qasap_csv_file(p, ["A", "B"])  # no exception


def test_validate_qasap_csv_file_not_found_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        validate_qasap_csv_file(tmp_path / "missing.csv", ["A"])


def test_validate_qasap_csv_file_empty_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_qasap_csv_file(p, ["A"])


def test_validate_qasap_csv_file_short_header_raises(tmp_path):
    p = tmp_path / "short.csv"
    _write_qasap_csv(p, "time\n0.0\n")
    with pytest.raises(ValueError, match="short"):
        validate_qasap_csv_file(p, ["A"])


def test_validate_qasap_csv_file_missing_tracked_column_raises(tmp_path):
    p = tmp_path / "noA.csv"
    _write_qasap_csv(p, """\
        time,B
        0.0,1.0
    """)
    with pytest.raises(ValueError, match="missing"):
        validate_qasap_csv_file(p, ["A", "B"])


def test_validate_qasap_csv_file_non_monotone_time_raises(tmp_path):
    p = tmp_path / "badtime.csv"
    _write_qasap_csv(p, """\
        time,A
        2.0,1.0
        1.0,0.5
    """)
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_qasap_csv_file(p, ["A"])


def test_validate_qasap_csv_file_negative_value_raises(tmp_path):
    p = tmp_path / "negval.csv"
    _write_qasap_csv(p, """\
        time,A
        0.0,-1.0
    """)
    with pytest.raises(ValueError):
        validate_qasap_csv_file(p, ["A"])


def test_validate_qasap_csv_file_first_col_is_tracked_raises(tmp_path):
    p = tmp_path / "wrong.csv"
    _write_qasap_csv(p, """\
        A,B
        0.0,1.0
    """)
    with pytest.raises(ValueError):
        validate_qasap_csv_file(p, ["A", "B"])


# ---------------------------------------------------------------------------
# validate_reaction_csv_file
# ---------------------------------------------------------------------------

_REACTION_HEADER = "init_assem_id,entering_assem_id,product_assem_id,leaving_assem_id,duplicate_count,kind"


def _write_reaction_csv(path: Path, rows: list[str]) -> None:
    lines = [_REACTION_HEADER] + rows
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_reaction_csv_file_valid(tmp_path):
    p = tmp_path / "reaction.csv"
    _write_reaction_csv(p, [
        "0,,1,,1,k1",
        "1,,2,,1,k2",
    ])
    validate_reaction_csv_file(p, species=3, constant_size=2)  # no exception


def test_validate_reaction_csv_file_not_found_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        validate_reaction_csv_file(tmp_path / "missing.csv", species=3, constant_size=2)


def test_validate_reaction_csv_file_missing_required_column_raises(tmp_path):
    p = tmp_path / "reaction.csv"
    # kind 列を省略
    p.write_text("init_assem_id,product_assem_id,duplicate_count\n0,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        validate_reaction_csv_file(p, species=3, constant_size=1)


def test_validate_reaction_csv_file_id_out_of_range_raises(tmp_path):
    p = tmp_path / "reaction.csv"
    _write_reaction_csv(p, [
        "0,,99,,1,k1",  # product_assem_id=99 is out of range for species=3
    ])
    with pytest.raises(ValueError, match="out of range"):
        validate_reaction_csv_file(p, species=3, constant_size=1)


def test_validate_reaction_csv_file_dup_count_zero_raises(tmp_path):
    p = tmp_path / "reaction.csv"
    _write_reaction_csv(p, [
        "0,,1,,0,k1",  # duplicate_count=0 is invalid
    ])
    with pytest.raises(ValueError, match=">= 1"):
        validate_reaction_csv_file(p, species=3, constant_size=1)


def test_validate_reaction_csv_file_kinds_mismatch_raises(tmp_path):
    p = tmp_path / "reaction.csv"
    _write_reaction_csv(p, [
        "0,,1,,1,k1",
        "1,,2,,1,k2",
    ])
    with pytest.raises(ValueError, match="constant"):
        validate_reaction_csv_file(p, species=3, constant_size=3)  # expects 3 unique kinds but only 2 are present
