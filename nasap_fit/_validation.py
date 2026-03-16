from __future__ import annotations

import csv
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


def validate_qasap_data(data: Sequence[Sequence[float]], expected_columns: int) -> None:
    if not isinstance(data, Sequence) or len(data) == 0:
        raise ValueError("qasap_data must be a non-empty 2D sequence")

    previous_time = None
    for row_index, row in enumerate(data):
        if not isinstance(row, Sequence):
            raise TypeError(f"row {row_index} is not a sequence")
        if len(row) < expected_columns:
            raise ValueError(
                f"row {row_index} has {len(row)} columns, expected {expected_columns}"
            )

        numeric_row = []
        for col_index, value in enumerate(row):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"row {row_index} col {col_index} is not numeric: {value!r}"
                ) from exc

            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"row {row_index} col {col_index} must be finite: {numeric_value!r}"
                )
            if numeric_value < 0:
                raise ValueError(
                    f"row {row_index} col {col_index} must be >= 0: {numeric_value!r}"
                )
            numeric_row.append(numeric_value)

        current_time = numeric_row[0]
        # time is already checked to be >= 0 above
        if previous_time is not None and current_time < previous_time:
            raise ValueError("time values must be non-decreasing")
        previous_time = current_time


def _is_int_string(value: object) -> bool:
    s = str(value).strip()
    if s == "":
        return False
    try:
        f = float(s)
    except ValueError:
        return False
    return f.is_integer()


def _require_int(name: str, value: object, *, min_value: int | None = None) -> int:
    try:
        v = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer (got {value!r})") from exc
    if min_value is not None and v < min_value:
        raise ValueError(f"{name} must be >= {min_value} (got {v})")
    return v


def _require_float(
    name: str,
    value: object,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number (got {value!r})") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite (got {v!r})")
    if min_value is not None and v < min_value:
        raise ValueError(f"{name} must be >= {min_value} (got {v})")
    if max_value is not None and v > max_value:
        raise ValueError(f"{name} must be <= {max_value} (got {v})")
    return v


def _require_nonempty_str(name: str, value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{name} is required and must be a non-empty string")
    return value


def find_project_root(start: str | os.PathLike[str]) -> Path | None:
    """Best-effort: walk up from start to find a repo root (pyproject.toml/.git/CMakeLists.txt)."""
    p = Path(start).resolve()
    if p.is_file():
        p = p.parent
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists() or (parent / "CMakeLists.txt").exists():
            return parent
    return None


def resolve_path(
    raw_path: str,
    *,
    config_path: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if project_root is not None:
        return (Path(project_root) / p).resolve()
    if config_path is not None:
        base = Path(config_path).resolve().parent
        return (base / p).resolve()
    # last resort: current working directory
    return (Path.cwd() / p).resolve()


def validate_qasap_csv_file(path: str | os.PathLike[str], tracked_names: Sequence[str]) -> None:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"QASAPDataFile not found: {str(p)}")

    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"QASAPDataFile is empty: {str(p)}") from exc

        header = [h.strip() for h in header]
        if len(header) < 2:
            raise ValueError(f"QASAPDataFile header is too short: {str(p)}")

        time_col = header[0]
        if time_col in tracked_names:
            raise ValueError("QASAPDataFile: first column must be time, not a tracked species")

        missing = [n for n in tracked_names if n not in header]
        if missing:
            raise ValueError(f"QASAPDataFile missing tracked species columns: {missing}")

        col_index = {name: header.index(name) for name in tracked_names}

        prev_time: float | None = None
        row_no = 1
        for row in reader:
            row_no += 1
            if len(row) == 0 or all(str(x).strip() == "" for x in row):
                continue
            if len(row) < len(header):
                raise ValueError(f"QASAPDataFile row {row_no} has too few columns")

            try:
                t = float(row[0])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"QASAPDataFile row {row_no} time is not numeric: {row[0]!r}") from exc
            if not math.isfinite(t) or t < 0:
                raise ValueError(f"QASAPDataFile row {row_no} time must be finite and >= 0 (got {t!r})")
            if prev_time is not None and t < prev_time:
                raise ValueError(
                    f"QASAPDataFile time values must be non-decreasing (row {row_no}: {t} < {prev_time})"
                )
            prev_time = t

            for name in tracked_names:
                idx = col_index[name]
                cell = row[idx]
                try:
                    v = float(cell)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"QASAPDataFile row {row_no} col {name!r} is not numeric: {cell!r}"
                    ) from exc
                if not math.isfinite(v) or v < 0:
                    raise ValueError(
                        f"QASAPDataFile row {row_no} col {name!r} must be finite and >= 0 (got {v!r})"
                    )


def validate_reaction_csv_file(
    path: str | os.PathLike[str],
    *,
    species: int,
    constant_size: int,
) -> None:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"reactionDataFile not found: {str(p)}")

    required = {
        "init_assem_id",
        "entering_assem_id",
        "product_assem_id",
        "leaving_assem_id",
        "duplicate_count",
        "kind",
    }
    kinds: set[str] = set()

    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"reactionDataFile has no header: {str(p)}")

        fields = [fn.strip() for fn in reader.fieldnames]
        # allow an extra leading unnamed index column
        fields_set = set(fields)
        missing = sorted(required - fields_set)
        if missing:
            raise ValueError(f"reactionDataFile missing required columns: {missing}")

        row_no = 1
        for row in reader:
            row_no += 1
            if row is None:
                continue

            # required IDs: init/product must exist and be valid
            for col in ("init_assem_id", "product_assem_id"):
                raw = row.get(col, "")
                if raw is None or str(raw).strip() == "":
                    raise ValueError(f"reactionDataFile row {row_no}: column {col!r} is missing")

            for col in ("init_assem_id", "entering_assem_id", "product_assem_id", "leaving_assem_id"):
                raw = row.get(col, "")
                s = "" if raw is None else str(raw).strip()
                if s == "":
                    continue  # entering/leaving may be empty
                if not _is_int_string(s):
                    raise ValueError(f"reactionDataFile row {row_no} column {col!r}: not an integer ({raw!r})")
                v = int(float(s))
                if v < 0 or v >= species:
                    raise ValueError(
                        f"reactionDataFile row {row_no} column {col!r}: {v} out of range [0, {species - 1}]"
                    )

            dup_raw = row.get("duplicate_count", "")
            dup_s = "" if dup_raw is None else str(dup_raw).strip()
            if dup_s == "" or not _is_int_string(dup_s):
                raise ValueError(
                    f"reactionDataFile row {row_no} column 'duplicate_count': must be an integer >= 1 (got {dup_raw!r})"
                )
            dup_v = int(float(dup_s))
            if dup_v < 1:
                raise ValueError(
                    f"reactionDataFile row {row_no} column 'duplicate_count': must be >= 1 (got {dup_v})"
                )

            kind_raw = row.get("kind", "")
            kind = "" if kind_raw is None else str(kind_raw).strip()
            if kind == "":
                raise ValueError(f"reactionDataFile row {row_no}: column 'kind' is missing")
            kinds.add(kind)

    if len(kinds) != constant_size:
        raise ValueError(
            f"reactionDataFile: number of unique kinds ({len(kinds)}) does not match constantSize ({constant_size}); kinds={sorted(kinds)}"
        )


@dataclass(frozen=True)
class ParsedYamlConfig:
    qasap_path: Path
    reaction_path: Path
    species: int
    constant_size: int
    tracked_names: list[str]
    tracked_indices: list[int]
    full_conc: list[float]
    init_conc: dict[int, float]
    tol_abs: float | None = None
    tol_rel: float | None = None
    scalar: float | None = None
    cross_over: float | None = None
    upper_lim: float | None = None
    lower_lim: float | None = None
    cvode_max_num_steps: int | None = None
    log_level: str | int | None = None


def validate_config_yaml(
    cfg: Mapping[str, Any],
    *,
    config_path: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
    validate_files: bool = True,
) -> ParsedYamlConfig:
    qasap_raw = _require_nonempty_str("QASAPDataFile", cfg.get("QASAPDataFile"))
    reaction_raw = _require_nonempty_str("reactionDataFile", cfg.get("reactionDataFile"))

    species = _require_int("species", cfg.get("species"), min_value=1)
    constant_size = _require_int("constantSize", cfg.get("constantSize"), min_value=1)

    tracked = cfg.get("trackedSpecies")
    if not isinstance(tracked, Mapping) or len(tracked) == 0:
        raise ValueError("trackedSpecies is required and must be a non-empty mapping")
    tracked_names: list[str] = []
    tracked_indices: list[int] = []
    full_conc: list[float] = []
    for name, info in tracked.items():
        if not isinstance(name, str) or name.strip() == "":
            raise ValueError("trackedSpecies keys must be non-empty strings")
        if not isinstance(info, Mapping):
            raise ValueError(f"trackedSpecies[{name!r}] must be a mapping")
        idx = _require_int(f"trackedSpecies[{name}].index", info.get("index"), min_value=0)
        if idx >= species:
            raise ValueError(
                f"trackedSpecies[{name}].index must be < species ({species}), got {idx}"
            )
        conc = _require_float(
            f"trackedSpecies[{name}]['100%concentration']",
            info.get("100%concentration"),
            min_value=0.0,
        )
        tracked_names.append(name)
        tracked_indices.append(idx)
        full_conc.append(conc)

    init_conc_raw = cfg.get("initConc")
    if not isinstance(init_conc_raw, Mapping) or len(init_conc_raw) == 0:
        raise ValueError("initConc is required and must be a non-empty mapping")
    init_conc: dict[int, float] = {}
    for k, v in init_conc_raw.items():
        if not _is_int_string(k):
            raise ValueError(f"initConc key must be an integer species index (got {k!r})")
        idx = int(float(str(k).strip()))
        if idx < 0 or idx >= species:
            raise ValueError(f"initConc index {idx} out of range [0, {species - 1}]")
        init_conc[idx] = _require_float(f"initConc[{idx}]", v, min_value=0.0)

    # Optional scalar params mirroring the YAML conventions
    tol_abs = cfg.get("tolAbsError")
    tol_rel = cfg.get("tolRelError")
    scalar = cfg.get("scalar")
    cross = cfg.get("crossOver")
    upper = cfg.get("upperLim")
    lower = cfg.get("lowerLim")
    cvode_steps = cfg.get("cvodeMaxNumSteps")
    log_level = cfg.get("logLevel")

    parsed = ParsedYamlConfig(
        qasap_path=resolve_path(qasap_raw, config_path=config_path, project_root=project_root),
        reaction_path=resolve_path(reaction_raw, config_path=config_path, project_root=project_root),
        species=species,
        constant_size=constant_size,
        tracked_names=tracked_names,
        tracked_indices=tracked_indices,
        full_conc=full_conc,
        init_conc=init_conc,
        tol_abs=_require_float("tolAbsError", tol_abs, min_value=0.0) if tol_abs is not None else None,
        tol_rel=_require_float("tolRelError", tol_rel, min_value=0.0) if tol_rel is not None else None,
        scalar=_require_float("scalar", scalar, min_value=0.0, max_value=1.0) if scalar is not None else None,
        cross_over=_require_float("crossOver", cross, min_value=0.0, max_value=1.0) if cross is not None else None,
        upper_lim=_require_float("upperLim", upper, min_value=0.0) if upper is not None else None,
        lower_lim=_require_float("lowerLim", lower, min_value=0.0) if lower is not None else None,
        cvode_max_num_steps=_require_int("cvodeMaxNumSteps", cvode_steps, min_value=1) if cvode_steps is not None else None,
        log_level=log_level,
    )
    if parsed.upper_lim is not None and parsed.lower_lim is not None and parsed.upper_lim < parsed.lower_lim:
        raise ValueError(f"upperLim must be >= lowerLim (got {parsed.upper_lim} < {parsed.lower_lim})")

    if validate_files:
        validate_qasap_csv_file(parsed.qasap_path, parsed.tracked_names)
        validate_reaction_csv_file(parsed.reaction_path, species=parsed.species, constant_size=parsed.constant_size)
    return parsed


def validate_constants_vector(constants: Sequence[float], *, expected_size: int | None = None) -> list[float]:
    if not isinstance(constants, Sequence) or len(constants) == 0:
        raise ValueError("constants must be a non-empty sequence")
    if expected_size is not None and len(constants) != expected_size:
        raise ValueError(f"constants must have length {expected_size} (got {len(constants)})")
    out: list[float] = []
    for i, v in enumerate(constants):
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"constants[{i}] is not numeric: {v!r}") from exc
        if not math.isfinite(fv) or fv <= 0:
            raise ValueError(f"constants[{i}] must be finite and > 0 (got {fv!r})")
        out.append(fv)
    return out


def validate_population(population: Sequence[Sequence[float]], *, constant_size: int, min_size: int = 3) -> list[list[float]]:
    if not isinstance(population, Sequence) or len(population) < min_size:
        raise ValueError(f"population must be a 2D sequence with at least {min_size} rows")
    out: list[list[float]] = []
    for row_i, row in enumerate(population):
        if not isinstance(row, Sequence):
            raise TypeError(f"population[{row_i}] is not a sequence")
        out.append(validate_constants_vector(row, expected_size=constant_size))
    return out


def build_termination_condition(
    core: Any,
    term: Mapping[str, Any] | None,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> Any:
    """Create _core.TerminationCondition from a dict.

    - Unknown keys: warning
    - Invalid values: ValueError
    """
    tc = core.TerminationCondition()
    allowed = {
        "maxIter": ("int", 0),
        "timeLimit": ("float", 0.0),
        "xtol": ("float", 0.0),
        "ftolAbs": ("float", 0.0),
        "ftolRel": ("float", 0.0),
        "targetError": ("float", 0.0),
        "stall": ("int", 0),
    }

    # apply defaults first
    if defaults:
        for k, v in defaults.items():
            if k not in allowed:
                continue
            _set_tc_field(tc, k, v, allowed)

    if term is None:
        return tc
    if not isinstance(term, Mapping):
        raise TypeError("terminationCondition must be a dict-like mapping")

    for k, v in term.items():
        if k not in allowed:
            warnings.warn(f"Unknown TerminationCondition key: {k!r}", UserWarning, stacklevel=2)
            continue
        _set_tc_field(tc, k, v, allowed)
    return tc


def _set_tc_field(tc: Any, key: str, value: Any, allowed: Mapping[str, tuple[str, Any]]) -> None:
    kind, min_v = allowed[key]
    if kind == "int":
        iv = _require_int(f"terminationCondition.{key}", value, min_value=int(min_v))
        setattr(tc, key, iv)
    else:
        fv = _require_float(f"terminationCondition.{key}", value, min_value=float(min_v))
        setattr(tc, key, fv)