from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


from . import _core  # type: ignore
from ._validation import (
    build_termination_condition,
    find_project_root,
    validate_config_yaml,
    validate_constants_vector,
    validate_population,
    validate_qasap_data,
)


def expected_input_columns() -> int:
    # time + trackedSpecies columns
    return int(_core.default_config().trackedSpecies) + 1


def default_config() -> Any:
    return _core.default_config()


def config_from_yaml(
    config_yaml_path: str | Path,
    *,
    project_root: str | Path | None = None,
    validate_files: bool = True,
) -> Any:
    """Load config.yaml (NASAP format), validate it, and convert to _core.Config."""
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to load config.yaml") from exc

    config_yaml_path = Path(config_yaml_path)
    with config_yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("config.yaml must parse to a mapping")

    root = project_root
    if root is None:
        root = find_project_root(config_yaml_path)
    parsed = validate_config_yaml(
        raw,
        config_path=str(config_yaml_path),
        project_root=str(root) if root is not None else None,
        validate_files=validate_files,
    )

    cfg = _core.default_config()
    # file paths
    cfg.QASAPFile = str(parsed.qasap_path)
    cfg.reactNetworkFile = str(parsed.reaction_path)
    # core sizes
    cfg.species = int(parsed.species)
    cfg.constantSize = int(parsed.constant_size)
    cfg.trackedSpecies = int(len(parsed.tracked_names))
    cfg.trackedNames = list(parsed.tracked_names)
    cfg.trackedIndex = list(parsed.tracked_indices)
    cfg.fullConc = list(parsed.full_conc)
    cfg.initConc = dict(parsed.init_conc)

    # optional tuning
    if parsed.tol_abs is not None:
        cfg.tolAbsError = float(parsed.tol_abs)
    if parsed.tol_rel is not None:
        cfg.tolRelError = float(parsed.tol_rel)
    if parsed.scalar is not None:
        cfg.scalar = float(parsed.scalar)
    if parsed.cross_over is not None:
        cfg.crossOver = float(parsed.cross_over)
    if parsed.upper_lim is not None:
        cfg.upperLim = float(parsed.upper_lim)
    if parsed.lower_lim is not None:
        cfg.lowerLim = float(parsed.lower_lim)
    if parsed.cvode_max_num_steps is not None:
        cfg.cvodeMaxNumSteps = int(parsed.cvode_max_num_steps)
    if parsed.log_level is not None:
        cfg.logLevel = _parse_log_level(parsed.log_level)
    return cfg


def _parse_log_level(v: Any) -> Any:
    if isinstance(v, _core.LogLevel):
        return v
    if isinstance(v, int):
        # match enum underlying values: quiet=0, normal=1, verbose=2
        if v == 0:
            return _core.LogLevel.quiet
        if v == 1:
            return _core.LogLevel.normal
        if v == 2:
            return _core.LogLevel.verbose
        raise ValueError(f"logLevel int must be 0/1/2 (got {v!r})")
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("quiet", "q", "0"):
            return _core.LogLevel.quiet
        if s in ("normal", "n", "1"):
            return _core.LogLevel.normal
        if s in ("verbose", "v", "2"):
            return _core.LogLevel.verbose
        raise ValueError(f"logLevel must be one of quiet/normal/verbose (got {v!r})")
    raise TypeError(f"logLevel must be str|int|LogLevel (got {type(v).__name__})")


class NASAP_fit:
    def __init__(
        self,
        cfg: Any | None = None,
        qasap_data: Sequence[Sequence[float]] | None = None,
    ) -> None:
        # NOTE: the C++ core currently loads QASAP data from cfg.QASAPFile at construction time.
        # This wrapper accepts qasap_data for validation/backwards-compatibility only.
        # If you want in-memory QASAP data injection, extend the bindings to expose setQASAPData + setUpCasADiFunctions.
        if qasap_data is not None:
            columns = expected_input_columns()
            validate_qasap_data(qasap_data, expected_columns=columns)

        self._engine = _core.NASAP_fit() if cfg is None else _core.NASAP_fit(cfg)

    @classmethod
    def from_yaml(
        cls,
        config_yaml_path: str | Path,
        *,
        project_root: str | Path | None = None,
        validate_files: bool = True,
    ) -> "NASAP_fit":
        cfg = config_from_yaml(config_yaml_path, project_root=project_root, validate_files=validate_files)
        return cls(cfg)

    @property
    def config(self) -> Any:
        return self._engine.constants()

    def run_de(
        self,
        max_gen: int,
        pop_size: int,
        lower_lim: float = 1e-3,
        upper_lim: float = 1e4,
        terminationCondition: Mapping[str, Any] | None = None,
        seed: int = 1,
    ) -> list[Any]:
        term_defaults = {"maxIter": int(max_gen)}
        term = build_termination_condition(_core, terminationCondition, defaults=term_defaults)
        return list(self._engine.runDE(int(pop_size), float(lower_lim), float(upper_lim), term, int(seed)))

    def run_de_from_population(
        self,
        population: Sequence[Sequence[float]],
        terminationCondition: Mapping[str, Any] | None = None,
        seed: int = 1,
    ) -> list[Any]:
        constant_size = int(self._engine.constants().constantSize)
        normalized = validate_population(population, constant_size=constant_size)
        term = build_termination_condition(_core, terminationCondition)
        return list(self._engine.runDE(normalized, term, int(seed)))

    def run_lm(self, theta0: Sequence[float], terminationCondition: Mapping[str, Any] | None = None) -> Any:
        constant_size = int(self._engine.constants().constantSize)
        vec = validate_constants_vector(theta0, expected_size=constant_size)
        term = build_termination_condition(_core, terminationCondition)
        return self._engine.runLM(vec, term)

    def run_lm_batch(
        self,
        thetas: Sequence[Sequence[float]],
        terminationCondition: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        constant_size = int(self._engine.constants().constantSize)
        normalized = validate_population(thetas, constant_size=constant_size, min_size=1)
        term = build_termination_condition(_core, terminationCondition)
        return list(self._engine.runLM(normalized, term))

    def get_hessian(self, point: Sequence[float]) -> list[list[float]]:
        vec = [float(v) for v in point]
        return [list(row) for row in self._engine.getHessian(vec)]

#    def get_hessian_parallel(self, point: Sequence[float]) -> list[list[float]]:
#        vec = [float(v) for v in point]
#        return [list(row) for row in self._engine.getHessian_parallel(vec)]

    def pseudo_hessian(self, point: Sequence[float]) -> list[list[float]]:
        vec = [float(v) for v in point]
        return [list(row) for row in self._engine.pseudoHessian(vec)]

#    def put_cvode_sim(self, constant: Sequence[float]) -> None:
#        vec = [float(v) for v in constant]
#        self._engine.putCVODESim(vec)

    def simulate(
        self,
        t: Sequence[float],
        constant: Sequence[float],
        reaction_ids: Iterable[int],
    ) -> Any:
        t_vec = [float(v) for v in t]
        c_vec = [float(v) for v in constant]
        ids = [int(i) for i in reaction_ids]
        return self._engine.simulate(t_vec, c_vec, ids)