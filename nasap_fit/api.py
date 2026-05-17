from __future__ import annotations

import math
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

    # Spec: resolve relative paths inside YAML relative to the YAML file location.
    # Do NOT auto-detect repo/project root; it makes behavior depend on presence of .git/pyproject.toml/CMakeLists.txt.
    root = project_root
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


class NasapFit:
    def __init__(
        self,
        cfg: Any | None = None,
        qasap_data: Sequence[Sequence[float]] | None = None,
    ) -> None:
        # NOTE: the C++ core currently loads QASAP data from cfg.QASAPFile at construction time.
        # This wrapper accepts qasap_data for validation/backwards-compatibility only.
        # If you want in-memory QASAP data injection, extend the bindings to expose setQASAPData + setUpCasADiFunctions.
        if qasap_data is not None:
            if cfg is not None and hasattr(cfg, "trackedSpecies"):
                tracked_species = int(getattr(cfg, "trackedSpecies"))
            else:
                tracked_species = int(_core.default_config().trackedSpecies)
            # time + trackedSpecies columns
            columns = tracked_species + 1
            validate_qasap_data(qasap_data, expected_columns=columns)

        self._engine = _core.NasapFit() if cfg is None else _core.NasapFit(cfg)

    @classmethod
    def from_yaml(
        cls,
        config_yaml_path: str | Path,
        *,
        project_root: str | Path | None = None,
        validate_files: bool = True,
    ) -> "NasapFit":
        cfg = config_from_yaml(config_yaml_path, project_root=project_root, validate_files=validate_files)
        return cls(cfg)

    @property
    def config(self) -> Any:
        return self._engine.constants()

    def calcError(self, constant: Sequence[float]) -> float:
        constant_size = int(self._engine.constants().constantSize)
        vec = validate_constants_vector(constant, expected_size=constant_size)
        return float(self._engine.calcError(vec))

    def errorToNRMSE(self, error: float) -> float:
        err = float(error)
        if not math.isfinite(err) or err < 0.0:
            raise ValueError(f"error must be finite and >= 0 (got {error!r})")
        return float(self._engine.calcNRMSEFromError(err))

    def reactionCount(self) -> int:
        return int(self._engine.reactionCount())

    def termIndex(self) -> Mapping[str, int]:
        # pybind11 converts std::map to dict-like; normalize to a plain dict.
        return dict(self._engine.termIndex())

    def run_de(
        self,
        pop_size: int,
        terminationCondition: Mapping[str, Any],
        lower_lim: float = 1e-3,
        upper_lim: float = 1e4,
        seed: int = 1,
    ) -> list[Any]:
        term = build_termination_condition(_core, terminationCondition)
        return list(self._engine.runDE(int(pop_size), float(lower_lim), float(upper_lim), term, int(seed)))

    def run_de_from_population(
        self,
        population: Sequence[Sequence[float]],
        terminationCondition: Mapping[str, Any],
        seed: int = 1,
    ) -> list[Any]:
        constant_size = int(self._engine.constants().constantSize)
        normalized = validate_population(population, constant_size=constant_size)
        term = build_termination_condition(_core, terminationCondition)
        return list(self._engine.runDE(normalized, term, int(seed)))
    

    def simulate(
            self,
            t: Sequence[float],
            constant: Sequence[float],
            reaction_ids: Iterable[int],
        ) -> Any:
            validate_constants_vector(constant, expected_size=int(self._engine.constants().constantSize))
            for v in t:
                if v < 0.0:
                    raise ValueError(f"Time points must be non-negative (got {v})")
            id_upper = self.reactionCount()
            for i in reaction_ids:
                if not (0 <= i < id_upper):
                    raise ValueError(f"Reaction IDs must be in [0, {id_upper}) (got {i})")
            t_vec = [float(v) for v in t]
            c_vec = [float(v) for v in constant]
            ids = [int(i) for i in reaction_ids]
            return self._engine.simulate(t_vec, c_vec, ids)


    def run_lm(self, theta0: Sequence[float], terminationCondition: Mapping[str, Any]) -> Any:
        constant_size = int(self._engine.constants().constantSize)
        vec = validate_constants_vector(theta0, expected_size=constant_size)
        term = build_termination_condition(_core, terminationCondition)
        return self._engine.runLM(vec, term)

    def run_lm_batch(
        self,
        thetas: Sequence[Sequence[float]],
        terminationCondition: Mapping[str, Any],
    ) -> list[Any]:
        constant_size = int(self._engine.constants().constantSize)
        normalized = validate_population(thetas, constant_size=constant_size, min_size=1)
        term = build_termination_condition(_core, terminationCondition)
        return list(self._engine.runLM(normalized, term))
    
    #log point に対する残差二乗和のGaussNewtonHessianを導出
    def GaussNewtonHessian(self, point: Sequence[float]) -> list[list[float]]:
        validate_constants_vector(point, expected_size=int(self._engine.constants().constantSize))
        vec = [float(v) for v in point]
        return [list(row) for row in self._engine.GaussNewtonHessian(vec)]
"""
    def get_hessian(self, point: Sequence[float]) -> list[list[float]]:
        validate_constants_vector(point, expected_size=int(self._engine.constants().constantSize))
        vec = [float(v) for v in point]
        return [list(row) for row in self._engine.getHessian(vec)]

#    def get_hessian_parallel(self, point: Sequence[float]) -> list[list[float]]:
#        vec = [float(v) for v in point]
#        return [list(row) for row in self._engine.getHessian_parallel(vec)]
"""


#    def put_cvode_sim(self, constant: Sequence[float]) -> None:
#        vec = [float(v) for v in constant]
#        self._engine.putCVODESim(vec)
