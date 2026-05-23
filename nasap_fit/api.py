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
    """Return a default ``_core.Config`` object with factory settings.

    Returns
    -------
    Any
        A ``_core.Config`` instance populated with default values.
    """
    return _core.default_config()


def config_from_yaml(
    config_yaml_path: str | Path,
    *,
    project_root: str | Path | None = None,
    validate_files: bool = True,
) -> Any:
    """Load a YAML configuration file, validate it, and return a ``_core.Config`` object.

    Relative file paths inside the YAML (``QASAPDataFile``, ``reactionDataFile``)
    are resolved relative to the YAML file's own directory unless *project_root*
    overrides that base.

    Parameters
    ----------
    config_yaml_path : str or Path
        Path to the ``.yaml`` configuration file.
    project_root : str, Path, or None, optional
        Base directory used to resolve relative paths inside the YAML.
        When ``None`` (default), paths are resolved relative to the directory
        that contains the YAML file.
    validate_files : bool, optional
        When ``True`` (default), checks that the CSV files referenced by the
        YAML exist and have the expected format.

    Returns
    -------
    Any
        A ``_core.Config`` object ready to be passed to :class:`NasapFit`.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    ValueError
        If the YAML structure is invalid or referenced files fail validation.
    """
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
    """High-performance fitting engine for NASAP reaction-rate-constant optimization.

    Wraps a C++ backend (SUNDIALS / SuiteSparse) and exposes Python-friendly
    methods for global search (Differential Evolution) and local refinement
    (Levenberg-Marquardt), as well as forward simulation and Hessian computation.

    The recommended construction path is :meth:`from_yaml`.
    """

    def __init__(
        self,
        cfg: Any | None = None,
        qasap_data: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Initialize the fitting engine.

        Parameters
        ----------
        cfg : Any or None, optional
            A ``_core.Config`` object (e.g. returned by :func:`config_from_yaml`).
            When ``None``, the engine is created with default configuration.
        qasap_data : sequence of sequence of float, or None, optional
            In-memory QASAP data for validation purposes.  Providing this does
            **not** replace the CSV file specified in *cfg*; it is used only
            to validate the shape of the data at construction time.

        Raises
        ------
        ValueError
            If *qasap_data* has an unexpected number of columns.
        """
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
        """Construct a :class:`NasapFit` engine from a YAML configuration file.

        Parameters
        ----------
        config_yaml_path : str or Path
            Path to the ``.yaml`` configuration file (relative to CWD, or absolute).
        project_root : str, Path, or None, optional
            Base directory used to resolve relative paths inside the YAML.
            Defaults to the directory that contains the YAML file.
        validate_files : bool, optional
            When ``True`` (default), validates that the CSV files referenced by
            the YAML exist and have the expected format.

        Returns
        -------
        NasapFit
            A fully initialised fitting engine.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        ValueError
            If the YAML or referenced files are invalid.
        """
        cfg = config_from_yaml(config_yaml_path, project_root=project_root, validate_files=validate_files)
        return cls(cfg)

    @property
    def config(self) -> Any:
        return self._engine.constants()

    def calc_error(self, constants: Sequence[float]) -> float:
        """Compute the sum of squared residuals between simulation and QASAP data.

        Parameters
        ----------
        constants : sequence of float
            Reaction rate constants.  Must contain exactly ``reaction_count``
            positive finite values.

        Returns
        -------
        float
            Sum of squared residuals (SSR) between the simulated concentrations
            and the QASAP reference data.

        Raises
        ------
        ValueError
            If *constants* has the wrong length or contains non-positive values.
        """
        constant_size = int(self._engine.constants().constantSize)
        vec = validate_constants_vector(constants, expected_size=constant_size)
        return float(self._engine.calcError(vec))

    def calc_nrmse(self, error: float) -> float:
        """Convert a raw SSR value to Normalised Root Mean Squared Error (NRMSE).

        Parameters
        ----------
        error : float
            Sum of squared residuals as returned by :meth:`calc_error`.
            Must be finite and non-negative.

        Returns
        -------
        float
            NRMSE normalised against the QASAP reference data.

        Raises
        ------
        ValueError
            If *error* is negative or non-finite.
        """
        err = float(error)
        if not math.isfinite(err) or err < 0.0:
            raise ValueError(f"error must be finite and >= 0 (got {error!r})")
        return float(self._engine.calcNRMSEFromError(err))

    @property
    def reaction_count(self) -> int:
        """Total number of reactions defined in the reaction-network file.

        Returns
        -------
        int
            Number of reactions.
        """
        return int(self._engine.reactionCount())

    @property
    def rate_constant_index(self) -> Mapping[str, int]:
        """Mapping from rate-constant label to its index in the constants vector.

        Returns
        -------
        Mapping[str, int]
            Dictionary keyed by rate-constant label (e.g. ``"k1"``) with the
            corresponding zero-based index into the constants sequence.
        """
        # pybind11 converts std::map to dict-like; normalize to a plain dict.
        return dict(self._engine.termIndex())

    def run_de(
        self,
        pop_size: int,
        termination_condition: Mapping[str, Any],
        lower_bound: float = 1e-3,
        upper_bound: float = 1e4,
        seed: int = 1,
    ) -> list[Any]:
        """Run Differential Evolution (DE) with randomly generated initial population.

        Initial rate constants are sampled from a log-uniform distribution on
        ``[lower_bound, upper_bound)``.

        Parameters
        ----------
        pop_size : int
            Number of candidate solutions (must be >= 3).
        termination_condition : Mapping[str, Any]
            Stopping criteria.  Recognised keys are ``maxIter``, ``timeLimit``,
            ``xtol``, ``ftolAbs``, ``ftolRel``, ``targetError``, and ``stall``.
            At least one must be set.
        lower_bound : float, optional
            Lower bound for the randomly generated rate constants (default ``1e-3``).
        upper_bound : float, optional
            Upper bound for the randomly generated rate constants (default ``1e4``).
        seed : int, optional
            Random seed for reproducibility (default ``1``).

        Returns
        -------
        list
            Optimised population sorted in ascending order of SSR.  Each
            element has ``.constants`` (list of float) and ``.error`` (float)
            attributes.

        Raises
        ------
        ValueError
            If *termination_condition* is empty or contains invalid values.
        """
        term = build_termination_condition(_core, termination_condition)
        return list(self._engine.runDE(int(pop_size), float(lower_bound), float(upper_bound), term, int(seed)))

    def run_de_from_population(
        self,
        population: Sequence[Sequence[float]],
        termination_condition: Mapping[str, Any],
        seed: int = 1,
    ) -> list[Any]:
        """Run Differential Evolution starting from a user-supplied initial population.

        Parameters
        ----------
        population : sequence of sequence of float
            Initial rate-constant vectors.  Must contain at least 3 rows, each
            with exactly ``reaction_count`` positive finite values.
        termination_condition : Mapping[str, Any]
            Stopping criteria (same keys as :meth:`run_de`).
        seed : int, optional
            Random seed for reproducibility (default ``1``).

        Returns
        -------
        list
            Optimised population (same structure as :meth:`run_de`).

        Raises
        ------
        ValueError
            If *population* is too small or *termination_condition* is invalid.
        """
        constant_size = int(self._engine.constants().constantSize)
        normalized = validate_population(population, constant_size=constant_size)
        term = build_termination_condition(_core, termination_condition)
        return list(self._engine.runDE(normalized, term, int(seed)))
    

    def simulate(
            self,
            t: Sequence[float],
            constants: Sequence[float],
            reaction_ids: Iterable[int],
        ) -> Any:
            """Simulate concentration profiles and reaction progress.

            Parameters
            ----------
            t : sequence of float
                Time points (minutes) at which to record the state.
                All values must be non-negative.
            constants : sequence of float
                Reaction rate constants.
            reaction_ids : iterable of int
                Zero-based indices of reactions whose progress to track.
                Each value must satisfy ``0 <= reaction_id < reaction_count``.

            Returns
            -------
            ReactionProgressResult
                An object with the following attributes:

                - ``timePoints`` - number of time points recorded.
                - ``t`` - list of recorded time points.
                - ``y`` - 2-D array of concentrations; ``y[time_index][species_index]``
                  is the concentration of species *species_index* at ``t[time_index]``.
                - ``reactionProgress.reaction_ids`` - tracked reaction IDs.
                - ``reactionProgress.J`` - 2-D array of cumulative reaction progress
                  (mol/L, i.e. extent of reaction ξ divided by volume V);
                  ``J[time_index][i]`` is the time-integrated rate
                  ``∫₀ᵗ rate_i(τ) dτ`` for the *i*-th entry of *reaction_ids*
                  at ``t[time_index]``.
                - ``reactionProgress.reaction_labels`` - human-readable label for
                  each tracked reaction (e.g. ``"1 + 2 -> 3 + 4"``).

            Raises
            ------
            ValueError
                If any time point is negative, a reaction ID is out of range, or
                *constants* is invalid.
            """
            validate_constants_vector(constants, expected_size=int(self._engine.constants().constantSize))
            for v in t:
                if v < 0.0:
                    raise ValueError(f"Time points must be non-negative (got {v})")
            id_upper = self.reaction_count
            for i in reaction_ids:
                if not (0 <= i < id_upper):
                    raise ValueError(f"Reaction IDs must be in [0, {id_upper}) (got {i})")
            t_vec = [float(v) for v in t]
            c_vec = [float(v) for v in constants]
            ids = [int(i) for i in reaction_ids]
            return self._engine.simulate(t_vec, c_vec, ids)


    def run_lm(self, initial_constants: Sequence[float], termination_condition: Mapping[str, Any]) -> Any:
        """Run the Levenberg-Marquardt algorithm from a single initial point.

        Parameters
        ----------
        initial_constants : sequence of float
            Starting rate-constant vector.
        termination_condition : Mapping[str, Any]
            Stopping criteria (same keys as :meth:`run_de`).

        Returns
        -------
        Any
            Optimised rate-constant vector with ``.constants`` and ``.error``
            attributes.

        Raises
        ------
        ValueError
            If *initial_constants* is invalid or *termination_condition* is empty.
        """
        constant_size = int(self._engine.constants().constantSize)
        vec = validate_constants_vector(initial_constants, expected_size=constant_size)
        term = build_termination_condition(_core, termination_condition)
        return self._engine.runLM(vec, term)

    def run_lm_batch(
        self,
        initial_constants: Sequence[Sequence[float]],
        termination_condition: Mapping[str, Any],
    ) -> list[Any]:
        """Run Levenberg-Marquardt in parallel from multiple initial points.

        Requires MPI support compiled into the package.  When MPI is not
        available, the method still runs but processes the batch sequentially.

        Parameters
        ----------
        initial_constants : sequence of sequence of float
            One or more starting rate-constant vectors.
        termination_condition : Mapping[str, Any]
            Stopping criteria (same keys as :meth:`run_de`).

        Returns
        -------
        list
            Optimised vectors, one per input row (same structure as
            :meth:`run_lm`).

        Raises
        ------
        ValueError
            If any row of *initial_constants* is invalid.
        """
        constant_size = int(self._engine.constants().constantSize)
        normalized = validate_population(initial_constants, constant_size=constant_size, min_size=1)
        term = build_termination_condition(_core, termination_condition)
        return list(self._engine.runLM(normalized, term))

    def gauss_newton_hessian(self, point: Sequence[float]) -> list[list[float]]:
        """Compute the Gauss-Newton approximation to the SSR Hessian at *point*.

        Parameters
        ----------
        point : sequence of float
            Rate-constant vector at which to evaluate the Hessian.

        Returns
        -------
        list of list of float
            Square Hessian matrix of size ``(reaction_count, reaction_count)``.

        Raises
        ------
        ValueError
            If *point* has the wrong length or contains invalid values.
        """
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
