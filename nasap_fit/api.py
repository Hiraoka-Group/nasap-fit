from __future__ import annotations

from typing import Any, Iterable, Sequence

from . import _core  # type: ignore
from ._validation import validate_qasap_data


def expected_input_columns() -> int:
    # time + trackedSpecies columns
    return int(_core.default_config().trackedSpecies) + 1


def default_config() -> Any:
    return _core.default_config()


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

    @property
    def config(self) -> Any:
        return self._engine.constants()

    def run_de(
        self,
        max_gen: int,
        pop_size: int,
        lower_lim: float = 1e-3,
        upper_lim: float = 1e4,
    ) -> list[Any]:
        return list(self._engine.runDE(int(max_gen), int(pop_size), float(lower_lim), float(upper_lim)))

    def run_de_from_population(self, population: Sequence[Sequence[float]]) -> list[Any]:
        normalized = [[float(v) for v in row] for row in population]
        return list(self._engine.runDE(normalized))

    def run_lm(self, theta0: Sequence[float]) -> Any:
        vec = [float(v) for v in theta0]
        return self._engine.runLM(vec)

    def run_lm_batch(self, thetas: Sequence[Sequence[float]]) -> list[Any]:
        normalized = [[float(v) for v in row] for row in thetas]
        return list(self._engine.runLM(normalized))

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