import math
from typing import Sequence


def validate_qasap_data(data: Sequence[Sequence[float]], expected_columns: int) -> None:
    if not isinstance(data, Sequence) or len(data) == 0:
        raise ValueError("qasap_data must be a non-empty 2D sequence")

    previous_time = None
    for row_index, row in enumerate(data):
        if not isinstance(row, Sequence):
            raise TypeError(f"row {row_index} is not a sequence")
        if len(row) != expected_columns:
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
            numeric_row.append(numeric_value)

        current_time = numeric_row[0]
        if current_time < 0:
            raise ValueError(f"row {row_index} time must be >= 0")
        if previous_time is not None and current_time < previous_time:
            raise ValueError("time values must be non-decreasing")
        previous_time = current_time