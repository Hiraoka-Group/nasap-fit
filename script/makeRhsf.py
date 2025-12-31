import numpy as np
import pandas as pd
import csv
import pprint
from pathlib import Path

def make_rhsf(input_path: Path | None = None) -> None:
    base = Path(__file__).parent.parent
    inputFile = Path(input_path) if input_path else base / 'data' / 'classified_reactions_str.csv'
    outputFile = base / 'include' / 'Rhsf.hpp'
    df = pd.read_csv(inputFile, index_col=0)

    records = df.to_dict(orient="records")
    reaction_kinds = list(df["kind"])
    reaction_kinds = [rk.strip() if isinstance(rk, str) else rk for rk in reaction_kinds]
    reaction_kinds = list(dict.fromkeys(reaction_kinds).keys())
    reaction_kinds.sort()

    typeToIndex = {reaction_kinds[i]: i for i in range(len(reaction_kinds))}

    species = 0
    for rec in records:
        species = max(species, rec["init_assem_id"])
        species = max(species, rec["entering_assem_id"])
        species = max(species, rec["product_assem_id"])
        species = max(species, rec["leaving_assem_id"])
    species = round(species) + 1

    ODE = [dict() for _ in range(species)]

    def add_dict(mp: dict, key, value: float):
        mp[key] = mp.get(key, 0) + value

    def to_formula(tup, value):
        if tup[1] is None:
            return f"({value} * k[{typeToIndex[tup[2]]}] * sp[{tup[0]}])"
        else:
            return f"({value} * k[{typeToIndex[tup[2]]}] * sp[{tup[0]}] * sp[{tup[1]}])"

    for rec in records:
        init = rec["init_assem_id"]
        entering = None if pd.isna(rec["entering_assem_id"]) else round(rec["entering_assem_id"])
        product = rec["product_assem_id"]
        leaving = None if pd.isna(rec["leaving_assem_id"]) else round(rec["leaving_assem_id"])

        key = (init, entering, rec["kind"])
        add_dict(ODE[init], key, -rec["duplicate_count"])
        if entering is not None:
            add_dict(ODE[entering], key, -rec["duplicate_count"])
        add_dict(ODE[product], key, rec["duplicate_count"])
        if leaving is not None:
            add_dict(ODE[leaving], key, rec["duplicate_count"])

    with open(outputFile, "w") as f:
        f.write(
            "#if USE_PREGENERATED_RHSF\n\n"
            "#pragma once\n"
            "#include <array>\n"
            "#include <nvector/nvector_serial.h>\n"
            "#include \"constants.hpp\"\n\n"
            "int rhsf(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {\n"
            "\tauto sp = N_VGetArrayPointer(y);\n"
            "\tauto ydotData = N_VGetArrayPointer(ydot);\n"
            "\tstd::array<double, config::constantSize> &k = *static_cast<std::array<double, config::constantSize>*>(user_data);\n\t"
        )
        for idx, t in enumerate(ODE):
            f.write(f"ydotData[{idx}] = ")
            first = True
            for key, val in t.items():
                if not first:
                    f.write(" + ")
                first = False
                f.write(to_formula(key, val))
            f.write(";\n\t")
        f.write("return 0;\n}\n\n#endif // USE_PREGENERATED_RHSF\n")


if __name__ == "__main__":
    make_rhsf()
