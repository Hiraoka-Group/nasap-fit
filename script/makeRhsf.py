import numpy as np
import pandas as pd
import csv
import pprint
from pathlib import Path

#inputFile = Path(__file__).parent.parent / 'data' / 'M9L6' / 'simply_classified_reactions.csv'
inputFile = Path(__file__).parent.parent / 'data' / 'classified_reactions_str.csv'
outputFile = Path(__file__).parent.parent / 'include' / 'Rhsf.hpp'
df = pd.read_csv(inputFile, index_col=0)

records=df.to_dict(orient="records")
reaction_kinds=df["kind"].unique() #反応速度定数の種類
reaction_kinds.sort()
typeToIndex={} #反応タイプから速度定数のindexへの変換
for i in range(len(reaction_kinds)):
    typeToIndex[reaction_kinds[i]]=i

species=0 #化学種の種類　これをもとに化学種の量のarrayを作る



for dict in records:
    species=max(species,dict["init_assem_id"])
    species=max(species,dict["entering_assem_id"])
    species=max(species,dict["product_assem_id"])
    species=max(species,dict["leaving_assem_id"])
species=round(species)
species+=1
ODE=[{} for _ in range(species)] #常微分方程式の素

def add_dict(mp,key,value):
    if key in mp:
        mp[key]+=value
    else:
        mp[key]=value

def to_fomula(tup,value):
        ret=""
        if tup[1] is None:
            return f"({value} * k[{typeToIndex[tup[2]]}] * sp[{tup[0]}])"
        else:
            return f"({value} * k[{typeToIndex[tup[2]]}] * sp[{tup[0]}] * sp[{tup[1]}])"

for dict in records:
    init=dict["init_assem_id"]
    entering=None if pd.isna(dict["entering_assem_id"]) else round(dict["entering_assem_id"])
    product=dict["product_assem_id"]
    leaving=None if pd.isna(dict["leaving_assem_id"]) else round(dict["leaving_assem_id"])

    hoge=(init,entering,dict["kind"])
    
    add_dict(ODE[init],hoge,-dict["duplicate_count"])
    if entering is not None:
        add_dict(ODE[entering],hoge,-dict["duplicate_count"])
    add_dict(ODE[product],hoge,dict["duplicate_count"])
    if leaving is not None:
        add_dict(ODE[leaving],hoge,dict["duplicate_count"])



with open(outputFile,"w") as f:
    f.write(
        "#pragma once\n"                                                                              \
        "#include <array>\n"                                                                          \
        "#include <nvector/nvector_serial.h>\n"                                                       \
        "#include \"constants.hpp\"\n\n"                                                              \
        "int rhsf(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {\n"                     \
        "\tauto sp = N_VGetArrayPointer(y);\n"                                                        \
        "\tauto ydotData = N_VGetArrayPointer(ydot);\n"                                               \
        "\tstd::array<double, config::constantSize> &k = *static_cast<std::array<double, config::constantSize>*>(user_data);\n\t")
    for t in ODE:
        isFirst=True
        f.write("ydotData[" + str(ODE.index(t)) + "] = ")
        for key,val in t.items():
            if isFirst:
                isFirst=False
            else:
                f.write(" + ")
            f.write(to_fomula(key,val))
        f.write(";\n\t")
    f.write("return 0;\n}")
