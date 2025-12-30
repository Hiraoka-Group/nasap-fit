import numpy as np
import pandas as pd
import csv
import pprint
from collections import OrderedDict
from pathlib import Path

#inputFile = Path(__file__).parent.parent / 'data' / 'M9L6' / 'simply_classified_reactions.csv'
inputFile = Path(__file__).parent.parent / 'data' / 'classified_reactions_str.csv'
outputFile = Path(__file__).parent.parent / 'include' / 'Jacf.hpp'
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
matDict=OrderedDict() #ヤコビ行列の素 key:(row,col) value:keyが(k[i],sp[j]),valueがその係数のmap

def add_matDict(row,col,keyTup,value):
    #assert row is not None and col is not None
    key=(row,col)
    if not (key in matDict):
        matDict[key]=OrderedDict()
    if keyTup in matDict[key]:
        matDict[key][keyTup]+=value
    else:
        matDict[key][keyTup]=value

def to_fomula(tup,value):
        ret=""
        if tup[0] is None:
            return f"({value} * k[{typeToIndex[tup[1]]}])"
        else:
            return f"({value} * k[{typeToIndex[tup[1]]}] * sp[{tup[0]}])"

for dict in records:
    init=dict["init_assem_id"]
    entering=None if pd.isna(dict["entering_assem_id"]) else round(dict["entering_assem_id"])
    product=dict["product_assem_id"]
    leaving=None if pd.isna(dict["leaving_assem_id"]) else round(dict["leaving_assem_id"])

    hoge=(init,entering,dict["kind"])

    # A + B -> C + D のとき
    # dA/dt = - k * A * B
    # dA'/dA = - k * B
    # dA'/dB = - k * A
    if entering is not None:
        add_matDict(init,entering,(init,dict["kind"]), -dict["duplicate_count"])
    add_matDict(init,init,(entering,dict["kind"]), -dict["duplicate_count"])
    if entering is not None:
        # dB/dt = - k * A * B
        # dB'/dA = - k * B
        # dB'/dB = - k * A
        add_matDict(entering,entering,(init,dict["kind"]), -dict["duplicate_count"])
        add_matDict(entering,init,(entering,dict["kind"]), -dict["duplicate_count"])
    # dC/dt = k * A * B
    # dC'/dA = k * B
    # dC'/dB = k * A
    if entering is not None:
        add_matDict(product,entering,(init,dict["kind"]), dict["duplicate_count"])
    add_matDict(product,init,(entering,dict["kind"]), dict["duplicate_count"])
    if leaving is not None:
        # dD/dt = k * A * B
        # dD'/dA = k * B
        # dD'/dB = k * A
        if entering is not None:
            add_matDict(leaving,entering,(init,dict["kind"]), dict["duplicate_count"])
        add_matDict(leaving,init,(entering,dict["kind"]), dict["duplicate_count"])

# non zero elements of Jacobian matrix
nnz=len(matDict)

matDict = OrderedDict(sorted(matDict.items(), key=lambda x: (x[0][1], x[0][0])))

numsInColumn = [0 for _ in range(species)]
for key,val in matDict.items():
    col=key[1]
    numsInColumn[col]+=1

intro= \
"#if USE_PREGENERATED_JACOBIAN==0\n\n" \
"#pragma once\n" \
"#include <array>\n" \
"#include <vector>\n" \
"#include <nvector/nvector_serial.h>\n" \
"#include <sundials/sundials_types.h>\n" \
"#include <sunmatrix/sunmatrix_sparse.h>\n" \
"#include \"constants.hpp\"\n\n" \
f"constexpr size_t nonZeroElems={nnz};\n\n" \
"int JacFn(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3){\n" \
"\tauto sp = N_VGetArrayPointer(y);\n"  \
"\tstd::array<double, config::constantSize> &k = *static_cast<std::array<double, config::constantSize>*>(user_data);\n" \
"\tsunindextype* Jp = SUNSparseMatrix_IndexPointers(Jac);\n" \
"\tsunindextype* Ji = SUNSparseMatrix_IndexValues(Jac);\n" \
"\tsunrealtype* Jx = SUNSparseMatrix_Data(Jac);\n"

JpAssign="\tstd::vector<int>Jp_vals ={0"
sum=0
for num in numsInColumn:
    sum+=num
    JpAssign+=f", {sum}"
JpAssign+="};\n"
JpAssign+=f"\tfor(int i=0;i<{len(numsInColumn)+1};i++) Jp[i]=Jp_vals[i]; \n"

JiAssign="\tstd::vector<int>Ji_vals ={"
for key,val in matDict.items():
    row = key[0]
    JiAssign+=f"{row}, "
JiAssign=JiAssign[:-2]+"};\n"
JiAssign+="\tfor(int i=0;i<nonZeroElems;i++) Ji[i]=Ji_vals[i]; \n"

JxAssign=""
for i, (key, val) in enumerate(matDict.items()):
    row=key[0]
    col=key[1]
    JxAssign+=f"\tJx[{i}] = "
    for keyTup,value in val.items():
        JxAssign+=to_fomula(keyTup,value)+ " + "
    JxAssign=JxAssign[:-3] + ";\n"

afterword= "\treturn 0;\n}"



with open(outputFile,"w") as f:
    f.write(intro)
    f.write(JpAssign)
    f.write(JiAssign)
    f.write(JxAssign)
    f.write(afterword)