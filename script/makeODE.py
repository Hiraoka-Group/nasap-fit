import numpy as np
import pandas as pd
import csv
import pprint


inputFile="../data/classified_reactions_str.csv"
outputFile="../include/ODE.hpp"
df = pd.read_csv(inputFile, index_col=0)

records=df.to_dict(orient="records")
reaction_kinds=df["kind"].unique() #反応速度定数の種類
reaction_kinds.sort()
typeToIndex={} #反応タイプから速度定数のindexへの変換
for i in range(len(reaction_kinds)):
    typeToIndex[reaction_kinds[i]]=i

species=0 #化学種の種類　これをもとに化学種の量のarrayを作る

#常微分方程式の項を表現するクラス
class term:
    reactant1=""
    reactant2=None
    duplicate=0
    kind=None
    def __init__(self):
        pass
    def __init__(self,arg1,arg2,arg3,arg4):
        self.reactant1=arg1
        self.reactant2=arg2
        self.duplicate=arg3
        self.kind=arg4
    def minus(self):
        return term(self.reactant1,self.reactant2,-self.duplicate,self.kind)
    def to_fomula(self):
        ret=""
        if self.reactant2 is None:
            return f"({self.duplicate} * k[{typeToIndex[self.kind]}] * sp[{self.reactant1}])"
        else:
            return f"({self.duplicate} * k[{typeToIndex[self.kind]}] * sp[{self.reactant1}] * sp[{self.reactant2}])"
#k[1] * sp[2] - k[0] * sp[0] * sp[1];


    


for dict in records:
    species=max(species,dict["init_assem_id"])
    species=max(species,dict["entering_assem_id"])
    species=max(species,dict["product_assem_id"])
    species=max(species,dict["leaving_assem_id"])
species=round(species)
species+=1
ODE=[[] for _ in range(species)] #常微分方程式の素


for dict in records:
    init=dict["init_assem_id"]
    entering=None if pd.isna(dict["entering_assem_id"]) else round(dict["entering_assem_id"])
    product=dict["product_assem_id"]
    leaving=None if pd.isna(dict["leaving_assem_id"]) else round(dict["leaving_assem_id"])

    hoge=term(init,entering,dict["duplicate_count"],dict["kind"])
    
    ODE[init].append(hoge.minus())
    if entering is not None:
        ODE[entering].append(hoge.minus())
    ODE[product].append(hoge)
    if leaving is not None:
        ODE[leaving].append(hoge)



with open(outputFile,"w") as f:
    f.write(
        "#pragma once\n" \
        "#include <array>\n" \
        "#include \"constants.hpp\"\n" \
        "#include \"speciesAmount.hpp\"\n\n" \
        "inline speciesAmount diffCoeff(const std::array<double, constantSize>& k, const speciesAmount& sp) {\n" \
	    "\tspeciesAmount dxdt;\n\t")
    for t in ODE:
        isFirst=True
        f.write("dxdt[" + str(ODE.index(t)) + "] = ")
        for t2 in t:
            if isFirst:
                isFirst=False
            else:
                f.write(" + ")
            f.write(t2.to_fomula())
        f.write(";\n\t")
    
    f.write("return dxdt;\n}")

