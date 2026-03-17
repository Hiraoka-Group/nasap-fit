import warnings
warnings.simplefilter("default")  # 未知キーwarningを見たい場合

from nasap_fit import NASAP_fit

engine = NASAP_fit.from_yaml("../data/config.yaml")  # YAML検証→Config生成→C++初期化

# DE（terminationConditionはdict、未知キーはwarning、不正値はValueError）
pop = engine.run_de(
    pop_size=128,
    terminationCondition={"maxIter": 50},
    seed=1,
)

best = min(pop, key=lambda r: r.error)

# LM
refined = engine.run_lm(
    best.constants,
    terminationCondition={"maxIter": 30, "xtol": 1e-10, "timeLimit": 60.0},
)

print("best error:", refined.error)
print("best constants:", refined.constants)