import warnings
warnings.simplefilter("default")  # 未知キーwarningを見たい場合

from nasap_fit import NASAP_fit

engine = NASAP_fit.from_yaml("data/config.yaml")  # YAML検証→Config生成→C++初期化

# DE（terminationConditionはdict、未知キーはwarning、不正値はValueError）
pop = engine.run_de(
    pop_size=128,
    terminationCondition={"maxIter": 50,
                           "timeLimit": 60.0,
                           "xtol": 1e-6, #runDEでは無効
                           "ftolAbs": 1e-5,
                           "ftolRel": 0.02,
                           "stall": 20,},  
    seed=1,
)

best = min(pop, key=lambda r: r.error)

# LM
refined = engine.run_lm(
    best.constants,
    terminationCondition={"maxIter": 50,
                           "timeLimit": 60.0,
                           "xtol": 1e-6, 
                           "timeLimit": 60.0,
                           "ftolAbs": 1e-5,
                           "ftolRel": 0.02,
                           "stall": 10},
)


print("best error:", best.error)
print("best constants:", best.constants)

simulationResult = engine.simulate(
    t=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0],
    constant=best.constants,
    reaction_ids=[22, 67],)
for i, id in enumerate(simulationResult.reactionProgress.reaction_ids):
    print(f"Reaction ID: {id}, label: {simulationResult.reactionProgress.reaction_labels[i]}")
for i, time_point in enumerate(simulationResult.t):
    print(f"Time: {time_point:.1f} min, J[{simulationResult.reactionProgress.reaction_ids[0]}]: {simulationResult.reactionProgress.J[i][0]}, J[{simulationResult.reactionProgress.reaction_ids[1]}]: {simulationResult.reactionProgress.J[i][1]}")