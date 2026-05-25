# -*- coding: utf-8 -*-
"""
B题：冬小麦产量预测与智慧种植布局优化
统一求解脚本（问题1 + 问题2 + 问题3）
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from itertools import combinations

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# 0. 读取数据
# ============================================================
XLS = "附件1.xlsx"
df_vi = pd.read_excel(XLS, sheet_name="长势时序数据")
df_wx = pd.read_excel(XLS, sheet_name="气象数据")
df_y  = pd.read_excel(XLS, sheet_name="产量数据")

df_vi.columns = ["year", "block", "step", "doy", "vi"]
df_wx.columns = ["year", "block", "temp", "prec", "sun"]
df_y.columns  = ["year", "block", "yield_"]

BLOCKS = [f"A{i}" for i in range(1, 11)]
YEARS_HIST = list(range(2011, 2026))
YEARS_PLAN = list(range(2026, 2031))

# 附件 2 表格：管理参数
MGMT = pd.DataFrame([
    ("A1",  "中等", 1.00,    0),
    ("A2",  "中等", 1.00,    0),
    ("A3",  "优良", 1.10,  150),
    ("A4",  "一般", 1.00, -100),
    ("A5",  "中等", 1.00,    0),
    ("A6",  "优良", 1.12,  200),
    ("A7",  "一般", 1.00, -120),
    ("A8",  "中等", 1.00,    0),
    ("A9",  "优良", 1.15,  180),
    ("A10", "一般", 1.00,   50),
], columns=["block", "level", "alpha", "extra_cost"])
ALPHA = dict(zip(MGMT.block, MGMT.alpha))
EXTRA = dict(zip(MGMT.block, MGMT.extra_cost))

PRICE = 2.8     # 元/公斤
COST0 = 800.0   # 基础成本/亩

# ============================================================
# 1. 问题1：构造特征 + 多模型预测
# ============================================================
# ---- 1.1 长势时序特征工程 ----
def vi_features(g):
    g = g.sort_values("doy")
    vi = g["vi"].values
    doy = g["doy"].values
    # 数值积分（梯形法 -> AUC）
    auc = np.trapz(vi, doy)
    feat = {
        "vi_max":     vi.max(),
        "vi_mean":    vi.mean(),
        "vi_std":     vi.std(),
        "vi_auc":     auc,
        "vi_peakdoy": doy[np.argmax(vi)],
        "vi_early":   vi[:4].mean(),    # 返青期 60-90
        "vi_mid":     vi[4:8].mean(),   # 拔节-灌浆 100-130
        "vi_late":    vi[8:].mean(),    # 成熟期 140-170
        "vi_grow":    vi.max() - vi[0], # 生长幅度
        "vi_decay":   vi.max() - vi[-1],# 衰减幅度
    }
    return pd.Series(feat)

vi_feat = df_vi.groupby(["year", "block"]).apply(vi_features).reset_index()

# ---- 1.2 合并气象+产量+管理 ----
data = vi_feat.merge(df_wx, on=["year", "block"]).merge(df_y, on=["year", "block"])
data["alpha"] = data["block"].map(ALPHA)
data["extra"] = data["block"].map(EXTRA)
# 将 8 月以上的截尾视作真实样本（亩产 800 是上限，可视作设备限）
data["censored"] = (data["yield_"] >= 800).astype(int)

# 特征：长势 + 气象 + 管理
FEATS = ["vi_max","vi_mean","vi_std","vi_auc","vi_peakdoy",
         "vi_early","vi_mid","vi_late","vi_grow","vi_decay",
         "temp","prec","sun","alpha","extra"]

X = data[FEATS].values
y = data["yield_"].values

# ---- 1.3 训练/验证 ----
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

models = {
    "Ridge":  Ridge(alpha=1.0),
    "RF":     RandomForestRegressor(n_estimators=400, max_depth=8,
                                    random_state=42, n_jobs=-1),
    "GBR":    GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                        learning_rate=0.05, random_state=42),
}

print("=" * 65)
print("问题1：冬小麦产量预测模型 —— 交叉验证结果")
print("=" * 65)
print(f"{'模型':<10}{'R²(CV)':>10}{'MAE(CV)':>12}{'RMSE(CV)':>12}")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_summary = {}
for name, m in models.items():
    r2s, maes, rmses = [], [], []
    for tr, te in kf.split(Xs):
        m.fit(Xs[tr], y[tr])
        p = m.predict(Xs[te])
        r2s.append(r2_score(y[te], p))
        maes.append(mean_absolute_error(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
    cv_summary[name] = (np.mean(r2s), np.mean(maes), np.mean(rmses))
    print(f"{name:<10}{np.mean(r2s):>10.4f}{np.mean(maes):>12.2f}{np.mean(rmses):>12.2f}")

# ---- 1.4 选择最优模型并全样本训练 ----
best = max(cv_summary, key=lambda k: cv_summary[k][0])
print(f"\n最优模型：{best}（按 R² 选取）")
final_model = models[best]
final_model.fit(Xs, y)

# 训练集表现
yhat = final_model.predict(Xs)
print(f"训练集 R²={r2_score(y,yhat):.4f}  MAE={mean_absolute_error(y,yhat):.2f}"
      f"  RMSE={np.sqrt(mean_squared_error(y,yhat)):.2f}")

# 特征重要性
if hasattr(final_model, "feature_importances_"):
    imp = pd.Series(final_model.feature_importances_, index=FEATS)\
            .sort_values(ascending=False)
    print("\n特征重要性 Top 8:")
    for k, v in imp.head(8).items():
        print(f"  {k:<12}{v:.4f}")

# ============================================================
# 2. 2026-2030 各区块基线亩产预测
# ============================================================
# 思路：
#  长势/气象未来值用 2021-2025 近 5 年均值（趋势平稳假设）
#  alpha=1, extra=0 视作"常规管理"基线
recent = data[data.year >= 2021].groupby("block")[FEATS[:-2]].mean()
def predict_baseline(block):
    row = recent.loc[block].to_dict()
    row["alpha"] = 1.0
    row["extra"] = 0.0
    x = np.array([[row[f] for f in FEATS]])
    return float(final_model.predict(scaler.transform(x))[0])

baseline_yield = {b: predict_baseline(b) for b in BLOCKS}
print("\n2026-2030 基线预测亩产（kg/亩，常规管理）：")
for b in BLOCKS:
    print(f"  {b:<4}{baseline_yield[b]:7.1f}")

# ============================================================
# 3. 问题2：种植布局整数规划
# ============================================================
import pulp

# 历史平均总产量（区块产量按年汇总后再求 15 年平均）
hist_avg_total = df_y.groupby("year")["yield_"].sum().mean()
print(f"\n历史 15 年平均总产量 = {hist_avg_total:.1f} kg")
print(f"产能稳定性下限（5 年总产 >= 65% × 5 × 历史均值）= "
      f"{0.65*5*hist_avg_total:.1f} kg")

# 决策：x[i,t] = 1 表示第 t 年种植第 i 区块（采用最优管理 -> alpha 上限）
profit = {}     # 元/亩
yield_block = {}# kg/亩（含 alpha 提升）
for b in BLOCKS:
    y_b = baseline_yield[b] * ALPHA[b]
    yield_block[b] = y_b
    profit[b] = y_b * PRICE - (COST0 + EXTRA[b])

print("\n各区块在最优管理下的预测亩产、单亩成本、单亩效益：")
print(f"{'区块':<5}{'亩产':>9}{'成本':>9}{'效益':>10}")
for b in BLOCKS:
    print(f"{b:<5}{yield_block[b]:>9.1f}{COST0+EXTRA[b]:>9.1f}{profit[b]:>10.1f}")

def solve_plan(profit_d, yield_d, label="问题2"):
    m = pulp.LpProblem(label, pulp.LpMaximize)
    x = {(b,t): pulp.LpVariable(f"x_{b}_{t}", cat="Binary")
         for b in BLOCKS for t in YEARS_PLAN}
    # 目标：5 年总效益最大化
    m += pulp.lpSum(profit_d[b]*x[(b,t)] for b in BLOCKS for t in YEARS_PLAN)
    # 每年 ≤ 6 个区块
    for t in YEARS_PLAN:
        m += pulp.lpSum(x[(b,t)] for b in BLOCKS) <= 6
    # 每区块 5 年内至少休耕 1 年
    for b in BLOCKS:
        m += pulp.lpSum(x[(b,t)] for t in YEARS_PLAN) <= 4
    # 产能稳定性约束
    m += pulp.lpSum(yield_d[b]*x[(b,t)] for b in BLOCKS for t in YEARS_PLAN)\
         >= 0.65 * 5 * hist_avg_total
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    sol = {(b,t): int(round(x[(b,t)].value())) for b in BLOCKS for t in YEARS_PLAN}
    return sol, pulp.value(m.objective)

sol2, obj2 = solve_plan(profit, yield_block, "Q2")
print("\n" + "=" * 65)
print(f"问题2 最优方案：5 年总效益 = {obj2:,.1f} 元")
print("=" * 65)
plan_df = pd.DataFrame(0, index=BLOCKS, columns=YEARS_PLAN)
for (b,t),v in sol2.items():
    plan_df.loc[b, t] = v
plan_df["种植年数"] = plan_df.sum(axis=1)
print(plan_df)

# 验证产能
total_y = sum(yield_block[b]*sol2[(b,t)] for b in BLOCKS for t in YEARS_PLAN)
print(f"5 年合计产量 = {total_y:,.1f} kg "
      f"({total_y/(5*hist_avg_total)*100:.1f}% × 历史均值)")

# ============================================================
# 4. 问题3：风险量化 + 鲁棒决策模型
# ============================================================
print("\n" + "=" * 65)
print("问题3：考虑气象灾害不确定性的智慧决策模型")
print("=" * 65)

# ---- 4.1 风险情景设计（依据附件2的灾害参数） ----
# 情景 s = (灾害等级, 价格波动)
# 等级：0 正常；1 轻度；2 中度；3 重度
# 概率：参考附件 2 给出的范围中点
# 影响：减产比 / 价格波动比（价格灾害年价格反而上涨，缓和效益损失）
SCEN = [
    # name,        prob,  yield_loss, price_factor
    ("正常",       0.80,  0.00,       1.00),
    ("轻度灾害",   0.10,  0.10,       1.05),
    ("中度灾害",   0.07,  0.20,       1.10),
    ("重度灾害",   0.03,  0.30,       1.20),
]
sc_names = [s[0] for s in SCEN]
probs   = np.array([s[1] for s in SCEN])
losses  = np.array([s[2] for s in SCEN])
pfacs   = np.array([s[3] for s in SCEN])
print("\n情景参数表：")
for s in SCEN:
    print(f"  {s[0]:<8} 概率={s[1]:.2f}  减产={s[2]*100:.0f}%  价格系数={s[3]}")

# ---- 4.2 各情景下区块单亩效益 ----
def profit_scen(b, k):
    y_real = baseline_yield[b] * ALPHA[b] * (1 - losses[k])
    p_real = PRICE * pfacs[k]
    return y_real * p_real - (COST0 + EXTRA[b]), y_real

profit_s = {(b,k): profit_scen(b,k)[0] for b in BLOCKS for k in range(len(SCEN))}
yield_s  = {(b,k): profit_scen(b,k)[1] for b in BLOCKS for k in range(len(SCEN))}

# ---- 4.3 综合决策：期望效益 - λ × 标准差 (Mean-Variance / Markowitz 风险厌恶) ----
# 同时引入 CVaR(α=0.9) 作为附加约束（重度灾害下损益下界）
def solve_robust(lam=0.5, cvar_alpha=0.9):
    m = pulp.LpProblem("Q3", pulp.LpMaximize)
    x = {(b,t): pulp.LpVariable(f"x_{b}_{t}", cat="Binary")
         for b in BLOCKS for t in YEARS_PLAN}

    # 各情景下 5 年总效益的线性表达式
    Z = {k: pulp.lpSum(profit_s[(b,k)]*x[(b,t)]
                       for b in BLOCKS for t in YEARS_PLAN)
         for k in range(len(SCEN))}
    EZ = pulp.lpSum(probs[k]*Z[k] for k in range(len(SCEN)))

    # 用情景间最大-最小差作为离散度近似（线性化标准差）
    # 引入 dev_k = max(0, EZ - Z_k) 作为下行风险
    dev = {k: pulp.LpVariable(f"dev_{k}", lowBound=0)
           for k in range(len(SCEN))}
    for k in range(len(SCEN)):
        m += dev[k] >= EZ - Z[k]
    risk_term = pulp.lpSum(probs[k]*dev[k] for k in range(len(SCEN)))

    # CVaR(α): 仅在重度灾害（最差情景）保证 5 年总效益 >= 阈值
    worst = len(SCEN) - 1
    # 设阈值为问题2解的 60%
    cvar_threshold = 0.6 * obj2
    m += Z[worst] >= cvar_threshold

    # 目标：期望 - λ * 下行风险
    m += EZ - lam * risk_term

    # 通用约束（同问题2）
    for t in YEARS_PLAN:
        m += pulp.lpSum(x[(b,t)] for b in BLOCKS) <= 6
    for b in BLOCKS:
        m += pulp.lpSum(x[(b,t)] for t in YEARS_PLAN) <= 4
    # 期望产能稳定性
    EY = pulp.lpSum(probs[k]*yield_s[(b,k)]*x[(b,t)]
                    for b in BLOCKS for t in YEARS_PLAN
                    for k in range(len(SCEN)))
    m += EY >= 0.65 * 5 * hist_avg_total

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    sol = {(b,t): int(round(x[(b,t)].value())) for b in BLOCKS for t in YEARS_PLAN}
    EZ_v  = sum(probs[k]*sum(profit_s[(b,k)]*sol[(b,t)]
                for b in BLOCKS for t in YEARS_PLAN)
                for k in range(len(SCEN)))
    Zk_v  = [sum(profit_s[(b,k)]*sol[(b,t)]
                 for b in BLOCKS for t in YEARS_PLAN)
             for k in range(len(SCEN))]
    return sol, EZ_v, Zk_v

sol3, EZ3, Zk3 = solve_robust(lam=0.5)
plan3 = pd.DataFrame(0, index=BLOCKS, columns=YEARS_PLAN)
for (b,t),v in sol3.items():
    plan3.loc[b,t] = v
plan3["种植年数"] = plan3.sum(axis=1)
print("\n问题3 最优鲁棒种植方案 (λ=0.5)：")
print(plan3)
print(f"期望 5 年总效益 E[Z] = {EZ3:,.1f} 元")
print("各情景下 5 年总效益：")
for k,(n,_,_,_) in enumerate(SCEN):
    print(f"  {n:<8}: {Zk3[k]:>14,.1f} 元")

# ---- 4.4 灵敏度分析：风险厌恶系数 λ ----
print("\n灵敏度分析：风险厌恶系数 λ 对方案的影响")
print(f"{'λ':>6}{'E[Z]':>15}{'最差情景效益':>18}{'差值':>14}")
sens_rows = []
for lam in [0.0, 0.25, 0.5, 1.0, 2.0]:
    s, ez, zk = solve_robust(lam=lam)
    sens_rows.append((lam, ez, zk[-1], ez - zk[-1]))
    print(f"{lam:>6.2f}{ez:>15,.1f}{zk[-1]:>18,.1f}{ez-zk[-1]:>14,.1f}")

# ============================================================
# 5. 写出结果文件
# ============================================================
out = "结果汇总.xlsx"
with pd.ExcelWriter(out) as w:
    pd.DataFrame({"模型":list(cv_summary),
                  "R2":[v[0] for v in cv_summary.values()],
                  "MAE":[v[1] for v in cv_summary.values()],
                  "RMSE":[v[2] for v in cv_summary.values()]
                 }).to_excel(w, sheet_name="问题1_模型对比", index=False)
    pd.DataFrame({"区块":BLOCKS,
                  "基线预测亩产":[baseline_yield[b] for b in BLOCKS],
                  "alpha":[ALPHA[b] for b in BLOCKS],
                  "管理后亩产":[yield_block[b] for b in BLOCKS],
                  "单亩效益":[profit[b] for b in BLOCKS]
                 }).to_excel(w, sheet_name="问题1_预测", index=False)
    plan_df.to_excel(w, sheet_name="问题2_方案")
    plan3.to_excel(w, sheet_name="问题3_方案")
    pd.DataFrame(sens_rows, columns=["lambda","E[Z]","最差情景","差值"])\
      .to_excel(w, sheet_name="问题3_灵敏度", index=False)
print(f"\n结果已写入 {out}")
