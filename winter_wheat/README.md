# 冬小麦产量预测与智慧种植布局优化（B 题）

一套**可直接运行**的端到端代码，覆盖三个问题：
- **Q1**：长势时序 + 气象的多源融合产量预测（双 Logistic 物候 + Stacking）
- **Q2**：2026–2030 种植布局确定性 MIP 优化（PuLP + CBC）
- **Q3**：气象灾害风险下的多目标（期望效益 − λ·CVaR）智慧决策

## 目录结构

```
winter_wheat/
├── data/                     # 题目附件（已放好）
│   ├── 附件1.xlsx
│   ├── 附件2.docx
│   └── 题目.docx
├── src/
│   ├── q1_yield_model.py     # 问题1：产量预测
│   ├── q2_layout_mip.py      # 问题2：MIP布局
│   ├── q3_risk_decision.py   # 问题3：风险决策（蒙特卡洛+CVaR）
│   └── run_all.py            # 一键跑全流程
├── outputs/                  # 所有 csv/json 输出
├── figures/                  # 所有图
└── README.md
```

## 环境

- Python ≥ 3.9
- 依赖：`pip install pulp xgboost scikit-learn scipy numpy pandas openpyxl matplotlib`

## 一键运行

```bash
cd winter_wheat
python src/run_all.py
```

运行约 2–5 分钟（受 CBC 求解器 & 蒙特卡洛 N 影响）。

## 各脚本单独运行

```bash
python src/q1_yield_model.py     # 训练并产出 yield_pred_2026_2030.csv
python src/q2_layout_mip.py      # 读取 Q1 预测 → 输出 q2_plan.csv
python src/q3_risk_decision.py   # 蒙特卡洛 + CVaR → 输出 q3_plan.csv 与 pareto.csv
```

## 输出说明

| 文件 | 含义 |
|---|---|
| `outputs/q1_metrics.json` | LOYO-CV 与 hold-out 指标 (R²/RMSE/MAPE/NSE) |
| `outputs/q1_feature_importance.csv` | XGBoost 特征重要性 |
| `outputs/yield_pred_2026_2030.csv` | 10 区块 × 5 年基准预测亩产 |
| `outputs/q2_plan.csv` | Q2 种植方案 (区块×年) 0/1 矩阵 + 强度 |
| `outputs/q2_summary.json` | Q2 总效益、年均产量、约束校验 |
| `outputs/q3_plan.csv` | Q3 风险方案 |
| `outputs/q3_pareto.csv` | λ 扫描的 Pareto 前沿 |
| `figures/*.png` | EDA、预测对比、Gantt、Pareto 等图 |

## 模型要点

### Q1 特征
- 12 期长势的 AUC、峰值、到峰时间、上升/下降斜率、半峰宽
- 双 Logistic 物候拟合 6 参数（鲁棒返青/成熟刻画）
- 气象：均温、累计降水、日照时数 + 与长势的交互项
- 区块 one-hot 吸收地力固定效应

### Q1 模型
- 底层：Ridge / RandomForest / XGBoost
- 顶层：Ridge Stacking
- 验证：Leave-One-Year-Out CV + 2024–2025 hold-out

### Q2 MIP（PuLP-CBC）
- 决策：`x[i,t]` 是否种、`u[i,t]` 水肥强度（1.0~k_i）
- 目标：max 5 年总效益
- 约束：①每年≤6 块；②每块 5 年至少休 1 年；③5 年平均产量 ≥ 65% 历史均值

### Q3 风险决策
- 蒙特卡洛 N=500 条 5 年灾害情景（干旱/涝/高温/低温/干热风）
- 气候漂移：高温灾害概率随年份线性增大
- 价格随机：U(-10%, +20%)
- 目标：max E[Profit] − λ·CVaR₀.₉₅(Loss)
- λ 扫描出 Pareto 前沿，TOPSIS 选折中解
