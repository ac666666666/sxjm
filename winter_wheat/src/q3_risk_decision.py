"""
问题3：考虑气象灾害不确定性的智慧种植决策模型 (优化版)
改进点：
- d ≤ x 约束 (彻底杜绝休耕年份强度污染)
- Q1 残差注入：Yhat 加正态噪声 σ = Y_std (端到端不确定性)
- 多灾种 + 气候漂移 + 价格-灾害耦合
- SAA: N_train 训练 + N_eval 独立评估
- 灾害概率敏感性分析 (scale ∈ {0.5, 1.0, 1.5, 2.0})
- SAA 情景数收敛曲线
- TOPSIS 折中解 + Pareto 前沿 (含 3D 投影)
- 中文图
"""
import os, json, sys
import numpy as np
import pandas as pd
import pulp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for _f in ['Songti SC', 'Heiti SC', 'Arial Unicode MS', 'PingFang SC',
           'Microsoft YaHei', 'SimHei', 'STHeiti']:
    try:
        from matplotlib import font_manager
        if any(_f in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [_f]
            break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'outputs')
FIGDIR = os.path.join(ROOT, 'figures')
os.makedirs(OUTDIR, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)

BLOCKS = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']
YEARS  = [2026, 2027, 2028, 2029, 2030]
K_MAX = {'A1':1.00,'A2':1.00,'A3':1.10,'A4':1.00,'A5':1.00,
         'A6':1.12,'A7':1.00,'A8':1.00,'A9':1.15,'A10':1.00}
C_ADD = {'A1':0,'A2':0,'A3':150,'A4':-100,'A5':0,
         'A6':200,'A7':-120,'A8':0,'A9':180,'A10':50}
PRICE0 = 2.8
BASE_COST = 800
GAMMA = 300


def sample_scenarios(N=300, seed=42, hazard_scale=1.0):
    """生成 N 条 5 年 × 区块 × 灾害衰减系数 lambda + 价格 price.
    hazard_scale: 灾害概率整体放缩 (灵敏性分析用)
    """
    rng = np.random.default_rng(seed)
    nB, nT = len(BLOCKS), len(YEARS)
    Lam = np.ones((N, nB, nT))
    Price = np.full((N, nT), PRICE0)
    cat = {
        'drought':  (0.10, 0.05, 0.02, (0.05, 0.30)),
        'flood':    (0.08, 0.04, 0.015, (0.05, 0.25)),
        'heat':     (0.12, 0.05, 0.02, (0.05, 0.28)),
        'cold':     (0.08, 0.04, 0.015, (0.05, 0.20)),
        'hotdry':   (0.10, 0.04, 0.015, (0.05, 0.25)),
    }
    for w in range(N):
        for ti in range(nT):
            year_drift = 0.015 * ti
            for c, (pl, pm, ps, rng_imp) in cat.items():
                pl_e = (pl + (year_drift if c in ('heat', 'hotdry') else 0)) * hazard_scale
                pm_e = (pm + (year_drift * 0.6 if c in ('heat', 'hotdry') else 0)) * hazard_scale
                ps_e = ps * hazard_scale
                r = rng.random()
                if r < ps_e:
                    severity = rng.uniform(0.20, rng_imp[1])
                    affected = rng.random(nB) < 0.9
                elif r < ps_e + pm_e:
                    severity = rng.uniform(0.10, 0.20)
                    affected = rng.random(nB) < 0.6
                elif r < ps_e + pm_e + pl_e:
                    severity = rng.uniform(rng_imp[0], 0.10)
                    affected = rng.random(nB) < 0.4
                else:
                    continue
                for bi in range(nB):
                    if affected[bi]:
                        Lam[w, bi, ti] *= (1 - severity)
            avg_loss = 1 - Lam[w, :, ti].mean()
            if avg_loss > 0.05:
                Price[w, ti] = PRICE0 * rng.uniform(1.0, 1.20)
            elif avg_loss > 0.02:
                Price[w, ti] = PRICE0 * rng.uniform(0.95, 1.10)
            else:
                Price[w, ti] = PRICE0 * rng.uniform(0.90, 1.05)
    return Lam, Price


def sample_yhat_with_noise(Yhat, Y_std, N, seed=123):
    """对 Yhat 注入 Q1 残差 std 形成 (N,B,T) 张量，建模产量预测不确定性。"""
    rng = np.random.default_rng(seed)
    nB, nT = len(BLOCKS), len(YEARS)
    base = np.zeros((nB, nT))
    sigma = np.zeros((nB, nT))
    for bi, b in enumerate(BLOCKS):
        for ti, t in enumerate(YEARS):
            base[bi, ti] = Yhat[(b, t)]
            sigma[bi, ti] = Y_std.get((b, t), 0.0)
    noise = rng.normal(0, 1, size=(N, nB, nT)) * sigma[None, :, :]
    Y = np.clip(base[None, :, :] + noise, 50.0, 800.0)  # 物理边界
    return Y


def hist_avg_total(data_xlsx):
    yd = pd.read_excel(data_xlsx, sheet_name='产量数据')
    yd.columns = ['年份','区块','Y']
    return float(yd.groupby('年份')['Y'].sum().mean())


def _build_and_solve_once(hist_avg, Lam, Price, Y_train,
                          lambda_cvar, alpha, time_limit, beta_stab):
    """单次建模 + 求解，返回 (status, plan, objective)。"""
    N, nB, nT = Lam.shape

    m = pulp.LpProblem('Q3_Risk', pulp.LpMaximize)
    x = {(i,t): pulp.LpVariable(f'x_{i}_{t}', cat='Binary') for i in BLOCKS for t in YEARS}
    delta_max = {i: K_MAX[i] - 1.0 for i in BLOCKS}
    d = {(i,t): pulp.LpVariable(f'd_{i}_{t}', lowBound=0, upBound=1) for i in BLOCKS for t in YEARS}
    w_v = {(i,t): pulp.LpVariable(f'w_{i}_{t}', lowBound=0, upBound=1) for i in BLOCKS for t in YEARS}
    for i in BLOCKS:
        for t in YEARS:
            m += d[(i,t)] <= x[(i,t)]                # ★ d=0 当休耕
            m += w_v[(i,t)] <= x[(i,t)]
            m += w_v[(i,t)] <= d[(i,t)]
            m += w_v[(i,t)] >= d[(i,t)] - (1 - x[(i,t)])

    cost_expr = pulp.lpSum((BASE_COST + C_ADD[i]) * x[(i,t)] + GAMMA * delta_max[i] * w_v[(i,t)]
                            for i in BLOCKS for t in YEARS)
    profit_w = []
    for w_idx in range(N):
        rev_w = pulp.lpSum(
            Price[w_idx, YEARS.index(t)] * Lam[w_idx, BLOCKS.index(i), YEARS.index(t)] *
            Y_train[w_idx, BLOCKS.index(i), YEARS.index(t)] *
            (x[(i,t)] + delta_max[i] * w_v[(i,t)])
            for i in BLOCKS for t in YEARS
        )
        profit_w.append(rev_w - cost_expr)

    E_profit = pulp.lpSum(profit_w) / N

    eta = pulp.LpVariable('eta')
    q = [pulp.LpVariable(f'q_{w}', lowBound=0) for w in range(N)]
    for w_idx in range(N):
        m += q[w_idx] >= (-profit_w[w_idx]) - eta
    CVaR = eta + pulp.lpSum(q) / ((1 - alpha) * N)

    m += E_profit - lambda_cvar * CVaR

    for t in YEARS:
        m += pulp.lpSum(x[(i,t)] for i in BLOCKS) <= 6
    for i in BLOCKS:
        m += pulp.lpSum(x[(i,t)] for t in YEARS) <= 4

    # 稳产 ① 期望硬约束 (与 Q2 同口径，不计灾害；用 Q1 残差扰动后的产量均值)
    Y_base = Y_train.mean(axis=0)  # (B,T)
    EY = pulp.lpSum(
        Y_base[BLOCKS.index(i), YEARS.index(t)] *
        (x[(i,t)] + delta_max[i] * w_v[(i,t)])
        for i in BLOCKS for t in YEARS
    ) / 5.0
    m += EY >= 0.65 * hist_avg

    # 稳产 ② 机会约束 (chance constraint, SAA + Big-M):
    #   对每个情景 w 引入 0/1 指示 z[w] = 1 表示该情景"达标"
    #   AnnualYield_w = (1/5) * Σ_{i,t} Lam[w,i,t] * Y_train[w,i,t] * (x[i,t] + δ_i*w_v[i,t])
    #   AnnualYield_w >= 0.65*hist_avg - M*(1 - z[w])
    #   Σ z[w] >= ⌈β·N⌉   →  P(达标) ≥ β
    threshold = 0.65 * hist_avg
    # M 取一个保守上界 (10 块 × 5 年 × 800kg × 1.15 / 5 = ~ 1840 远大于阈值)
    big_M = float(threshold + 5000.0)
    z = [pulp.LpVariable(f'z_{w_idx}', cat='Binary') for w_idx in range(N)]
    for w_idx in range(N):
        AY_w = pulp.lpSum(
            Lam[w_idx, BLOCKS.index(i), YEARS.index(t)] *
            Y_train[w_idx, BLOCKS.index(i), YEARS.index(t)] *
            (x[(i,t)] + delta_max[i] * w_v[(i,t)])
            for i in BLOCKS for t in YEARS
        ) / 5.0
        m += AY_w >= threshold - big_M * (1 - z[w_idx])
    # 至少 ⌈β·N⌉ 个情景达标
    import math as _math
    m += pulp.lpSum(z) >= _math.ceil(beta_stab * N)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    m.solve(solver)
    status = pulp.LpStatus[m.status]

    plan_rows = []
    for t in YEARS:
        for i in BLOCKS:
            xv_raw = x[(i,t)].value()
            xv = int(round(xv_raw)) if xv_raw is not None else 0
            xv = 1 if xv >= 1 else 0
            dv_raw = d[(i,t)].value()
            dv = float(dv_raw) if (dv_raw is not None and xv) else 0.0
            dv = max(0.0, min(1.0, dv))
            u_eff = (1.0 + delta_max[i] * dv) if xv else 0.0
            plan_rows.append(dict(年份=t, 区块=i, plant=xv,
                                  intensity=round(u_eff, 4)))
    plan = pd.DataFrame(plan_rows)
    return dict(plan=plan, status=status,
                objective=float(pulp.value(m.objective) or 0.0))


def solve_risk_model(y_pred_csv, hist_avg, Lam, Price, Y_train,
                     lambda_cvar=0.3, alpha=0.95, time_limit=180, log=True,
                     beta_stab=0.50):
    """Y_train: (N,B,T) 含 Q1 残差扰动的产量张量
    稳产策略 = 双层：
      ① 期望硬约束 E[YA] ≥ 0.65·hist (与 Q2 同口径，保证均值层面达标)
      ② 机会约束  P(YA ≥ 0.65·hist) ≥ beta_stab (Big-M, 控制下尾)
    自动 Fallback: 若不可行，按 [beta_stab, 0.6·beta_stab, 0.3·beta_stab, 0.0]
    依次降低稳产概率重试，直到求解成功为止。
    """
    fallback_betas = [beta_stab,
                      round(beta_stab * 0.6, 3),
                      round(beta_stab * 0.3, 3),
                      0.0]
    # 去重保留顺序
    seen = set(); ordered = []
    for b in fallback_betas:
        if b not in seen:
            seen.add(b); ordered.append(b)

    last = None
    for b in ordered:
        res = _build_and_solve_once(hist_avg, Lam, Price, Y_train,
                                    lambda_cvar=lambda_cvar, alpha=alpha,
                                    time_limit=time_limit, beta_stab=b)
        if log:
            print(f'  [solve] λ={lambda_cvar} β={b} status={res["status"]}')
        last = res
        last['beta_used'] = b
        if res['status'] == 'Optimal':
            return last
    return last


def evaluate_plan(plan, Yhat, Y_std, Lam, Price, hist_avg, seed_eval=2026):
    """对 plan 在 (Lam,Price) 上做含产量噪声的评估"""
    N, nB, nT = Lam.shape
    Y_eval = sample_yhat_with_noise(Yhat, Y_std, N, seed=seed_eval)

    x_val = {(r['区块'], int(r['年份'])): int(r['plant']) for _, r in plan.iterrows()}
    intensity = {(r['区块'], int(r['年份'])): float(r['intensity']) for _, r in plan.iterrows()}
    delta_max = {i: K_MAX[i] - 1.0 for i in BLOCKS}

    base_cost = 0.0
    for i in BLOCKS:
        for t in YEARS:
            if x_val[(i,t)]:
                base_cost += (BASE_COST + C_ADD[i]) + GAMMA * (intensity[(i,t)] - 1.0)

    sc_profit = np.zeros(N); sc_yield_avg = np.zeros(N)
    for w in range(N):
        rev = 0.0; ysum = 0.0
        for i in BLOCKS:
            for t in YEARS:
                if not x_val[(i,t)]:
                    continue
                bi, ti = BLOCKS.index(i), YEARS.index(t)
                y_eff = Lam[w, bi, ti] * Y_eval[w, bi, ti] * intensity[(i,t)]
                rev += Price[w, ti] * y_eff
                ysum += y_eff
        sc_profit[w] = rev - base_cost
        sc_yield_avg[w] = ysum / 5.0
    k = max(1, int(N * 0.05))
    return dict(
        E_profit=float(sc_profit.mean()),
        Std_profit=float(sc_profit.std()),
        CVaR95_loss=float(np.mean(np.sort(-sc_profit)[-k:])),
        VaR95_loss=float(np.quantile(-sc_profit, 0.95)),
        prob_meet_threshold=float((sc_yield_avg >= 0.65 * hist_avg).mean()),
        min_profit=float(sc_profit.min()),
        max_profit=float(sc_profit.max()),
        sc_profit=sc_profit, sc_yield_avg=sc_yield_avg,
    )


def topsis_select(pareto_df):
    df = pareto_df.copy()
    benefit = ['E_profit', 'prob_meet_threshold']
    cost = ['CVaR95_loss', 'Std_profit']
    norm = pd.DataFrame(index=df.index)
    for c in benefit + cost:
        v = df[c].astype(float).values
        denom = np.sqrt((v ** 2).sum()) or 1.0
        norm[c] = v / denom
    weight = 1.0 / (len(benefit) + len(cost))
    weighted = norm * weight
    ip = pd.Series({**{c: weighted[c].max() for c in benefit},
                    **{c: weighted[c].min() for c in cost}})
    inv = pd.Series({**{c: weighted[c].min() for c in benefit},
                     **{c: weighted[c].max() for c in cost}})
    dp = np.sqrt(((weighted - ip) ** 2).sum(axis=1))
    dn = np.sqrt(((weighted - inv) ** 2).sum(axis=1))
    df['TOPSIS_score'] = dn / (dp + dn + 1e-12)
    return df.sort_values('TOPSIS_score', ascending=False).reset_index(drop=True)


def saa_convergence(y_pred_csv, hist_avg, Yhat, Y_std,
                    Ns=(40, 80, 160, 320), lambda_cvar=0.3, time_limit=180):
    """SAA 情景数收敛曲线：固定 eval 集 N=2000, 不同 N_train 看目标值与 E_profit"""
    Lam_ev, Price_ev = sample_scenarios(N=2000, seed=999)
    rows = []
    for N in Ns:
        Lam_tr, Price_tr = sample_scenarios(N=N, seed=42)
        Y_tr = sample_yhat_with_noise(Yhat, Y_std, N, seed=100 + N)
        res = solve_risk_model(y_pred_csv, hist_avg, Lam_tr, Price_tr, Y_tr,
                               lambda_cvar=lambda_cvar, time_limit=time_limit, log=False)
        ev = evaluate_plan(res['plan'], Yhat, Y_std, Lam_ev, Price_ev, hist_avg, seed_eval=999)
        rows.append(dict(N=N, train_obj=res['objective'],
                         eval_E=ev['E_profit'], eval_CVaR=ev['CVaR95_loss']))
        print(f"  [SAA-conv] N={N}  train_obj={res['objective']:.0f}  eval_E={ev['E_profit']:.0f}  CVaR={ev['CVaR95_loss']:.0f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, 'q3_saa_convergence.csv'), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df['N'], df['eval_E'], marker='o'); ax[0].set_title('期望利润随 N_train 收敛')
    ax[0].set_xlabel('训练情景数 N'); ax[0].set_ylabel('eval E[Profit]')
    ax[0].set_xscale('log')
    ax[1].plot(df['N'], df['eval_CVaR'], marker='o', color='red'); ax[1].set_title('CVaR_0.95 随 N_train 收敛')
    ax[1].set_xlabel('训练情景数 N'); ax[1].set_ylabel('eval CVaR_0.95')
    ax[1].set_xscale('log')
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'q3_saa_convergence.png'), dpi=130); plt.close()
    return df


def hazard_sensitivity(y_pred_csv, hist_avg, Yhat, Y_std,
                       scales=(0.5, 1.0, 1.5, 2.0), lambda_cvar=0.3,
                       N_train=80, N_eval=1000, time_limit=180):
    """灾害概率灵敏性：方案如何变化"""
    rows = []
    for s in scales:
        Lam_tr, Price_tr = sample_scenarios(N=N_train, seed=42, hazard_scale=s)
        Y_tr = sample_yhat_with_noise(Yhat, Y_std, N_train, seed=200)
        res = solve_risk_model(y_pred_csv, hist_avg, Lam_tr, Price_tr, Y_tr,
                               lambda_cvar=lambda_cvar, time_limit=time_limit, log=False)
        Lam_ev, Price_ev = sample_scenarios(N=N_eval, seed=2026, hazard_scale=s)
        ev = evaluate_plan(res['plan'], Yhat, Y_std, Lam_ev, Price_ev, hist_avg, seed_eval=999)
        n_planted = int(res['plan']['plant'].sum())
        rows.append(dict(hazard_scale=s, n_planted=n_planted,
                         E_profit=ev['E_profit'], CVaR95=ev['CVaR95_loss'],
                         prob_meet=ev['prob_meet_threshold']))
        print(f"  [hazard-sens] scale={s}  种植数={n_planted}  E={ev['E_profit']:.0f}  CVaR={ev['CVaR95_loss']:.0f}  P={ev['prob_meet_threshold']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, 'q3_hazard_sensitivity.csv'), index=False)
    plt.figure(figsize=(7, 4))
    plt.plot(df['hazard_scale'], df['E_profit'], marker='o', label='期望利润')
    plt.plot(df['hazard_scale'], df['CVaR95'], marker='s', label='CVaR_0.95 损失')
    plt.xlabel('灾害概率倍率'); plt.ylabel('元/亩 (5年)')
    plt.title('Q3 灾害概率灵敏性')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'q3_hazard_sensitivity.png'), dpi=130); plt.close()
    return df


def main(N_scenarios=80, N_eval=2000, time_limit_per=600):
    data_xlsx = os.path.join(ROOT, 'data', '附件1.xlsx')
    y_pred_csv = os.path.join(OUTDIR, 'yield_pred_2026_2030.csv')
    hist_avg = hist_avg_total(data_xlsx)

    pred = pd.read_csv(y_pred_csv)
    Yhat = {(r['区块'], int(r['年份'])): float(r['Y_pred']) for _, r in pred.iterrows()}
    Y_std = {(r['区块'], int(r['年份'])): float(r.get('Y_std', 30.0)) for _, r in pred.iterrows()} \
            if 'Y_std' in pred.columns else {k: 30.0 for k in Yhat}

    print(f'[Q3] 采样 训练 N={N_scenarios} + 评估 N={N_eval} 情景 ...')
    Lam_tr, Price_tr = sample_scenarios(N=N_scenarios, seed=42)
    Lam_ev, Price_ev = sample_scenarios(N=N_eval, seed=2026)
    Y_tr = sample_yhat_with_noise(Yhat, Y_std, N_scenarios, seed=100)
    print(f'[Q3] 训练集 平均 lambda = {Lam_tr.mean():.4f}, 评估集 = {Lam_ev.mean():.4f}')
    print(f'[Q3] Q1 残差 σ 注入: 平均 = {np.mean(list(Y_std.values())):.1f}')

    lambdas = [0.0, 0.1, 0.3, 0.5, 0.8, 1.2]
    pareto = []
    best_plans = {}
    for lam in lambdas:
        res = solve_risk_model(y_pred_csv, hist_avg, Lam_tr, Price_tr, Y_tr,
                               lambda_cvar=lam, time_limit=time_limit_per, log=True)
        if res['status'] != 'Optimal':
            print(f"  [WARN] λ={lam} 未求得最优 (status={res['status']})，跳过")
            continue
        ev = evaluate_plan(res['plan'], Yhat, Y_std, Lam_ev, Price_ev, hist_avg, seed_eval=2026)
        pareto.append({
            'lambda_cvar': lam,
            'E_profit': ev['E_profit'],
            'CVaR95_loss': ev['CVaR95_loss'],
            'VaR95_loss': ev['VaR95_loss'],
            'Std_profit': ev['Std_profit'],
            'prob_meet_threshold': ev['prob_meet_threshold'],
            'min_profit': ev['min_profit'],
            'max_profit': ev['max_profit'],
        })
        best_plans[lam] = res['plan']
        res['plan'].to_csv(os.path.join(OUTDIR, f'q3_plan_lambda_{lam}.csv'), index=False)
        print(f"  [eval] λ={lam}  E={ev['E_profit']:.0f}  CVaR95={ev['CVaR95_loss']:.0f}  σ={ev['Std_profit']:.0f}  P(stab)={ev['prob_meet_threshold']:.3f}")

    pareto_df = pd.DataFrame(pareto)
    pareto_df.to_csv(os.path.join(OUTDIR, 'q3_pareto.csv'), index=False)
    print('\n[Q3] Pareto 前沿 (独立评估集):'); print(pareto_df.round(2))

    ranked = topsis_select(pareto_df)
    print('\n[Q3] TOPSIS 排序:'); print(ranked.round(3))
    chosen_lam = float(ranked.iloc[0]['lambda_cvar'])
    chosen_plan = best_plans[chosen_lam]
    chosen_plan.to_csv(os.path.join(OUTDIR, 'q3_plan_recommended.csv'), index=False)
    chosen_eval = evaluate_plan(chosen_plan, Yhat, Y_std, Lam_ev, Price_ev, hist_avg, seed_eval=2026)

    # --- 灵敏性 + 收敛性 (论文加分) ---
    print('\n[Q3] SAA 收敛性分析 ...')
    saa_convergence(y_pred_csv, hist_avg, Yhat, Y_std,
                    Ns=(40, 80, 160), lambda_cvar=chosen_lam, time_limit=time_limit_per)
    print('\n[Q3] 灾害概率灵敏性分析 ...')
    hazard_sensitivity(y_pred_csv, hist_avg, Yhat, Y_std,
                       scales=(0.5, 1.0, 1.5, 2.0), lambda_cvar=chosen_lam,
                       N_train=N_scenarios, N_eval=1000, time_limit=time_limit_per)

    with open(os.path.join(OUTDIR, 'q3_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'pareto': pareto,
            'chosen_lambda': chosen_lam,
            'chosen_metrics': pareto_df[pareto_df['lambda_cvar'] == chosen_lam].iloc[0].to_dict(),
            'N_train': N_scenarios, 'N_eval': N_eval,
            'q1_residual_sigma_mean': float(np.mean(list(Y_std.values()))),
            'beta_stab': 0.50,
            'stability_threshold_kg_mu': 0.65 * hist_avg,
        }, f, indent=2, ensure_ascii=False, default=float)

    # ---- 可视化 ----
    # Pareto 2D
    plt.figure(figsize=(6, 5))
    plt.scatter(pareto_df['CVaR95_loss'], pareto_df['E_profit'], s=70, c='steelblue', zorder=3)
    for _, r in pareto_df.iterrows():
        plt.annotate(f"λ={r['lambda_cvar']}", (r['CVaR95_loss'], r['E_profit']), fontsize=9)
    chosen_row = pareto_df[pareto_df['lambda_cvar'] == chosen_lam].iloc[0]
    plt.scatter([chosen_row['CVaR95_loss']], [chosen_row['E_profit']],
                s=180, facecolors='none', edgecolors='red', linewidth=2,
                label='TOPSIS 折中解', zorder=4)
    plt.xlabel('CVaR₀.₉₅ 损失 (元/亩)'); plt.ylabel('期望利润 (元/亩)')
    plt.title('Q3 Pareto 前沿 (独立评估)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'q3_pareto.png'), dpi=130); plt.close()

    # Pareto 3D 投影
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pareto_df['E_profit'], pareto_df['CVaR95_loss'],
                   pareto_df['prob_meet_threshold'], c='steelblue', s=60)
        for _, r in pareto_df.iterrows():
            ax.text(r['E_profit'], r['CVaR95_loss'], r['prob_meet_threshold'],
                    f"λ={r['lambda_cvar']}", fontsize=8)
        ax.set_xlabel('E[Profit]'); ax.set_ylabel('CVaR_0.95'); ax.set_zlabel('P(稳产)')
        ax.set_title('Q3 三维 Pareto 前沿')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGDIR, 'q3_pareto_3d.png'), dpi=130); plt.close()
    except Exception as e:
        print('[3D plot skipped]', e)

    piv = chosen_plan.pivot(index='区块', columns='年份', values='plant').loc[BLOCKS]
    piv_int = chosen_plan.pivot(index='区块', columns='年份', values='intensity').loc[BLOCKS]
    plt.figure(figsize=(7, 4.5))
    plt.imshow(piv_int.values, aspect='auto', cmap='YlGn', vmin=0, vmax=1.2)
    plt.xticks(range(len(YEARS)), YEARS); plt.yticks(range(len(BLOCKS)), BLOCKS)
    for i in range(len(BLOCKS)):
        for j in range(len(YEARS)):
            v = piv_int.values[i, j]
            txt = f'{v:.2f}' if piv.values[i, j] else '休耕'
            plt.text(j, i, txt, ha='center', va='center', fontsize=8)
    plt.colorbar(label='管理强度系数')
    plt.title(f'Q3 推荐方案 (λ={chosen_lam}, TOPSIS)')
    plt.xlabel('年份'); plt.ylabel('区块')
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'q3_plan.png'), dpi=130); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(chosen_eval['sc_profit'], bins=40, color='lightcoral', edgecolor='black')
    plt.axvline(chosen_eval['E_profit'], ls='--', color='blue',
                label=f"E={chosen_eval['E_profit']:.0f}")
    plt.axvline(-chosen_eval['CVaR95_loss'], ls='--', color='red',
                label=f"-CVaR₉₅={-chosen_eval['CVaR95_loss']:.0f}")
    plt.xlabel('5年总利润 (元/亩)'); plt.ylabel('频次')
    plt.title('Q3 推荐方案利润分布 (评估集)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'q3_profit_hist.png'), dpi=130); plt.close()

    print(f"\n[Q3] 推荐 λ = {chosen_lam}, 方案保存到 outputs/q3_plan_recommended.csv")


if __name__ == '__main__':
    main()
