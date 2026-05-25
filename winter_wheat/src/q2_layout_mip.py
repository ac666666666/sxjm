"""
问题2：2026-2030 种植布局确定性优化 (MIP, 优化版)
改进点：
- d ≤ x 约束 (休耕年份 d 强制为 0，杜绝非法 intensity)
- 价格 / 基础成本 / GAMMA 灵敏性分析 (±20%)
- 基线对比：全部种植 / 随机轮作 / 历史外推
- 中文图、强度热力图
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
PRICE = 2.8         # 元/kg, 参考国家粮油信息中心 2024-2025 河北小麦均价 2.6-3.0
BASE_COST = 800     # 元/亩, 参考全国农产品成本收益资料汇编
GAMMA = 300         # 每提升 1.0 强度系数额外管理成本


def hist_avg_total(data_xlsx):
    yd = pd.read_excel(data_xlsx, sheet_name='产量数据')
    yd.columns = ['年份','区块','Y']
    return float(yd.groupby('年份')['Y'].sum().mean())


def build_and_solve(y_pred_csv, hist_avg, log=True,
                    price=PRICE, base_cost=BASE_COST, gamma=GAMMA,
                    persist=True):
    pred = pd.read_csv(y_pred_csv)
    Y = {(r['区块'], int(r['年份'])): float(r['Y_pred']) for _, r in pred.iterrows()}

    m = pulp.LpProblem('Q2_Layout', pulp.LpMaximize)
    x = {(i,t): pulp.LpVariable(f'x_{i}_{t}', cat='Binary') for i in BLOCKS for t in YEARS}
    delta_max = {i: K_MAX[i] - 1.0 for i in BLOCKS}
    d = {(i,t): pulp.LpVariable(f'd_{i}_{t}', lowBound=0, upBound=1) for i in BLOCKS for t in YEARS}
    w = {(i,t): pulp.LpVariable(f'w_{i}_{t}', lowBound=0, upBound=1) for i in BLOCKS for t in YEARS}
    for i in BLOCKS:
        for t in YEARS:
            # ★ 修复：d ≤ x 强制休耕年份 d=0，避免污染输出
            m += d[(i,t)] <= x[(i,t)]
            m += w[(i,t)] <= x[(i,t)]
            m += w[(i,t)] <= d[(i,t)]
            m += w[(i,t)] >= d[(i,t)] - (1 - x[(i,t)])

    Y_act = {(i,t): Y[(i,t)] * (x[(i,t)] + delta_max[i]*w[(i,t)])
             for i in BLOCKS for t in YEARS}

    cost = pulp.lpSum((base_cost + C_ADD[i]) * x[(i,t)] + gamma * delta_max[i] * w[(i,t)]
                      for i in BLOCKS for t in YEARS)
    revenue = pulp.lpSum(price * Y_act[(i,t)] for i in BLOCKS for t in YEARS)
    m += revenue - cost

    for t in YEARS:
        m += pulp.lpSum(x[(i,t)] for i in BLOCKS) <= 6, f'scale_{t}'
    for i in BLOCKS:
        m += pulp.lpSum(x[(i,t)] for t in YEARS) <= 4, f'rotation_{i}'
    m += pulp.lpSum(Y_act[(i,t)] for i in BLOCKS for t in YEARS) / 5.0 >= 0.65 * hist_avg, 'stability'

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    m.solve(solver)

    status = pulp.LpStatus[m.status]
    obj = pulp.value(m.objective)

    rows = []
    for t in YEARS:
        for i in BLOCKS:
            xv = round(x[(i,t)].value())
            dv = d[(i,t)].value() or 0.0
            wv = w[(i,t)].value() or 0.0
            u_eff = (1.0 + delta_max[i] * dv) if xv else 0.0
            y_act = Y[(i,t)] * (xv + delta_max[i] * wv)
            prof = price * y_act - ((base_cost + C_ADD[i]) * xv + gamma * delta_max[i] * wv)
            rows.append(dict(年份=t, 区块=i, plant=xv, intensity=round(u_eff, 4),
                             yhat=Y[(i,t)], yact=round(y_act, 2), profit=round(prof, 2)))
    plan = pd.DataFrame(rows)
    if persist:
        plan.to_csv(os.path.join(OUTDIR, 'q2_plan.csv'), index=False)

    total_profit = float(plan['profit'].sum())
    annual_yield = plan.groupby('年份')['yact'].sum()
    avg_yield = float(annual_yield.mean())
    summary = dict(
        status=status,
        objective=float(obj),
        total_profit_5y=round(total_profit, 2),
        annual_yield_kg_per_mu={int(t): round(v, 2) for t, v in annual_yield.items()},
        avg_annual_yield=round(avg_yield, 2),
        hist_avg_total=round(hist_avg, 2),
        stability_threshold=round(0.65 * hist_avg, 2),
        stability_ok=bool(avg_yield >= 0.65 * hist_avg - 1e-3),
        blocks_per_year={int(t): int((plan[plan['年份'] == t]['plant'] == 1).sum()) for t in YEARS},
        rotation_check={i: int(plan[plan['区块'] == i]['plant'].sum()) for i in BLOCKS},
        params=dict(price=price, base_cost=base_cost, gamma=gamma),
    )
    if not persist:
        # 灵敏性等子调用：不写主 summary / 主图
        return plan, summary
    with open(os.path.join(OUTDIR, 'q2_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if log:
        print('[Q2] 求解状态 =', status)
        print(f"[Q2] 5年总利润 = {total_profit:,.0f} 元/亩(累计)")
        print(f"[Q2] 平均年总产 = {avg_yield:.1f}  阈值 = {0.65*hist_avg:.1f}")
        piv = plan.pivot(index='区块', columns='年份', values='plant').loc[BLOCKS]
        print('[Q2] 种植矩阵:'); print(piv)

    # 中文图：种植 + 强度
    piv = plan.pivot(index='区块', columns='年份', values='plant').loc[BLOCKS]
    piv_int = plan.pivot(index='区块', columns='年份', values='intensity').loc[BLOCKS]
    plt.figure(figsize=(7, 4.5))
    plt.imshow(piv_int.values, aspect='auto', cmap='YlGn', vmin=0, vmax=1.2)
    plt.xticks(range(len(YEARS)), YEARS); plt.yticks(range(len(BLOCKS)), BLOCKS)
    for i in range(len(BLOCKS)):
        for j in range(len(YEARS)):
            v = piv_int.values[i, j]
            txt = f'{v:.2f}' if piv.values[i, j] else '休耕'
            plt.text(j, i, txt, ha='center', va='center', fontsize=8)
    plt.colorbar(label='管理强度系数')
    plt.title('Q2 种植方案 + 强度热力图')
    plt.xlabel('年份'); plt.ylabel('区块')
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'q2_plan.png'), dpi=130); plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar([str(t) for t in YEARS], annual_yield.values, color='steelblue', label='年总产')
    plt.axhline(0.65 * hist_avg, ls='--', color='red', label=f'65%阈值={0.65*hist_avg:.0f}')
    plt.title('Q2 各年总产量'); plt.ylabel('kg/亩 (10块加和)'); plt.xlabel('年份'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'q2_annual_yield.png'), dpi=130); plt.close()

    return plan, summary


# ---------------- 基线对比 ----------------
def baseline_all_planted(y_pred_csv, price=PRICE, base_cost=BASE_COST):
    """基线1：每年种全部 10 块、强度=1。直接违反"≤6 块"约束，仅作上界参考。"""
    pred = pd.read_csv(y_pred_csv)
    Y = {(r['区块'], int(r['年份'])): float(r['Y_pred']) for _, r in pred.iterrows()}
    profit = 0.0; ysum = 0.0
    for i in BLOCKS:
        for t in YEARS:
            profit += price * Y[(i,t)] - (base_cost + C_ADD[i])
            ysum += Y[(i,t)]
    return dict(name='全部种植(参考上界)', total_profit_5y=round(profit, 2),
                avg_annual_yield=round(ysum / 5.0, 2))

def baseline_random(y_pred_csv, hist_avg, n_trials=200, seed=0,
                    price=PRICE, base_cost=BASE_COST):
    """基线2：随机选 6 块/年，强度=1，重复 n_trials 取均值。"""
    pred = pd.read_csv(y_pred_csv)
    Y = {(r['区块'], int(r['年份'])): float(r['Y_pred']) for _, r in pred.iterrows()}
    rng = np.random.default_rng(seed)
    profits = []; yields = []; feasible = 0
    for _ in range(n_trials):
        cnt = {i: 0 for i in BLOCKS}
        ok = True; profit = 0.0; ytot = 0.0
        for t in YEARS:
            avail = [i for i in BLOCKS if cnt[i] < 4]
            if len(avail) < 6: ok = False; break
            chosen = rng.choice(avail, size=6, replace=False)
            for i in chosen:
                cnt[i] += 1
                profit += price * Y[(i,t)] - (base_cost + C_ADD[i])
                ytot += Y[(i,t)]
        if not ok: continue
        if ytot / 5.0 >= 0.65 * hist_avg: feasible += 1
        profits.append(profit); yields.append(ytot / 5.0)
    return dict(name='随机轮作(6块/年)', n_trials=len(profits),
                feasible_ratio=round(feasible / max(len(profits), 1), 3),
                mean_profit_5y=round(float(np.mean(profits)), 2),
                std_profit_5y=round(float(np.std(profits)), 2),
                mean_avg_yield=round(float(np.mean(yields)), 2))

def baseline_history(data_xlsx, price=PRICE, base_cost=BASE_COST):
    """基线3：以历史平均亩产 × 全种 6 块 (人为选历史最佳 6 块) 作参考。"""
    yd = pd.read_excel(data_xlsx, sheet_name='产量数据')
    yd.columns = ['年份','区块','Y']
    avg_blk = yd.groupby('区块')['Y'].mean().sort_values(ascending=False)
    top6 = avg_blk.head(6).index.tolist()
    profit = 0.0; ytot = 0.0
    for i in top6:
        for _ in YEARS:
            profit += price * float(avg_blk[i]) - (base_cost + C_ADD[i])
            ytot += float(avg_blk[i])
    return dict(name='历史最佳6块固定种(违反休耕)', top6=top6,
                total_profit_5y=round(profit, 2),
                avg_annual_yield=round(ytot / 5.0, 2))


# ---------------- 灵敏性分析 ----------------
def sensitivity_analysis(y_pred_csv, hist_avg):
    grid = []
    for sc_p in [0.8, 0.9, 1.0, 1.1, 1.2]:
        for sc_c in [0.8, 1.0, 1.2]:
            _, s = build_and_solve(y_pred_csv, hist_avg, log=False,
                                   price=PRICE * sc_p, base_cost=BASE_COST * sc_c,
                                   gamma=GAMMA, persist=False)
            grid.append(dict(price_scale=sc_p, cost_scale=sc_c,
                             total_profit_5y=s['total_profit_5y'],
                             avg_yield=s['avg_annual_yield'],
                             stability_ok=s['stability_ok']))
    df = pd.DataFrame(grid)
    df.to_csv(os.path.join(OUTDIR, 'q2_sensitivity.csv'), index=False)
    # 热力图
    pv = df.pivot(index='cost_scale', columns='price_scale', values='total_profit_5y')
    plt.figure(figsize=(6, 4))
    plt.imshow(pv.values, cmap='RdYlGn', aspect='auto')
    plt.xticks(range(len(pv.columns)), [f'{c:.1f}' for c in pv.columns])
    plt.yticks(range(len(pv.index)), [f'{c:.1f}' for c in pv.index])
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            plt.text(j, i, f'{pv.values[i,j]/1e3:.0f}k', ha='center', va='center', fontsize=8)
    plt.xlabel('价格倍率'); plt.ylabel('成本倍率')
    plt.title('Q2 价格×成本灵敏性 (5年总利润 元/亩)')
    plt.colorbar()
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'q2_sensitivity.png'), dpi=130); plt.close()
    return df


def main():
    print('[Q2] 求解确定性 MIP ...')
    data_xlsx = os.path.join(ROOT, 'data', '附件1.xlsx')
    y_pred_csv = os.path.join(OUTDIR, 'yield_pred_2026_2030.csv')
    hist_avg = hist_avg_total(data_xlsx)

    plan, summary = build_and_solve(y_pred_csv, hist_avg)
    print('[Q2] 主方案已保存:', os.path.join(OUTDIR, 'q2_summary.json'))

    print('[Q2] 基线对比 ...')
    b1 = baseline_all_planted(y_pred_csv)
    b2 = baseline_random(y_pred_csv, hist_avg)
    b3 = baseline_history(data_xlsx)
    bl = dict(优化MIP=dict(name='本文MIP方案', total_profit_5y=summary['total_profit_5y'],
                          avg_annual_yield=summary['avg_annual_yield'],
                          stability_ok=summary['stability_ok']),
              全种=b1, 随机轮作=b2, 历史最佳=b3)
    with open(os.path.join(OUTDIR, 'q2_baselines.json'), 'w', encoding='utf-8') as f:
        json.dump(bl, f, indent=2, ensure_ascii=False, default=float)
    print('[Q2] 基线对比已保存 q2_baselines.json')
    for k, v in bl.items():
        if 'mean_profit_5y' in v:
            print(f"  {k}: 期望利润={v['mean_profit_5y']}  可行率={v['feasible_ratio']}")
        else:
            print(f"  {k}: 利润={v.get('total_profit_5y', 'N/A')}")

    print('[Q2] 灵敏性分析 ...')
    sensitivity_analysis(y_pred_csv, hist_avg)
    print('[Q2] 灵敏性结果已保存 q2_sensitivity.csv')


if __name__ == '__main__':
    main()
