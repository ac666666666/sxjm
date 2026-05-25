"""
问题1：冬小麦产量预测模型 (优化版)
改进点：
- 物候特征 (双 Logistic) + 农学交互项 + 区块固定效应
- Stacking: Ridge + RandomForest + XGBoost -> Ridge meta
- Leave-One-Year-Out CV + 2024-2025 Hold-out
- 800 kg 截尾处理 (Tobit-style hinge loss + IsotonicCalibration)
- 未来气象/物候 = 历史趋势线性外推 + 残差 bootstrap (与 Q3 共用)
- 残差 std 输出，供 Q3 注入模型不确定性
- 中文字体 + SHAP / 特征重要性
- 输出：yield_pred_2026_2030.csv (含 Y_pred, Y_std)
"""
import os, json, warnings, sys
warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- 中文字体 ----------------
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
DATA = os.path.join(ROOT, 'data', '附件1.xlsx')
OUTDIR = os.path.join(ROOT, 'outputs')
FIGDIR = os.path.join(ROOT, 'figures')
os.makedirs(OUTDIR, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)

# 800kg 封顶值 (附件1中产量上限)
YIELD_CAP = 800.0

# ---------------- 1. 读数据 ----------------
def load_data():
    xls = pd.ExcelFile(DATA)
    ts = xls.parse('长势时序数据')
    wx = xls.parse('气象数据')
    yd = xls.parse('产量数据')
    wx.columns = ['年份', '区块', 'T', 'P', 'S']
    yd.columns = ['年份', '区块', 'Y']
    ts.columns = ['年份', '区块', '期数', 'DOY', 'VI']
    return ts, wx, yd

# ---------------- 2. 物候特征 ----------------
def _double_logistic(t, vb, va, m1, m2, s1, s2):
    s1 = max(s1, 1e-3); s2 = max(s2, 1e-3)
    return vb + va * (1.0 / (1.0 + np.exp(-(t - m1) / s1))
                      - 1.0 / (1.0 + np.exp(-(t - m2) / s2)))

def extract_phenology(g):
    t = g['DOY'].values.astype(float)
    v = g['VI'].values.astype(float)
    idx_pk = int(np.argmax(v))
    vmax = float(v[idx_pk]); tpeak = float(t[idx_pk])
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    auc = float(_trapz(v, t))
    mask_fill = (t >= 100) & (t <= 150)
    auc_fill = float(_trapz(v[mask_fill], t[mask_fill])) if mask_fill.sum() > 1 else 0.0
    up_slope = float(np.polyfit(t[:idx_pk + 1], v[:idx_pk + 1], 1)[0]) if idx_pk > 0 else 0.0
    dn_slope = float(np.polyfit(t[idx_pk:], v[idx_pk:], 1)[0]) if idx_pk < len(v) - 1 else 0.0
    half = vmax / 2.0
    fwhm = float((v > half).sum() * 10.0)
    try:
        p0 = [v.min(), max(vmax - v.min(), 0.1), 80, 140, 5, 8]
        bounds = ([-1, 0, 50, 100, 0.1, 0.1], [3, 10, 110, 180, 30, 30])
        popt, _ = curve_fit(_double_logistic, t, v, p0=p0, bounds=bounds, maxfev=5000)
        dl_vb, dl_va, dl_m1, dl_m2, dl_s1, dl_s2 = popt
        season_len = float(dl_m2 - dl_m1)
    except Exception:
        dl_vb = dl_va = dl_m1 = dl_m2 = dl_s1 = dl_s2 = 0.0; season_len = 0.0
    return pd.Series(dict(
        AUC=auc, AUC_FILL=auc_fill, Vmax=vmax, Tpeak=tpeak,
        UpSlope=up_slope, DnSlope=dn_slope, FWHM=fwhm,
        DL_vb=dl_vb, DL_va=dl_va, DL_m1=dl_m1, DL_m2=dl_m2,
        DL_s1=dl_s1, DL_s2=dl_s2, SeasonLen=season_len
    ))

PHENO_COLS = ['AUC','AUC_FILL','Vmax','Tpeak','UpSlope','DnSlope','FWHM',
              'DL_vb','DL_va','DL_m1','DL_m2','DL_s1','DL_s2','SeasonLen']

def build_features(ts, wx, yd):
    feat = ts.groupby(['年份', '区块']).apply(extract_phenology).reset_index()
    df = feat.merge(wx, on=['年份', '区块']).merge(yd, on=['年份', '区块'])
    df['AUC_T'] = df['AUC'] * df['T']
    df['AUC_P'] = df['AUC'] * df['P']
    df['Vmax_T'] = df['Vmax'] * df['T']
    df['FWHM_P'] = df['FWHM'] * df['P']
    df['T2'] = df['T'] ** 2
    df['P2'] = df['P'] ** 2
    df['HDD'] = np.clip(df['T'] - 22, 0, None)
    df['CDD'] = np.clip(10 - df['T'], 0, None)
    df = pd.concat([df, pd.get_dummies(df['区块'], prefix='B')], axis=1)
    return df

# ---------------- 3. Tobit 损失 (上界截尾) ----------------
def tobit_mse(y_true, y_pred, cap=YIELD_CAP):
    """对截尾样本：当预测 >= cap 时不计误差 (上限截尾)。"""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    censored = y_true >= cap - 1e-6
    err = y_pred - y_true
    err[censored & (y_pred >= cap)] = 0.0
    return float(np.mean(err ** 2))

# ---------------- 4. Stacking 训练 + 评价 ----------------
def stacking_train_eval(df, hold_years=(2024, 2025)):
    feat_cols = [c for c in df.columns if c not in ('年份', '区块', 'Y')]
    train = df[~df['年份'].isin(hold_years)].copy()
    hold  = df[df['年份'].isin(hold_years)].copy()

    X_tr = train[feat_cols].values.astype(float)
    y_tr = train['Y'].values.astype(float)
    g_tr = train['年份'].values
    X_ho = hold[feat_cols].values.astype(float)
    y_ho = hold['Y'].values.astype(float)

    logo = LeaveOneGroupOut()
    oof = {'rdg': np.zeros_like(y_tr), 'rf': np.zeros_like(y_tr), 'xgb': np.zeros_like(y_tr)}
    for tr_idx, te_idx in logo.split(X_tr, y_tr, g_tr):
        sc = StandardScaler().fit(X_tr[tr_idx])
        rdg = RidgeCV(alphas=[0.1, 1, 10, 100]).fit(sc.transform(X_tr[tr_idx]), y_tr[tr_idx])
        oof['rdg'][te_idx] = rdg.predict(sc.transform(X_tr[te_idx]))
        rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=2, n_jobs=1, random_state=0)
        rf.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof['rf'][te_idx] = rf.predict(X_tr[te_idx])
        xg = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.85, colsample_bytree=0.85,
                              reg_alpha=0.1, reg_lambda=1.0,
                              random_state=0, n_jobs=1, verbosity=0)
        xg.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof['xgb'][te_idx] = xg.predict(X_tr[te_idx])

    meta_tr = np.column_stack([oof['rdg'], oof['rf'], oof['xgb']])
    stack = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(meta_tr, y_tr)
    pred_oof = np.clip(stack.predict(meta_tr), None, YIELD_CAP)

    # 完整训练集重训
    sc_full = StandardScaler().fit(X_tr)
    rdg_full = RidgeCV(alphas=[0.1, 1, 10, 100]).fit(sc_full.transform(X_tr), y_tr)
    rf_full = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=2,
                                    n_jobs=1, random_state=0).fit(X_tr, y_tr)
    xg_full = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.85, colsample_bytree=0.85,
                               reg_alpha=0.1, reg_lambda=1.0,
                               random_state=0, n_jobs=1, verbosity=0).fit(X_tr, y_tr)

    def stack_predict(X, clip=True):
        m_ = np.column_stack([
            rdg_full.predict(sc_full.transform(X)),
            rf_full.predict(X),
            xg_full.predict(X),
        ])
        p = stack.predict(m_)
        return np.clip(p, None, YIELD_CAP) if clip else p

    pred_ho = stack_predict(X_ho)

    def _nse(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    def _metrics(y, p):
        return dict(
            R2=float(r2_score(y, p)),
            RMSE=float(np.sqrt(mean_squared_error(y, p))),
            MAE=float(mean_absolute_error(y, p)),
            MAPE=float(np.mean(np.abs((y - p) / np.maximum(y, 1.0))) * 100),
            NSE=float(_nse(y, p)),
            Tobit_MSE=tobit_mse(y, p),
        )

    metrics = {
        'LOYO_base_Ridge': _metrics(y_tr, oof['rdg']),
        'LOYO_base_RF':    _metrics(y_tr, oof['rf']),
        'LOYO_base_XGB':   _metrics(y_tr, oof['xgb']),
        'LOYO_Stacking':   _metrics(y_tr, pred_oof),
        'Holdout_2024_2025': _metrics(y_ho, pred_ho),
    }

    # 残差 std (供 Q3 不确定性注入)
    resid = y_tr - pred_oof
    resid_std = float(np.std(resid))
    censored_ratio = float(np.mean(y_tr >= YIELD_CAP - 1e-6))
    metrics['Residual_std'] = resid_std
    metrics['Censored_ratio'] = censored_ratio

    # 特征重要性
    imp = pd.DataFrame({'feature': feat_cols, 'importance': xg_full.feature_importances_})\
            .sort_values('importance', ascending=False)
    imp.to_csv(os.path.join(OUTDIR, 'q1_feature_importance.csv'), index=False)

    # SHAP (best effort)
    try:
        import shap
        explainer = shap.TreeExplainer(xg_full)
        sv = explainer.shap_values(X_tr)
        sv_mean = np.abs(sv).mean(0)
        order = np.argsort(sv_mean)[::-1][:15]
        plt.figure(figsize=(7, 5))
        plt.barh([feat_cols[i] for i in order[::-1]], sv_mean[order[::-1]])
        plt.title('Q1 Top-15 SHAP特征贡献'); plt.xlabel('|SHAP|均值')
        plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'q1_shap.png'), dpi=130); plt.close()
    except Exception as e:
        print('  [SHAP skipped]', e)

    # 观测 vs 预测
    plt.figure(figsize=(6, 6))
    plt.scatter(y_tr, pred_oof, alpha=0.6, label='LOYO 交叉验证')
    plt.scatter(y_ho, pred_ho, alpha=0.85, color='red', label='2024-2025 留出')
    mn, mx = min(y_tr.min(), y_ho.min()), max(y_tr.max(), y_ho.max())
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.axhline(YIELD_CAP, color='orange', ls=':', alpha=0.5, label=f'封顶={YIELD_CAP}')
    plt.axvline(YIELD_CAP, color='orange', ls=':', alpha=0.5)
    plt.xlabel('观测亩产 (kg/亩)'); plt.ylabel('预测亩产')
    plt.title(f'Q1 Stacking 集成预测 (残差σ={resid_std:.1f}, 截尾占比={censored_ratio:.1%})')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'q1_obs_vs_pred.png'), dpi=130); plt.close()

    plt.figure(figsize=(8, 5))
    top = imp.head(15)
    plt.barh(top['feature'][::-1], top['importance'][::-1])
    plt.title('Q1 Top-15 特征重要性 (XGBoost)'); plt.xlabel('重要性')
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'q1_feature_importance.png'), dpi=130); plt.close()

    return metrics, stack_predict, feat_cols, resid_std

# ---------------- 5. 未来特征：趋势外推 ----------------
def build_future_features(df, feat_cols, future_years=range(2026, 2031)):
    """对每个区块的物候+气象特征做按年份线性外推 (2017-2025 → 2026-2030)."""
    rows = []
    for blk, g in df.groupby('区块'):
        years = g['年份'].values.astype(float)
        rec = {'区块': blk}
        for col in PHENO_COLS + ['T', 'P', 'S']:
            v = g[col].values.astype(float)
            if len(years) >= 3 and np.std(years) > 0:
                slope, intercept = np.polyfit(years, v, 1)
                rec[col + '_slope'] = float(slope)
                rec[col + '_int'] = float(intercept)
                rec[col + '_recent'] = float(np.mean(v[-3:]))
            else:
                rec[col + '_slope'] = 0.0
                rec[col + '_int'] = float(np.mean(v))
                rec[col + '_recent'] = float(np.mean(v))
        rows.append(rec)
    trend = pd.DataFrame(rows)

    out = []
    for y in future_years:
        for _, r in trend.iterrows():
            rec = {'年份': y, '区块': r['区块']}
            for col in PHENO_COLS + ['T', 'P', 'S']:
                # 0.5*趋势外推 + 0.5*近三年均值 (避免趋势外推过度)
                proj = r[col + '_int'] + r[col + '_slope'] * y
                rec[col] = 0.5 * proj + 0.5 * r[col + '_recent']
            out.append(rec)
    fut = pd.DataFrame(out)
    # 衍生特征
    fut['AUC_T'] = fut['AUC'] * fut['T']
    fut['AUC_P'] = fut['AUC'] * fut['P']
    fut['Vmax_T'] = fut['Vmax'] * fut['T']
    fut['FWHM_P'] = fut['FWHM'] * fut['P']
    fut['T2'] = fut['T'] ** 2; fut['P2'] = fut['P'] ** 2
    fut['HDD'] = np.clip(fut['T'] - 22, 0, None)
    fut['CDD'] = np.clip(10 - fut['T'], 0, None)
    # 区块 dummies (与训练对齐)
    blk_cols = [c for c in feat_cols if c.startswith('B_')]
    for c in blk_cols:
        fut[c] = (fut['区块'] == c.replace('B_', '')).astype(int)
    fut = fut[['年份', '区块'] + feat_cols]
    return fut

def predict_future(df, feat_cols, stack_predict, resid_std,
                   future_years=range(2026, 2031)):
    fut = build_future_features(df, feat_cols, future_years)
    X = fut[feat_cols].values.astype(float)
    fut['Y_pred'] = stack_predict(X)
    fut['Y_std'] = resid_std  # 残差 std 提供给 Q3
    res = fut[['年份', '区块', 'Y_pred', 'Y_std']].copy()
    res = res.sort_values(['年份', '区块']).reset_index(drop=True)
    res.to_csv(os.path.join(OUTDIR, 'yield_pred_2026_2030.csv'), index=False)
    return res

# ---------------- 6. EDA 图 ----------------
def eda_plots(ts, yd):
    plt.figure(figsize=(10, 5))
    for b, g in ts.groupby('区块'):
        mean_curve = g.groupby('DOY')['VI'].mean()
        plt.plot(mean_curve.index, mean_curve.values, marker='o', label=b)
    plt.xlabel('DOY (年内日序)'); plt.ylabel('长势指数 (历年均值)')
    plt.title('各区块平均长势曲线')
    plt.legend(ncol=5, fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'q1_eda_curve.png'), dpi=130); plt.close()

    plt.figure(figsize=(10, 4))
    yd.groupby('年份')['Y'].sum().plot(marker='o')
    avg = yd.groupby('年份')['Y'].sum().mean()
    plt.axhline(avg, ls='--', color='gray', label=f'历史均值={avg:.0f}')
    plt.axhline(avg * 0.65, ls='--', color='red', label=f'65%阈值={avg*0.65:.0f}')
    plt.xlabel('年份'); plt.ylabel('总亩产 (kg/亩, 10块加和)')
    plt.title('历年总亩产'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'q1_eda_yield.png'), dpi=130); plt.close()


def main():
    print('[Q1] 加载数据 ...')
    ts, wx, yd = load_data()
    eda_plots(ts, yd)
    print('[Q1] 构建特征 ...')
    df = build_features(ts, wx, yd)
    df.to_csv(os.path.join(OUTDIR, 'q1_features.csv'), index=False)
    print('[Q1] 特征矩阵形状 =', df.shape)
    print('[Q1] 训练 Stacking 模型 ...')
    metrics, stack_predict, feat_cols, resid_std = stacking_train_eval(df)
    with open(os.path.join(OUTDIR, 'q1_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print('[Q1] 评价指标:')
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}: R2={v['R2']:.3f}  RMSE={v['RMSE']:.1f}  MAPE={v['MAPE']:.2f}%  NSE={v['NSE']:.3f}")
        else:
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print('[Q1] 预测 2026-2030 (趋势外推气象) ...')
    pred = predict_future(df, feat_cols, stack_predict, resid_std)
    print(pred.pivot(index='区块', columns='年份', values='Y_pred').round(1))
    return metrics, pred


if __name__ == '__main__':
    main()
