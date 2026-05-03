"""
04_robustness.py
稳健性检验

4 项检验:
  1. 更换 ML 方法: LASSO, XGBoost
  2. 改变交叉拟合折数: K = 2, 10
  3. 替换处理变量定义: full only = 1
  4. 替换结果变量: productivity_change_percent

输出: output/tables/robustness_results.csv
"""

import pandas as pd
import numpy as np
import os
import warnings
import time
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_TABLES = os.path.join(PROJECT_ROOT, "output", "tables")
os.makedirs(OUTPUT_TABLES, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("04 稳健性检验")
print("=" * 60)

# ============================================================
# 1. 数据加载 & 通用准备
# ============================================================
df = pd.read_csv(os.path.join(OUTPUT_TABLES, "cleaned_data.csv"))
print(f"\n数据加载: {df.shape[0]} 行 × {df.shape[1]} 列")

# ---------- DML 函数（同 03，保持脚本独立） ----------
def dml_cross_fit(Y, D, X, model_y, model_t, n_folds=5, seed=42, verbose=True):
    from sklearn.model_selection import KFold
    from scipy import stats

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n = len(Y)
    Y_res = np.zeros(n)
    D_res = np.zeros(n)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        model_y.fit(X_tr, Y_tr)
        Y_hat = model_y.predict(X_te)

        model_t.fit(X_tr, D_tr)
        D_hat = model_t.predict(X_te)

        Y_res[test_idx] = Y_te - Y_hat
        D_res[test_idx] = D_te - D_hat

    theta = np.nansum(D_res * Y_res) / np.nansum(D_res ** 2)
    e = Y_res - theta * D_res
    var_theta = np.nansum(D_res ** 2 * e ** 2) / (np.nansum(D_res ** 2) ** 2)
    se = np.sqrt(var_theta)

    t_stat = theta / se
    ci_lower = theta - stats.norm.ppf(0.975) * se
    ci_upper = theta + stats.norm.ppf(0.975) * se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return theta, se, ci_lower, ci_upper, p_value


# ---------- 协变量设定（同 03） ----------
# 仅含前定企业特征，排除与 D 定义接近的 AI 强度变量
COVARIATE_COLS = [
    "industry", "country", "region", "company_size",
    "company_age", "company_age_group", "company_founding_year",
    "survey_year", "quarter",
    "task_automation_rate", "remote_work_percentage",
    "regulatory_compliance_score", "data_privacy_level",
    "ai_ethics_committee", "ai_risk_management_score",
    "employee_satisfaction_score", "innovation_score",
]
COVARIATE_COLS = [c for c in COVARIATE_COLS if c in df.columns]


FORCE_CATEGORICAL = ["survey_year"]

def prepare_X(df, extra_exclude=None):
    """one-hot 编码协变量矩阵"""
    cols = [c for c in COVARIATE_COLS if c not in (extra_exclude or [])]
    categorical_x = [c for c in cols if c in df.select_dtypes(include=["object"]).columns]
    categorical_x = list(set(categorical_x + FORCE_CATEGORICAL))
    df_x = pd.get_dummies(df[cols], columns=categorical_x, drop_first=True)
    return df_x.values.astype(np.float64)


# 基线 X 矩阵
X_base = prepare_X(df)
Y_base = df["y_log_productivity"].values
D_base = df["d_ai_adoption"].values
print(f"协变量矩阵: {X_base.shape}")

# ============================================================
# 2. 运行各稳健性检验
# ============================================================
results = []

# ---------- 基线结果（引用 03 的输出） ----------
baseline_df = pd.read_csv(os.path.join(OUTPUT_TABLES, "dml_results.csv"))
print(f"\n基线结果: theta = {baseline_df['estimate'].values[0]:.6f}")
results.append({
    "specification": "Baseline (RF, K=5)",
    "model": "Random Forest",
    "n_folds": 5,
    "estimate": baseline_df["estimate"].values[0],
    "std_err": baseline_df["std_err"].values[0],
    "ci_lower": baseline_df["ci_lower"].values[0],
    "ci_upper": baseline_df["ci_upper"].values[0],
    "p_value": baseline_df["p_value"].values[0],
    "n": int(baseline_df["n"].values[0]),
})

# ============================================================
# 检验 1: 更换 ML 方法
# ============================================================
print(f"\n{'='*60}")
print("检验 1: 更换 ML 方法")
print(f"{'='*60}")

# --- 1a: LASSO ---
print("\n[1a] LASSO (交叉验证选择 lambda)...")
from sklearn.linear_model import LassoCV

t0 = time.time()
lasso_y = LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=5000)
lasso_t = LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=5000)
theta, se, ci_l, ci_u, p_val = dml_cross_fit(
    Y_base, D_base, X_base, lasso_y, lasso_t, verbose=False
)
elapsed = time.time() - t0
print(f"  ATE = {theta:.6f}, SE = {se:.6f}, p = {p_val:.4f} ({elapsed:.0f}s)")
results.append({
    "specification": "LASSO (CV)",
    "model": "LASSO",
    "n_folds": 5,
    "estimate": theta, "std_err": se,
    "ci_lower": ci_l, "ci_upper": ci_u, "p_value": p_val,
    "n": len(Y_base),
})

# --- 1b: XGBoost ---
print("\n[1b] XGBoost...")
import xgboost as xgb

t0 = time.time()
xgb_y = xgb.XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
)
xgb_t = xgb.XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
)
theta, se, ci_l, ci_u, p_val = dml_cross_fit(
    Y_base, D_base, X_base, xgb_y, xgb_t, verbose=False
)
elapsed = time.time() - t0
print(f"  ATE = {theta:.6f}, SE = {se:.6f}, p = {p_val:.4f} ({elapsed:.0f}s)")
results.append({
    "specification": "XGBoost",
    "model": "XGBoost",
    "n_folds": 5,
    "estimate": theta, "std_err": se,
    "ci_lower": ci_l, "ci_upper": ci_u, "p_value": p_val,
    "n": len(Y_base),
})

# ============================================================
# 检验 2: 改变交叉拟合折数
# ============================================================
print(f"\n{'='*60}")
print("检验 2: 改变交叉拟合折数")
print(f"{'='*60}")

from sklearn.ensemble import RandomForestRegressor

for K in [2, 10]:
    print(f"\n[2] K = {K}...")
    rf_y = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=10,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf_t = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=10,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    t0 = time.time()
    theta, se, ci_l, ci_u, p_val = dml_cross_fit(
        Y_base, D_base, X_base, rf_y, rf_t, n_folds=K, verbose=False
    )
    elapsed = time.time() - t0
    print(f"  ATE = {theta:.6f}, SE = {se:.6f}, p = {p_val:.4f} ({elapsed:.0f}s)")
    results.append({
        "specification": f"K = {K}",
        "model": "Random Forest",
        "n_folds": K,
        "estimate": theta, "std_err": se,
        "ci_lower": ci_l, "ci_upper": ci_u, "p_value": p_val,
        "n": len(Y_base),
    })

# ============================================================
# 检验 3: 替换处理变量定义
# ============================================================
print(f"\n{'='*60}")
print("检验 3: 替换处理变量定义 (仅 full = 1)")
print(f"{'='*60}")

D_strict = (df["ai_adoption_stage"].str.lower() == "full").astype(int).values
treat_rate = D_strict.mean()
print(f"  新处理组占比: {treat_rate:.4f} ({D_strict.sum()}/{len(D_strict)})")

rf_y = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=10,
    random_state=RANDOM_SEED, n_jobs=-1
)
rf_t = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=10,
    random_state=RANDOM_SEED, n_jobs=-1
)
t0 = time.time()
theta, se, ci_l, ci_u, p_val = dml_cross_fit(
    Y_base, D_strict, X_base, rf_y, rf_t, verbose=False
)
elapsed = time.time() - t0
print(f"  ATE = {theta:.6f}, SE = {se:.6f}, p = {p_val:.4f} ({elapsed:.0f}s)")
results.append({
    "specification": "Strict D (full only)",
    "model": "Random Forest",
    "n_folds": 5,
    "estimate": theta, "std_err": se,
    "ci_lower": ci_l, "ci_upper": ci_u, "p_value": p_val,
    "n": len(Y_base),
})

# ============================================================
# 检验 4: 替换结果变量
# ============================================================
print(f"\n{'='*60}")
print("检验 4: 替换结果变量 (productivity_change_percent)")
print(f"{'='*60}")

if "productivity_change_percent" in df.columns:
    Y_alt = df["productivity_change_percent"].values
    # 排除 X 中的 productivity_change_percent（本身就是 Y 了）
    X_alt = prepare_X(df, extra_exclude=["productivity_change_percent"])
    print(f"  新 Y: productivity_change_percent (均值={Y_alt.mean():.2f})")
    print(f"  调整后协变量矩阵: {X_alt.shape}")

    rf_y = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=10,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf_t = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=10,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    t0 = time.time()
    theta, se, ci_l, ci_u, p_val = dml_cross_fit(
        Y_alt, D_base, X_alt, rf_y, rf_t, verbose=False
    )
    elapsed = time.time() - t0
    print(f"  ATE = {theta:.6f}, SE = {se:.6f}, p = {p_val:.4f} ({elapsed:.0f}s)")
    results.append({
        "specification": "Alt Y (productivity_change)",
        "model": "Random Forest",
        "n_folds": 5,
        "estimate": theta, "std_err": se,
        "ci_lower": ci_l, "ci_upper": ci_u, "p_value": p_val,
        "n": len(Y_alt),
    })
else:
    print("  [跳过] productivity_change_percent 不在数据中")

# ============================================================
# 3. 汇总结果
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_TABLES, "robustness_results.csv"), index=False)

print(f"\n{'='*60}")
print("稳健性检验汇总")
print(f"{'='*60}")
print(f"\n{'Specification':35s} {'ATE':>10s} {'SE':>8s} {'95% CI':>20s} {'p-value':>8s}")
print("-" * 85)
for _, row in results_df.iterrows():
    ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
    print(f"{row['specification']:35s} {row['estimate']:10.4f} {row['std_err']:8.4f}"
          f" {ci_str:>20s} {row['p_value']:8.4f}")

print(f"\n结果已保存: {os.path.join(OUTPUT_TABLES, 'robustness_results.csv')}")
