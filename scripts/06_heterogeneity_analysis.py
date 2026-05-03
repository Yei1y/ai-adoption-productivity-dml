"""
06_heterogeneity_analysis.py
异质性分析与多值处理

包含:
  1. 多值处理: 以 none 为参照组, 分别估计 pilot/partial/full 的 ATE
  2. 异质性分析: 按行业、规模、年份、自动化率分组的子组 ATE

输出:
  - output/tables/multivalue_results.csv
  - output/tables/hte_results.csv
  - output/figures/fig8_hte_forest.pdf
"""

import pandas as pd
import numpy as np
import os, warnings, time
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_TABLES = os.path.join(PROJECT_ROOT, "output", "tables")
OUTPUT_FIGURES = os.path.join(PROJECT_ROOT, "output", "figures")
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
})
COLOR = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]

# ============================================================
# 0. 数据与通用函数
# ============================================================
df = pd.read_csv(os.path.join(OUTPUT_TABLES, "cleaned_data.csv"))

COVARIATE_COLS = [
    "industry", "country", "region", "company_size",
    "company_age", "company_age_group", "company_founding_year",
    "survey_year", "quarter",
    "task_automation_rate", "remote_work_percentage",
    "regulatory_compliance_score", "data_privacy_level",
    "ai_ethics_committee", "ai_risk_management_score",
    "employee_satisfaction_score", "innovation_score",
]

# Build X once with consistent columns
cat_cols = [c for c in COVARIATE_COLS if c in df.select_dtypes(include=["object"]).columns]
# 强制将年份作为分类固定效应
if "survey_year" in COVARIATE_COLS and "survey_year" not in cat_cols:
    cat_cols.append("survey_year")
X_full_df = pd.get_dummies(df[COVARIATE_COLS], columns=cat_cols, drop_first=True)
X_COLS = X_full_df.columns.tolist()
X_full = X_full_df.values.astype(np.float64)

Y = df["y_log_productivity"].values
D = df["d_ai_adoption"].values
stages = df["ai_adoption_stage"].values

print("=" * 60)
print("06 异质性分析与多值处理")
print("=" * 60)
print(f"  总样本量: {len(Y)}, 协变量维度: {X_full.shape[1]}")


# ============================================================
# DML 函数
# ============================================================
def dml_cross_fit(Y, D, X, model_y, model_t, n_folds=5, seed=42):
    from sklearn.model_selection import KFold
    from scipy import stats
    n_ = len(Y)
    if n_ < 200 or np.sum(D > 0.5) < 30 or np.sum(D < 0.5) < 30:
        return np.nan, np.nan, np.nan, np.nan, np.nan, None, None

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    Y_res, D_res = np.zeros(n_), np.zeros(n_)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]
        try:
            model_y.fit(X_tr, Y_tr); Y_hat = model_y.predict(X_te)
            model_t.fit(X_tr, D_tr); D_hat = model_t.predict(X_te)
        except Exception:
            return np.nan, np.nan, np.nan, np.nan, np.nan, None, None
        Y_res[test_idx] = Y_te - Y_hat
        D_res[test_idx] = D_te - D_hat

    theta = np.nansum(D_res * Y_res) / (np.nansum(D_res ** 2) + 1e-10)
    e = Y_res - theta * D_res
    se = np.sqrt(np.nansum(D_res ** 2 * e ** 2) / (np.nansum(D_res ** 2) ** 2 + 1e-10))
    t = theta / (se + 1e-10)
    pv = 2 * (1 - stats.norm.cdf(abs(t)))
    return theta, se, theta - 1.96 * se, theta + 1.96 * se, pv, D_res, Y_res


# ============================================================
# 1. 多值处理: pairwise with "none" as reference
# ============================================================
print(f"\n{'='*60}")
print("1. 多值处理分析")
print(f"{'='*60}")

from sklearn.ensemble import RandomForestRegressor

def make_rf(seed):
    return RandomForestRegressor(
        n_estimators=80, max_depth=8, min_samples_leaf=15,
        random_state=seed, n_jobs=-1, verbose=0
    )

N_FOLDS = 5  # use 5 for main multi-value analysis

multivalue_results = []
comparisons = [
    ("Pilot vs None (实验 vs 无)", "pilot", "none"),
    ("Partial vs None (部分采纳 vs 无)", "partial", "none"),
    ("Full vs None (全面采纳 vs 无)", "full", "none"),
]

for label, treat, control in comparisons:
    mask = (stages == treat) | (stages == control)
    idx = np.where(mask)[0]
    Y_sub = Y[idx]
    D_sub = (stages[idx] == treat).astype(float)
    X_sub = X_full[idx]
    n_treat = int(D_sub.sum())
    n_ctrl = int((1 - D_sub).sum())
    n_folds = min(N_FOLDS, max(2, min(n_treat, n_ctrl) // 100))

    t0 = time.time()
    th, se, cl, cu, pv, _, _ = dml_cross_fit(
        Y_sub, D_sub, X_sub,
        make_rf(RANDOM_SEED), make_rf(RANDOM_SEED + 1),
        n_folds=n_folds
    )
    mv = {"comparison": label, "ATE": th, "SE": se,
          "CI_lower": cl, "CI_upper": cu, "p_value": pv,
          "n_treat": n_treat, "n_control": n_ctrl, "n_total": len(Y_sub)}
    multivalue_results.append(mv)
    sig = " ***" if pv < 0.01 else " **" if pv < 0.05 else " *" if pv < 0.1 else ""
    print(f"  {label:30s}: ATE={th:.4f}, SE={se:.4f}, p={pv:.3f}{sig}, "
          f"n={len(Y_sub)} (t={n_treat}/c={n_ctrl}), {time.time()-t0:.0f}s")

mv_df = pd.DataFrame(multivalue_results)
mv_df.to_csv(os.path.join(OUTPUT_TABLES, "multivalue_results.csv"), index=False)
print(f"  结果保存: output/tables/multivalue_results.csv")


# ============================================================
# 2. 异质性分析
# ============================================================
print(f"\n{'='*60}")
print("2. 异质性分析")
print(f"{'='*60}")

# Faster settings for HTE (many runs)
N_FOLDS_HTE = 3

# Build subgroup configs
hte_configs = []

# 2a. By industry
for ind in df["industry"].unique():
    mask = df["industry"] == ind
    # Exclude industry from X when analyzing by industry
    exclude_cols = [c for c in X_COLS if c.startswith("industry_")]
    idx_exclude = [X_COLS.index(c) for c in exclude_cols if c in X_COLS]
    hte_configs.append(("industry", ind, mask.values, idx_exclude))

# 2b. By firm size
for sz in df["company_size"].unique():
    mask = df["company_size"] == sz
    exclude_cols = [c for c in X_COLS if c.startswith("company_size_")]
    idx_exclude = [X_COLS.index(c) for c in exclude_cols if c in X_COLS]
    hte_configs.append(("company_size", sz, mask.values, idx_exclude))

# 2c. By year
for yr in sorted(df["survey_year"].unique()):
    mask = df["survey_year"] == yr
    exclude_cols = [c for c in X_COLS if c.startswith("survey_year_")]
    idx_exclude = [X_COLS.index(c) for c in exclude_cols if c in X_COLS]
    hte_configs.append(("survey_year", str(yr), mask.values, idx_exclude))

# 2d. By automation rate (median split)
med_auto = df["task_automation_rate"].median()
for label, condition in [("high", df["task_automation_rate"].values >= med_auto),
                          ("low", df["task_automation_rate"].values < med_auto)]:
    hte_configs.append(("automation_rate", label, condition, []))


def run_hte(cfg):
    dim, level, mask, exclude_idx = cfg
    idx = np.where(mask)[0]
    Y_sub = Y[idx]; D_sub = D[idx]

    if exclude_idx:
        X_sub = np.delete(X_full[idx], exclude_idx, axis=1)
    else:
        X_sub = X_full[idx]

    n_treat = int(D_sub.sum())
    n_ctrl = int((1 - D_sub).sum())
    n_folds = min(N_FOLDS_HTE, max(2, min(n_treat, n_ctrl) // 50))

    result = {"dimension": dim, "level": level,
              "n_treat": n_treat, "n_control": n_ctrl, "n_total": len(Y_sub)}

    if n_treat < 50 or n_ctrl < 50:
        result.update({"ATE": np.nan, "SE": np.nan, "p_value": np.nan,
                       "error": "insufficient sample"})
        return result

    try:
        th, se, cl, cu, pv, _, _ = dml_cross_fit(
            Y_sub, D_sub, X_sub,
            make_rf(RANDOM_SEED), make_rf(RANDOM_SEED + 1),
            n_folds=n_folds
        )
        result.update({"ATE": th, "SE": se, "CI_lower": cl, "CI_upper": cu,
                       "p_value": pv})
    except Exception as e:
        result.update({"ATE": np.nan, "SE": np.nan, "p_value": np.nan, "error": str(e)})
    return result


hte_results = []
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(run_hte, cfg) for cfg in hte_configs]
    for future in as_completed(futures):
        r = future.result()
        hte_results.append(r)
        if not np.isnan(r.get("ATE", np.nan)):
            p = r["p_value"]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {r['dimension']:20s} {r['level']:15s}: ATE={r['ATE']:.4f}, "
                  f"SE={r['SE']:.4f}, p={p:.3f}{sig}, n={r['n_total']}")
        else:
            print(f"  {r['dimension']:20s} {r['level']:15s}: n={r['n_total']}, "
                  f"treat={r.get('n_treat', 0)}/ctrl={r.get('n_control', 0)} — "
                  f"SKIP ({r.get('error', 'N/A')})")

hte_df = pd.DataFrame(hte_results)
hte_df.to_csv(os.path.join(OUTPUT_TABLES, "hte_results.csv"), index=False)
print(f"\n  结果保存: output/tables/hte_results.csv")


# ============================================================
# 3. 可视化: 森林图 (仅展示有结果的子组)
# ============================================================
print(f"\n{'='*60}")
print("3. 绘制森林图")
print(f"{'='*60}")

plot_data = hte_df.dropna(subset=["ATE"]).copy()
if len(plot_data) > 0:
    plot_data["label"] = plot_data["dimension"] + ": " + plot_data["level"]
    plot_data = plot_data.sort_values("ATE")

    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_data) * 0.35)))
    y_pos = range(len(plot_data))
    colors = ["#4472C4" if p > 0.05 else "#ED7D31" for p in plot_data["p_value"]]

    for i, (_, row) in enumerate(plot_data.iterrows()):
        c = "#ED7D31" if row["p_value"] < 0.05 else "#4472C4"
        ax.errorbar(row["ATE"], i, xerr=1.96 * row["SE"], fmt="o",
                    color=c, capsize=4, markersize=6)
    ax.axvline(x=0, color="gray", ls="--", alpha=0.7)
    ax.axvline(x=0.0058, color="gray", ls=":", alpha=0.5, label="Baseline ATE (0.0058)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["label"], fontsize=9)
    ax.set_xlabel("ATE (95% CI)")
    ax.set_title("Heterogeneous Treatment Effects by Subgroup")

    # Add significance annotations
    for i, (_, row) in enumerate(plot_data.iterrows()):
        if row["p_value"] < 0.1:
            sig = "**" if row["p_value"] < 0.05 else "*"
            ax.annotate(sig, (row["ATE"] + 1.96 * row["SE"] + 0.002, i),
                       fontsize=10, va="center")

    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIGURES, "fig8_hte_forest.pdf"), bbox_inches="tight")
    plt.close()
    print(f"  图已保存: output/figures/fig8_hte_forest.pdf")
else:
    print("  无有效结果, 跳过绘图")

print(f"\n{'='*60}")
print("分析完成")
print(f"{'='*60}")
