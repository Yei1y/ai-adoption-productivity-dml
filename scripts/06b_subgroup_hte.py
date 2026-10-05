"""
06b_subgroup_hte.py
多值处理效应与子组异质性分析 —— 产出论文表 3 与表 4 的结果

背景:
  scripts/06_heterogeneity_analysis.py 现仅负责因果森林 CATE 估计;
  论文 4.3 节(多值处理)与 4.5 节(子组 HTE)的结果由本脚本产出。
  两者共同构成论文中"粗粒度分组 —— 细粒度数据驱动"的互补异质性视角。

方法:
  - 多值处理: 以"无采纳"(none)为参照组, 分别估计 pilot / partial / full 的 ATE
  - 子组 HTE: 沿行业、企业规模、调查年份、自动化率(中位数二分)四个维度切分,
    在每个子组内用 DML-RF (K <= 3) 估计 ATE, 并剔除该分组维度自身的哑变量,
    避免用分组变量本身去解释组内变异

输出:
  - output/tables/multivalue_results.csv
  - output/tables/hte_results.csv
  - output/figures/fig10_hte_forest.pdf
"""

import pandas as pd
import numpy as np
import os
import warnings
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
COLOR = ["#4472C4", "#ED7D31"]

print("=" * 60)
print("06b 多值处理效应与子组异质性分析")
print("=" * 60)

# ============================================================
# 1. 数据与协变量准备(与 03 基准回归一致)
# ============================================================
df = pd.read_csv(os.path.join(OUTPUT_TABLES, "cleaned_data.csv"))
print(f"\n数据加载: {df.shape[0]} 行 × {df.shape[1]} 列")

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

cat_cols = [c for c in COVARIATE_COLS
            if c in df.select_dtypes(include=["object"]).columns]
cat_cols = list(set(cat_cols + FORCE_CATEGORICAL))

df_x = pd.get_dummies(df[COVARIATE_COLS], columns=cat_cols, drop_first=True)
X_COLS = df_x.columns.tolist()
X_FULL = df_x.values.astype(np.float64)

Y = df["y_log_productivity"].values
D = df["d_ai_adoption"].values
STAGES = df["ai_adoption_stage"].str.lower().values

print(f"协变量维度(one-hot): {X_FULL.shape[1]}")


# ============================================================
# 2. DML 估计函数(与 03 相同的 Neyman 正交评分)
# ============================================================
def dml_cross_fit(Y, D, X, model_y, model_t, n_folds=5, seed=RANDOM_SEED):
    """DML 交叉拟合; 子组样本过小或无处理变异时返回 NaN"""
    from sklearn.model_selection import KFold
    from scipy import stats

    if len(Y) < 200 or np.sum(D > 0.5) < 30 or np.sum(D < 0.5) < 30:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n = len(Y)
    Y_res, D_res = np.zeros(n), np.zeros(n)

    for tr, te in kf.split(X):
        try:
            model_y.fit(X[tr], Y[tr])
            model_t.fit(X[tr], D[tr])
            Y_res[te] = Y[te] - model_y.predict(X[te])
            D_res[te] = D[te] - model_t.predict(X[te])
        except Exception:
            return np.nan, np.nan, np.nan, np.nan, np.nan

    theta = np.nansum(D_res * Y_res) / (np.nansum(D_res ** 2) + 1e-10)
    e = Y_res - theta * D_res
    se = np.sqrt(np.nansum(D_res ** 2 * e ** 2) / (np.nansum(D_res ** 2) ** 2 + 1e-10))
    p = 2 * (1 - stats.norm.cdf(abs(theta / (se + 1e-10))))
    return theta, se, theta - 1.96 * se, theta + 1.96 * se, p


def rf(seed):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(80, max_depth=8, min_samples_leaf=15,
                                 random_state=seed, n_jobs=-1)


# ============================================================
# 3. 多值处理效应(论文表 3)
# ============================================================
print(f"\n{'='*60}")
print("多值处理效应: 以 'none' 为参照组")
print(f"{'='*60}")

mv_results = []
for label, treat, ctrl in [
    ("Pilot vs None (实验 vs 无)", "pilot", "none"),
    ("Partial vs None (部分采纳 vs 无)", "partial", "none"),
    ("Full vs None (全面采纳 vs 无)", "full", "none"),
]:
    mask = (STAGES == treat) | (STAGES == ctrl)
    idx = np.where(mask)[0]
    D_sub = (STAGES[idx] == treat).astype(float)
    n_treat, n_ctrl = int(D_sub.sum()), int((1 - D_sub).sum())

    nf = min(5, max(2, min(n_treat, n_ctrl) // 100))
    th, se, cl, cu, pv = dml_cross_fit(
        Y[idx], D_sub, X_FULL[idx], rf(RANDOM_SEED), rf(RANDOM_SEED + 1), n_folds=nf
    )
    print(f"  {label:34s} ATE={th:9.6f} SE={se:8.6f} p={pv:.4f} "
          f"(n={len(idx)}, K={nf})")
    mv_results.append({
        "comparison": label, "ATE": th, "SE": se,
        "CI_lower": cl, "CI_upper": cu, "p_value": pv,
        "n_treat": n_treat, "n_control": n_ctrl, "n_total": len(idx),
    })

mv_df = pd.DataFrame(mv_results)
mv_df.to_csv(os.path.join(OUTPUT_TABLES, "multivalue_results.csv"), index=False)


# ============================================================
# 4. 子组异质性(论文表 4)
# ============================================================
print(f"\n{'='*60}")
print("子组 HTE: 行业 / 规模 / 年份 / 自动化率")
print(f"{'='*60}")

med_auto = df["task_automation_rate"].median()


def make_configs():
    """(分组维度, 水平名, 子组掩码, 是否剔除该维度哑变量)"""
    cfg = []
    for ind in sorted(df["industry"].unique()):
        cfg.append(("industry", ind, df["industry"].values == ind, True))
    for sz in sorted(df["company_size"].unique()):
        cfg.append(("company_size", sz, df["company_size"].values == sz, True))
    for yr in sorted(df["survey_year"].unique()):
        cfg.append(("survey_year", str(yr), df["survey_year"].values == yr, True))
    cfg.append(("automation_rate", "high",
                df["task_automation_rate"].values >= med_auto, False))
    cfg.append(("automation_rate", "low",
                df["task_automation_rate"].values < med_auto, False))
    return cfg


def run_hte(args):
    dim, lev, mask, do_exclude = args
    idx = np.where(mask)[0]
    exc = [X_COLS.index(c) for c in X_COLS if c.startswith(dim + "_")] if do_exclude else []
    X_sub = np.delete(X_FULL[idx], exc, axis=1) if exc else X_FULL[idx]

    n_treat = int(D[idx].sum())
    n_ctrl = int((1 - D[idx]).sum())
    if n_treat < 50 or n_ctrl < 50:
        return {"dimension": dim, "level": lev, "n_treat": n_treat,
                "n_control": n_ctrl, "n_total": len(idx),
                "ATE": np.nan, "SE": np.nan, "CI_lower": np.nan,
                "CI_upper": np.nan, "p_value": np.nan}

    nf = min(3, max(2, min(n_treat, n_ctrl) // 50))
    th, se, cl, cu, pv = dml_cross_fit(
        Y[idx], D[idx], X_sub, rf(RANDOM_SEED), rf(RANDOM_SEED + 1), n_folds=nf
    )
    return {"dimension": dim, "level": lev, "n_treat": n_treat,
            "n_control": n_ctrl, "n_total": len(idx),
            "ATE": th, "SE": se, "CI_lower": cl, "CI_upper": cu, "p_value": pv}


hte_res = []
with ThreadPoolExecutor(max_workers=3) as ex:
    for f in as_completed([ex.submit(run_hte, c) for c in make_configs()]):
        hte_res.append(f.result())

# 固定行序, 保证结果文件可复现(并行完成顺序本身不确定)
DIM_ORDER = {"industry": 0, "company_size": 1, "survey_year": 2, "automation_rate": 3}
hte_df = pd.DataFrame(hte_res)
hte_df["_k"] = hte_df["dimension"].map(DIM_ORDER)
hte_df = (hte_df.sort_values(["_k", "level"])
                .drop(columns="_k")
                .reset_index(drop=True))
hte_df.to_csv(os.path.join(OUTPUT_TABLES, "hte_results.csv"), index=False)

print(f"\n  {'dimension':16s} {'level':16s} {'ATE':>10s} {'SE':>9s} {'p':>8s} {'n':>7s}")
print("  " + "-" * 70)
for _, r in hte_df.iterrows():
    print(f"  {r['dimension']:16s} {r['level']:16s} {r['ATE']:10.6f} "
          f"{r['SE']:9.6f} {r['p_value']:8.4f} {r['n_total']:7d}")
print(f"\n  子组总数: {len(hte_df)}, "
      f"显著(p<0.05)的子组数: {int((hte_df['p_value'] < 0.05).sum())}")


# ============================================================
# 5. 森林图
# ============================================================
print(f"\n{'='*60}")
print("森林图")
print(f"{'='*60}")

plot_df = hte_df.dropna(subset=["ATE"]).copy()
plot_df["label"] = plot_df["dimension"] + ": " + plot_df["level"]
plot_df = plot_df.sort_values("ATE").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.35)))
for i, (_, r) in enumerate(plot_df.iterrows()):
    c = COLOR[1] if r["p_value"] < 0.05 else COLOR[0]
    ax.errorbar(r["ATE"], i, xerr=1.96 * r["SE"], fmt="o", color=c,
                capsize=4, markersize=5)
ax.axvline(x=0, color="gray", ls="--", alpha=0.7)
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df["label"], fontsize=9)
ax.set_xlabel("ATE (95% CI)")
ax.set_title("Heterogeneous Treatment Effects by Subgroup")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_FIGURES, "fig10_hte_forest.pdf"), bbox_inches="tight")
plt.close()
print("  -> output/figures/fig10_hte_forest.pdf")

print(f"\n{'='*60}")
print("完成")
print(f"{'='*60}")
print("  output/tables/multivalue_results.csv")
print("  output/tables/hte_results.csv")
print("  output/figures/fig10_hte_forest.pdf")
