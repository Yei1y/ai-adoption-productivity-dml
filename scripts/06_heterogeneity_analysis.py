"""
06_heterogeneity_analysis.py
基于因果森林的异质性分析 (Wager & Athey, 2018)

使用 EconML 的 CausalForestDML 在部分线性模型下估计条件平均处理效应 (CATE)

相比传统手动子组分析的改进:
  - 避免维度诅咒: 因果森林在高维协变量中自适应地寻找异质性结构
  - 避免主观分组: 数据驱动地识别处理效应异质性的来源
  - 提供个体级 CATE 估计而非粗略的子组均值

输出:
  - output/tables/cate_summary.csv: CATE 描述性统计
  - output/tables/causal_forest_results.csv: 完整估计结果
  - output/figures/fig8_cate_distribution.pdf: CATE 分布密度图
  - output/figures/fig9_feature_importance.pdf: 特征重要性条形图
"""

import pandas as pd
import numpy as np
import os
import warnings
import time
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

print("=" * 60)
print("06 基于因果森林的异质性分析")
print("=" * 60)

# ============================================================
# 1. 数据加载与协变量准备
# ============================================================
df = pd.read_csv(os.path.join(OUTPUT_TABLES, "cleaned_data.csv"))
print(f"\n数据加载: {df.shape[0]} 行 × {df.shape[1]} 列")

# 协变量设定（同 03 基准回归）
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

# 构造 X 矩阵，同时保留列名用于特征重要性映射
categorical_x = [c for c in COVARIATE_COLS
                 if c in df.select_dtypes(include=["object"]).columns]
categorical_x = list(set(categorical_x + FORCE_CATEGORICAL))
numeric_x = [c for c in COVARIATE_COLS if c not in categorical_x]

df_x = pd.get_dummies(df[COVARIATE_COLS], columns=categorical_x, drop_first=True)
X_COLUMNS = df_x.columns.tolist()
X = df_x.values.astype(np.float64)

Y = df["y_log_productivity"].values
D = df["d_ai_adoption"].values

n, p = X.shape
print(f"\n样本量: {n}")
print(f"协变量维度 (one-hot): {p}")
print(f"  数值: {len(numeric_x)} ({numeric_x})")
print(f"  分类: {len(categorical_x)} ({categorical_x})")

# ============================================================
# 2. 引入 CausalForestDML
# ============================================================
print(f"\n{'='*60}")
print("CausalForestDML 估计")
print(f"{'='*60}")

from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# 使用与基准回归相同的 nuisance 模型设定
model_y = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=10,
    random_state=RANDOM_SEED, n_jobs=-1
)
model_t = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=10,
    random_state=RANDOM_SEED, n_jobs=-1
)

cf = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=15,
    random_state=RANDOM_SEED,
    cv=5,
)

t0 = time.time()
cf.fit(Y, D, X=X)
elapsed = time.time() - t0
print(f"  训练完成: {elapsed:.1f} 秒")

# ============================================================
# 3. ATE 估计
# ============================================================
print(f"\n--- ATE 估计 ---")
ate = cf.ate(X)  # ATE
ate_lower, ate_upper = cf.ate_interval(X, alpha=0.05)
print(f"  ATE:       {ate:.6f}")
print(f"  95% CI:    [{ate_lower:.6f}, {ate_upper:.6f}]")

# ============================================================
# 4. CATE 估计
# ============================================================
print(f"\n--- CATE 估计 ---")
t0 = time.time()
cate = cf.effect(X)
print(f"  CATE 计算完成: {time.time()-t0:.1f} 秒")

cate_mean = np.mean(cate)
cate_std = np.std(cate)
cate_p10 = np.percentile(cate, 10)
cate_p90 = np.percentile(cate, 90)
cate_median = np.median(cate)
cate_min = cate.min()
cate_max = cate.max()

print(f"  均值 ± 标准差: {cate_mean:.6f} ± {cate_std:.6f}")
print(f"  中位数:        {cate_median:.6f}")
print(f"  P10 / P90:     [{cate_p10:.6f}, {cate_p90:.6f}]")
print(f"  最小值 / 最大值: {cate_min:.6f} / {cate_max:.6f}")

# ============================================================
# 5. CATE 特征重要性
# ============================================================
print(f"\n--- 特征重要性 ---")
feat_imp = cf.feature_importances_

# 将 one-hot dummy 变量的重要性归并到原始特征
def group_feature_importances(importances, feature_names, cat_features):
    """将单热编码的虚拟变量重要性归并到原始特征"""
    groups = {}
    # 按长度降序排列，避免短前缀误匹配（如 company_age 与 company_age_group）
    cat_sorted = sorted(cat_features, key=len, reverse=True)
    for name, imp in zip(feature_names, importances):
        matched = False
        for cat in cat_sorted:
            if name.startswith(cat + "_"):
                groups[cat] = groups.get(cat, 0) + imp
                matched = True
                break
        if not matched:
            # 数值变量直接使用原名
            groups[name] = groups.get(name, 0) + imp
    return groups

grouped_imp = group_feature_importances(feat_imp, X_COLUMNS, categorical_x)
imp_df = pd.DataFrame([
    {"feature": k, "importance": v} for k, v in grouped_imp.items()
]).sort_values("importance", ascending=False)

# 特征重要性归一化
imp_df["importance_pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100

print(f"\n  排名前 10 的特征:")
for i, (_, row) in enumerate(imp_df.head(10).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:25s} {row['importance_pct']:6.2f}%")

# ============================================================
# 6. 可视化 1: CATE 分布
# ============================================================
print(f"\n{'='*60}")
print("Fig8: CATE 分布")
print(f"{'='*60}")

fig, ax = plt.subplots(figsize=(8, 5))

# 直方图 + 密度曲线
ax.hist(cate, bins=80, density=True, alpha=0.6, color=COLOR[0],
        edgecolor="white", label="CATE Distribution")
# KDE 近似（使用 scipy 的 gaussian_kde 较慢，手工绘制阶梯近似）

# ATE 垂直线
ax.axvline(x=ate, color=COLOR[1], linewidth=2, linestyle="-",
           label=f"ATE = {ate:.4f}")
# 零线
ax.axvline(x=0, color="gray", linewidth=1.5, linestyle="--",
           alpha=0.7, label="Zero Effect")

# P10 / P90
ax.axvline(x=cate_p10, color=COLOR[2], linewidth=1, linestyle=":",
           alpha=0.7, label=f"P10 = {cate_p10:.4f}")
ax.axvline(x=cate_p90, color=COLOR[2], linewidth=1, linestyle=":",
           alpha=0.7, label=f"P90 = {cate_p90:.4f}")

# 标注右侧/左侧尾部比例
right_tail = np.mean(cate > 0.01)
left_tail = np.mean(cate < -0.01)
ax.annotate(f"P(CATE > 0.01) = {right_tail:.1%}",
            xy=(0.7, 0.85), xycoords="axes fraction", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.annotate(f"P(CATE < -0.01) = {left_tail:.1%}",
            xy=(0.7, 0.75), xycoords="axes fraction", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

ax.set_xlabel("Conditional Average Treatment Effect (CATE)")
ax.set_ylabel("Density")
ax.set_title("CATE Distribution from Causal Forest")
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_FIGURES, "fig8_cate_distribution.pdf"),
            bbox_inches="tight")
plt.close()
print("  -> output/figures/fig8_cate_distribution.pdf")

# ============================================================
# 7. 可视化 2: 特征重要性（top 10）
# ============================================================
print(f"\n{'='*60}")
print("Fig9: 特征重要性")
print(f"{'='*60}")

top10 = imp_df.head(10).iloc[::-1]  # 从大到小排列

fig, ax = plt.subplots(figsize=(8, 5))
colors_bar = [COLOR[0]] * len(top10)
# 突出最重要的特征
colors_bar[-1] = COLOR[1]

ax.barh(range(len(top10)), top10["importance_pct"], color=colors_bar,
        edgecolor="white")

ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10["feature"], fontsize=10)
ax.set_xlabel("Feature Importance (%)")
ax.set_title("Top 10 Features Driving Treatment Effect Heterogeneity")

# 柱上标注数值
for i, (_, row) in enumerate(top10.iterrows()):
    ax.text(row["importance_pct"] + 0.3, i,
            f"{row['importance_pct']:.1f}%", va="center", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_FIGURES, "fig9_feature_importance.pdf"),
            bbox_inches="tight")
plt.close()
print("  -> output/figures/fig9_feature_importance.pdf")

# ============================================================
# 8. 保存结果
# ============================================================
print(f"\n{'='*60}")
print("保存结果")
print(f"{'='*60}")

# CATE 描述性统计
cate_summary = pd.DataFrame([{
    "n": n,
    "p": p,
    "ate": round(ate, 6),
    "ate_ci_lower": round(ate_lower, 6),
    "ate_ci_upper": round(ate_upper, 6),
    "cate_mean": round(cate_mean, 6),
    "cate_std": round(cate_std, 6),
    "cate_median": round(cate_median, 6),
    "cate_p10": round(cate_p10, 6),
    "cate_p90": round(cate_p90, 6),
    "cate_min": round(cate_min, 6),
    "cate_max": round(cate_max, 6),
    "right_tail_pct_01": round(right_tail, 4),
    "left_tail_pct_01": round(left_tail, 4),
}])
cate_summary.to_csv(os.path.join(OUTPUT_TABLES, "cate_summary.csv"), index=False)

# 特征重要性（完整）
imp_df.to_csv(os.path.join(OUTPUT_TABLES, "causal_forest_results.csv"), index=False)

print(f"\n  已保存:")
print(f"    - output/tables/cate_summary.csv")
print(f"    - output/tables/causal_forest_results.csv")

# ============================================================
print(f"\n{'='*60}")
print("分析完成")
print(f"{'='*60}")
print(f"  CATE 均值: {cate_mean:.6f} (ATE: {ate:.6f})")
print(f"  CATE 标准差: {cate_std:.6f}  — 异质性程度")
print(f"  与基线比较: 基线 RF ATE = 0.0059 (不显著)")
print(f"  结论: CATE 分布 [P10={cate_p10:.4f}, P90={cate_p90:.4f}]")
print(f"       右尾 (CATE > 0.01): {right_tail:.1%}")
print(f"       左尾 (CATE < -0.01): {left_tail:.1%}")
