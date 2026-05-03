"""
02_visualization.py
数据可视化 —— 为论文"数据"部分提供图表

4 张图:
  Fig1: AI 采纳率的行业分布（水平柱状图）
  Fig2: 处理组 vs 对照组的劳动生产率分布（小提琴图）
  Fig3: 不同企业规模下的 AI 采纳率（分组柱状图）
  Fig4: 主要连续变量的相关性热图

输出: output/figures/fig{1..4}_*.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_TABLES = os.path.join(PROJECT_ROOT, "output", "tables")
OUTPUT_FIGURES = os.path.join(PROJECT_ROOT, "output", "figures")
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 全局样式
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COLOR_PALETTE = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]

# ============================================================
# 1. 加载数据
# ============================================================
df = pd.read_csv(os.path.join(OUTPUT_TABLES, "cleaned_data.csv"))
print(f"数据加载: {df.shape[0]} 行 × {df.shape[1]} 列")

# ============================================================
# 2. Fig1: AI 采纳率的行业分布
# ============================================================
print("\n[Fig1] AI 采纳率的行业分布...")

# 计算每个行业的采纳率
industry_rate = (
    df.groupby("industry")["d_ai_adoption"]
    .agg(rate="mean", count="count")
    .sort_values("rate")
)

fig, ax = plt.subplots(figsize=(8, 5))
colors = [COLOR_PALETTE[0] if v < 0.5 else COLOR_PALETTE[1] for v in industry_rate["rate"]]
ax.barh(industry_rate.index, industry_rate["rate"], color=colors, edgecolor="white")
ax.set_xlabel("AI Adoption Rate")
ax.set_ylabel("Industry")
ax.set_title("AI Adoption Rate by Industry")
ax.set_xlim(0, 1)

# 在柱子上标注数值
for i, (idx, row) in enumerate(industry_rate.iterrows()):
    ax.text(row["rate"] + 0.01, i, f"{row['rate']:.1%} (n={row['count']})",
            va="center", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_FIGURES, "fig1_ai_adoption_by_industry.pdf"),
            bbox_inches="tight")
plt.close()
print("  -> 已保存")

# ============================================================
# 3. Fig2: 劳动生产率分布对比（小提琴图）
# ============================================================
print("\n[Fig2] 劳动生产率分布对比...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# 左: 小提琴图
sns.violinplot(data=df, x="d_ai_adoption", y="y_log_productivity",
               palette=[COLOR_PALETTE[0], COLOR_PALETTE[1]],
               inner="quartile", ax=axes[0])
axes[0].set_xticklabels(["Low Adoption (D=0)", "High Adoption (D=1)"])
axes[0].set_ylabel("Log(Revenue / Employees)")
axes[0].set_title("Productivity Distribution by AI Adoption")

# 右: 均值对比柱状图（带 95% CI）
means = df.groupby("d_ai_adoption")["y_log_productivity"].agg(["mean", "sem"])
means.columns = ["mean", "se"]

axes[1].bar([0, 1], means["mean"], yerr=1.96 * means["se"],
            color=[COLOR_PALETTE[0], COLOR_PALETTE[1]],
            capsize=5, edgecolor="white")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Low Adoption\n(D=0)", "High Adoption\n(D=1)"])
axes[1].set_ylabel("Mean Log(Revenue / Employees)")
axes[1].set_title("Productivity Mean ± 95% CI")

# 标注均值差异
diff = means.loc[1, "mean"] - means.loc[0, "mean"]
axes[1].annotate(f"Difference = {diff:.3f}", xy=(0.5, 0.95),
                 xycoords="axes fraction", ha="center", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_FIGURES, "fig2_productivity_by_adoption.pdf"),
            bbox_inches="tight")
plt.close()
print("  -> 已保存")

# ============================================================
# 4. Fig3: 不同企业规模下的 AI 采纳率
# ============================================================
print("\n[Fig3] 不同企业规模下的 AI 采纳率...")

size_order = ["Startup", "SME", "Enterprise"]
# 实际数据中的顺序可能不同
existing_sizes = [s for s in size_order if s in df["company_size"].unique()]
other_sizes = [s for s in df["company_size"].unique() if s not in size_order]
size_order_final = existing_sizes + other_sizes

# 堆叠柱状图：每个规模下的采纳/非采纳比例
ct = pd.crosstab(df["company_size"], df["d_ai_adoption"], normalize="index")
ct = ct.reindex([s for s in size_order_final if s in ct.index])
ct.columns = ["Low Adoption", "High Adoption"]

fig, ax = plt.subplots(figsize=(7, 4.5))
ct.plot(kind="bar", stacked=True, ax=ax,
        color=[COLOR_PALETTE[0], COLOR_PALETTE[1]],
        edgecolor="white")
ax.set_ylabel("Proportion")
ax.set_xlabel("Company Size")
ax.set_title("AI Adoption Proportion by Company Size")
ax.legend(loc="lower right")

# 在柱子上标注采纳率
for i, size in enumerate(ct.index):
    rate = ct.loc[size, "High Adoption"]
    ax.text(i, rate / 2, f"{rate:.0%}", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_FIGURES, "fig3_adoption_by_firm_size.pdf"),
            bbox_inches="tight")
plt.close()
print("  -> 已保存")

# ============================================================
# 5. Fig4: 主要连续变量的相关性热图
# ============================================================
print("\n[Fig4] 相关性热图...")

# 选取主要数值变量（限制数量避免图太小）
key_numeric = [
    "y_log_productivity", "d_ai_adoption",
    "ai_adoption_rate", "task_automation_rate",
    "ai_maturity_score", "innovation_score",
    "num_ai_tools_used", "ai_investment_per_employee",
    "employee_satisfaction_score", "revenue_growth_percent",
    "customer_satisfaction", "regulatory_compliance_score",
]
key_numeric = [c for c in key_numeric if c in df.columns]

corr = df[key_numeric].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
            ax=ax)
ax.set_title("Correlation Matrix of Key Variables")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_FIGURES, "fig4_correlation_heatmap.pdf"),
            bbox_inches="tight")
plt.close()
print("  -> 已保存")

# ============================================================
print(f"\n{'='*60}")
print("可视化完成!")
print(f"{'='*60}")
print(f"4 张图表已保存至: {OUTPUT_FIGURES}")
