"""
01_data_prep.py
数据加载、清洗、变量构造

变量设计:
  D (处理变量) = ai_adoption_stage 二值化: partial/full → 1, none/pilot → 0
  Y (结果变量) = log(annual_revenue_usd_millions / num_employees)
  X (协变量)   = 除标识列和 D/Y 以外的所有可用变量

输出:
  - output/tables/cleaned_data.csv: 清洗后数据
  - output/tables/data_info.txt:   数据基本信息
"""

import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_TABLES = os.path.join(PROJECT_ROOT, "output", "tables")
os.makedirs(OUTPUT_TABLES, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("01 数据预处理")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
csv_path = os.path.join(DATA_RAW_DIR, "ai_company_adoption.csv")
df = pd.read_csv(csv_path)
print(f"\n原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")

# ============================================================
# 2. 列名标准化 & 变量设定
# ============================================================
# 标准化列名 -> 小写下划线
col_rename = {}
for col in df.columns:
    norm = col.lower().strip().replace(" ", "_").replace("-", "_")
    norm = norm.replace("(", "").replace(")", "")
    col_rename[col] = norm
df = df.rename(columns=col_rename)

print("\n标准化列名:")
for i, c in enumerate(df.columns, 1):
    print(f"  {i:2d}. {c}")

# ---------- 关键变量定义 ----------
# 根据实际数据集列名设定（已验证）
D_COL = "ai_adoption_stage"
Y_REVENUE = "annual_revenue_usd_millions"
Y_EMPLOYEES = "num_employees"

# 标识列（不作为协变量）
ID_COLS = ["response_id", "company_id"]

# 分类协变量（将 one-hot 编码）
CAT_COVARIATES = [
    "industry", "country", "region", "company_size",
    "survey_year", "quarter", "company_age_group",
    "data_privacy_level", "ai_ethics_committee",
    "survey_source", "data_collection_method"
]

# 数值协变量（自动选取: 所有数值列 - ID列 - D/Y相关列）
# 在 3. 中自动识别

print(f"\n核心变量:")
print(f"  D: {D_COL}")
print(f"  Y: log({Y_REVENUE} / {Y_EMPLOYEES})")
print(f"  分类协变量 ({len(CAT_COVARIATES)}): {CAT_COVARIATES}")

# ============================================================
# 3. 检查 AI 采纳阶段分布
# ============================================================
stage_counts = df[D_COL].value_counts()
print(f"\n{D_COL} 分布:")
for val, cnt in stage_counts.items():
    print(f"  {str(val):12s} {cnt:7d} ({cnt/len(df)*100:.1f}%)")

# ============================================================
# 4. 构造处理变量 D
# ============================================================
# partial/full → 高采纳 = 1（企业已有实质性的 AI 应用）
# none/pilot   → 低采纳 = 0（尚未采纳或仅试验阶段）
HIGH_STAGES = ["partial", "full"]

df["d_ai_adoption"] = df[D_COL].apply(
    lambda x: 1 if str(x).lower() in HIGH_STAGES else 0
)

treat_counts = df["d_ai_adoption"].value_counts()
print(f"\n二值化结果 (d_ai_adoption):")
print(f"  高采纳 (D=1): {treat_counts.get(1, 0):7d} ({treat_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"  低采纳 (D=0): {treat_counts.get(0, 0):7d} ({treat_counts.get(0, 0)/len(df)*100:.1f}%)")

# ============================================================
# 5. 构造结果变量 Y
# ============================================================
df["y_log_productivity"] = np.log(
    df[Y_REVENUE] / df[Y_EMPLOYEES]
)

print(f"\n劳动生产率 y_log_productivity:")
print(f"  均值 ± 标准差: {df['y_log_productivity'].mean():.4f} ± {df['y_log_productivity'].std():.4f}")
print(f"  中位数:       {df['y_log_productivity'].median():.4f}")
print(f"  最小值:       {df['y_log_productivity'].min():.4f}")
print(f"  最大值:       {df['y_log_productivity'].max():.4f}")

# ============================================================
# 6. 数值协变量自动识别
# ============================================================
all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
# 排除 ID 列、D/Y 源列
num_exclude = set(ID_COLS + [Y_REVENUE, Y_EMPLOYEES])
NUM_COVARIATES = [c for c in all_numeric if c not in num_exclude]

print(f"\n数值协变量 ({len(NUM_COVARIATES)}):")
print(f"  {NUM_COVARIATES}")

# ============================================================
# 7. 确认无缺失值，保存清洗后数据
# ============================================================
# 确认关键列无缺失
assert df["d_ai_adoption"].notna().all()
assert df["y_log_productivity"].notna().all()

# 选定保存的列
save_cols = (["d_ai_adoption", "y_log_productivity", D_COL,
              Y_REVENUE, Y_EMPLOYEES]
             + [c for c in CAT_COVARIATES if c in df.columns]
             + NUM_COVARIATES)
save_cols = list(dict.fromkeys(save_cols))  # 去重且保持顺序

df_clean = df[save_cols].copy()
df_clean.to_csv(os.path.join(OUTPUT_TABLES, "cleaned_data.csv"), index=False)

# ============================================================
# 8. 输出数据信息
# ============================================================
n_treat = (df_clean["d_ai_adoption"] == 1).sum()
n_control = (df_clean["d_ai_adoption"] == 0).sum()
y_mean = df_clean["y_log_productivity"].mean()
y_std = df_clean["y_log_productivity"].std()

info = [
    f"总样本量: {len(df_clean)}",
    f"高采纳组 (D=1): {n_treat} ({n_treat/len(df_clean)*100:.1f}%)",
    f"低采纳组 (D=0): {n_control} ({n_control/len(df_clean)*100:.1f}%)",
    f"数值协变量: {len(NUM_COVARIATES)}",
    f"分类协变量: {len([c for c in CAT_COVARIATES if c in df.columns])}",
    f"协变量总数: {len(NUM_COVARIATES) + len([c for c in CAT_COVARIATES if c in df.columns])}",
    f"Y 均值: {y_mean:.4f}",
    f"Y 标准差: {y_std:.4f}",
]

with open(os.path.join(OUTPUT_TABLES, "data_info.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(info))

print(f"\n{'='*60}")
print("预处理完成!")
print(f"{'='*60}")
for line in info:
    print(f"  {line}")
print(f"\n清洗后数据: {os.path.join(OUTPUT_TABLES, 'cleaned_data.csv')}")
print(f"  保留列 ({len(save_cols)}): {save_cols}")
