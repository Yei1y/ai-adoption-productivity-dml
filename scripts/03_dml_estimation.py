"""
03_dml_estimation.py
核心 DML 估计 —— 手动实现双重机器学习

模型: 部分线性模型 Y = θ₀D + g₀(X) + ε
方法: DML with cross-fitting (K=5), Neyman 正交评分
基线 ML: Random Forest

输出: output/tables/dml_results.csv
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
print("03 DML 估计")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
df = pd.read_csv(os.path.join(OUTPUT_TABLES, "cleaned_data.csv"))
print(f"\n数据加载: {df.shape[0]} 行 × {df.shape[1]} 列")

# ============================================================
# 2. 准备变量
# ============================================================
Y = df["y_log_productivity"].values
D = df["d_ai_adoption"].values

# 协变量选择（仅含前定企业特征，排除与 D 定义接近或后处理的变量）
# 注意: ai_adoption_rate 等 AI 强度变量会完全吸收 D 的变异，导致 DML 无法识别
COVARIATE_COLS = [
    # 企业基本特征
    "industry", "country", "region", "company_size",
    "company_age", "company_age_group", "company_founding_year",
    # 时间固定效应
    "survey_year", "quarter",
    # 企业运营特征（可视为前定）
    "task_automation_rate", "remote_work_percentage",
    "regulatory_compliance_score", "data_privacy_level",
    "ai_ethics_committee", "ai_risk_management_score",
    "employee_satisfaction_score", "innovation_score",
]
COVARIATE_COLS = [c for c in COVARIATE_COLS if c in df.columns]

# 分离数值和分类
FORCE_CATEGORICAL = ["survey_year"]
numeric_x = [c for c in COVARIATE_COLS if c in df.select_dtypes(include=[np.number]).columns]
categorical_x = [c for c in COVARIATE_COLS if c in df.select_dtypes(include=["object"]).columns]
# 强制将某些变量视为分类（如年份固定效应）
categorical_x = list(set(categorical_x + FORCE_CATEGORICAL))
numeric_x = [c for c in numeric_x if c not in categorical_x]

print(f"\n协变量构成:")
print(f"  数值: {len(numeric_x)} ({numeric_x})")
print(f"  分类: {len(categorical_x)} ({categorical_x})")

# 对分类变量进行 one-hot 编码（删除首类避免完全共线性）
df_x = pd.get_dummies(df[COVARIATE_COLS], columns=categorical_x, drop_first=True)
X = df_x.values.astype(np.float64)
p = X.shape[1]
n = len(Y)

print(f"  One-hot 后总维度: {p}")
print(f"  样本量: {n}")

# ============================================================
# 3. DML 函数定义
# ============================================================

def dml_cross_fit(Y, D, X, model_y, model_t, n_folds=5, seed=42):
    """
    DML with cross-fitting (手动实现)

    流程:
      1. 将样本分为 K 折
      2. 对每折 k:
         a. 在除 k 以外的数据上训练 nuisance 模型 E[Y|X] 和 E[D|X]
         b. 在 k 折上计算残差: Ỹ = Y - ℓ̂(X), D̃ = D - m̂(X)
      3. 合并所有残差
      4. OLS: Ỹ = θ · D̃ → θ̂ = (∑D̃ᵢỸᵢ) / (∑D̃ᵢ²)
      5. Huber-White 稳健标准误

    Parameters
    ----------
    Y : (n,) 结果变量
    D : (n,) 处理变量
    X : (n, p) 协变量矩阵
    model_y : 估计 E[Y|X] 的 ML 模型
    model_t : 估计 E[D|X] 的 ML 模型
    n_folds : 交叉拟合折数
    seed : 随机种子

    Returns
    -------
    theta, se, ci_lower, ci_upper, p_value
    """
    from sklearn.model_selection import KFold
    from scipy import stats

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    n = len(Y)
    # 存储跨折的残差
    Y_res = np.zeros(n)
    D_res = np.zeros(n)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        # Step 1: 估计 ℓ₀(X) = E[Y|X]
        model_y.fit(X_tr, Y_tr)
        Y_hat = model_y.predict(X_te)

        # Step 2: 估计 m₀(X) = E[D|X]
        model_t.fit(X_tr, D_tr)
        D_hat = model_t.predict(X_te)

        # Step 3: 残差
        Y_res[test_idx] = Y_te - Y_hat
        D_res[test_idx] = D_te - D_hat

        n_test = len(test_idx)
        print(f"    折 {fold+1}/{n_folds}: Y_res 标准差={Y_res[test_idx].std():.4f}, "
              f"D_res 标准差={D_res[test_idx].std():.4f} (测试集 n={n_test})")

    # Step 4: OLS (无截距) + epsilon 保护
    theta = np.nansum(D_res * Y_res) / (np.nansum(D_res ** 2) + 1e-10)

    # Step 5: 稳健标准误 (Eicker-Huber-White)
    e = Y_res - theta * D_res
    var_theta = np.nansum(D_res ** 2 * e ** 2) / ((np.nansum(D_res ** 2) ** 2) + 1e-10)
    se = np.sqrt(var_theta)

    # 推断
    t_stat = theta / se
    ci_lower = theta - stats.norm.ppf(0.975) * se
    ci_upper = theta + stats.norm.ppf(0.975) * se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return theta, se, ci_lower, ci_upper, p_value


# ============================================================
# 4. 基线 DML: Random Forest
# ============================================================
print(f"\n{'='*60}")
print("基线估计: DML with Random Forest")
print(f"{'='*60}")

from sklearn.ensemble import RandomForestRegressor

rf_y = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=10,
    random_state=RANDOM_SEED, n_jobs=-1
)
rf_t = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=10,
    random_state=RANDOM_SEED, n_jobs=-1
)

t0 = time.time()
theta, se, ci_l, ci_u, p_val = dml_cross_fit(Y, D, X, rf_y, rf_t, n_folds=5)
elapsed = time.time() - t0

print(f"\n{'='*60}")
print("估计结果")
print(f"{'='*60}")
print(f"  ATE (theta):      {theta:.6f}")
print(f"  标准误 (SE):     {se:.6f}")
print(f"  95% CI:         [{ci_l:.6f}, {ci_u:.6f}]")
print(f"  p 值:           {p_val:.6f}")
print(f"  样本量:         {n}")
print(f"  协变量维度:     {p}")
print(f"  运行时间:       {elapsed:.1f} 秒")

# ============================================================
# 5. 保存结果
# ============================================================
results = pd.DataFrame([{
    "method": "DML (Random Forest)",
    "estimate": round(theta, 6),
    "std_err": round(se, 6),
    "ci_lower": round(ci_l, 6),
    "ci_upper": round(ci_u, 6),
    "p_value": round(p_val, 6),
    "n": n,
    "n_covariates": p,
    "n_folds": 5,
}])
results.to_csv(os.path.join(OUTPUT_TABLES, "dml_results.csv"), index=False)

print(f"\n结果已保存: {os.path.join(OUTPUT_TABLES, 'dml_results.csv')}")
