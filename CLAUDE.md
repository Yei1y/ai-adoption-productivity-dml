# AI Productivity DML 项目说明

## 项目概述

研究 AI 技术采纳对企业生产率的因果效应，使用双重机器学习（Double/Debiased ML）方法。基于 Kaggle 的 Global AI Adoption & Workforce Impact Dataset。

## 论文结构（6页 LaTeX）

1. 引言（~0.5页）
2. 数据（~1.5-2页）：数据描述、变量选择理由、可视化
3. 模型建立（~1.5-2页）：Partial Linear Model 数学形式、DML流程
4. 实证结果（~1页）：基准结果、稳健性检验
5. 结论（~0.5页）

## 核心方法

- **模型**：部分线性模型 Y = θ₀D + g₀(X) + ε, D = m₀(X) + ν
- **方法**：DML with cross-fitting (K=5)
- **ML nuisance 模型**：LASSO, Random Forest, XGBoost
- **推断**：Neyman 正交评分 + 渐近正态分布

## 变量设计

| 类型 | 变量 | 构造 |
|------|------|------|
| D | AI 采纳程度 | AI adoption stage 二值化 |
| Y | 劳动生产率 | Revenue / Employees |
| X | 企业特征 | 行业、国家、规模、R&D、自动化率等 |

## 工作流程规范

### 编码与分析严格分离

所有脚本只负责：加载数据 → 处理 → 计算 → **输出结果到文件**
不分析结果，不 print 分析性结论

### 分析日志

运行脚本后，将结果解读写入 `logs/analysis_log.md`，格式：

```markdown
## [脚本名称] - [运行日期]

### 输出文件
- output/figures/xxx.png
- output/tables/xxx.csv

### 关键结果
[数值结果摘要]

### 简单分析
[对结果的解读、发现问题、注意事项，供写论文时参考]
```

## 目录结构

```
ai_productivity_dml/
├── data/raw/              # 原始数据（不提交 git）
├── scripts/               # Python 脚本（01_ 到 04_）
├── output/figures/        # 可视化输出
├── output/tables/         # 表格输出
├── report/paper.tex       # 论文 LaTeX
├── logs/analysis_log.md   # 分析日志
├── plan.md                # 项目计划
└── CLAUDE.md              # 本文件
```

## 环境

- Python：`D:\anaconda\envs\python31111`
- 主要包：pandas, numpy, scikit-learn, econml, matplotlib, seaborn, xgboost
- LaTeX 编译器建议：TeX Live 或 MiKTeX

## 注意事项

- 脚本按编号顺序执行：01 → 02 → 03 → 04
- 每个脚本应可独立运行
- 随机种子设为 42 以保证可复现性
- 所有图表保存为 PDF 格式（LaTeX 优先）
