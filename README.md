# AI Technology Adoption and Firm Productivity: A Double/Debiased Machine Learning Analysis

AI 技术采纳对企业劳动生产率的因果效应——基于双重机器学习的实证分析

## Overview

This project investigates the causal effect of AI technology adoption on firm labor productivity using Double/Debiased Machine Learning (DML). The analysis is based on the Global AI Adoption & Workforce Impact Dataset (150,000 firm-level observations, 2023–2026).

**Core finding**: After controlling for firm characteristics (industry, size, automation rate, innovation capacity, etc.), the causal effect of AI adoption on labor productivity is 0.006 (p = 0.342) and not statistically significant. This result is robust across multiple ML methods (Random Forest, LASSO, XGBoost) and cross-fitting folds.

## Methodology

- **Model**: Partial Linear Model
  - $Y_i = \theta_0 D_i + g_0(X_i) + \varepsilon_i$
  - $D_i = m_0(X_i) + \nu_i$
- **Method**: Double/Debiased Machine Learning (Chernozhukov et al., 2018)
  - Neyman orthogonal score
  - Cross-fitting (K=5)
- **ML nuisance models**: Random Forest (baseline), LASSO (CV), XGBoost
- **Inference**: Eicker-Huber-White robust standard errors

## Variable Design

| Type | Variable | Construction |
|------|----------|-------------|
| D (Treatment) | AI adoption | Binarized adoption stage (partial/full → 1, none/pilot → 0) |
| Y (Outcome) | Labor productivity | log(Annual Revenue / Employees) |
| X (Covariates) | Firm characteristics | Industry, country, region, size, age, automation rate, R&D, governance, etc. (64 dimensions after one-hot) |

## Repository Structure

```
ai_productivity_dml/
├── scripts/                  # Python analysis scripts
│   ├── 01_data_prep.py       # Data loading and variable construction
│   ├── 02_visualization.py   # Exploratory data analysis figures
│   ├── 03_dml_estimation.py  # Core DML estimation
│   ├── 04_robustness.py      # Robustness checks
│   ├── 05_diagnostics.py     # Endogeneity diagnostics
│   └── 06_heterogeneity_analysis.py  # Heterogeneous treatment effects
├── report/
│   ├── paper.tex             # Main LaTeX paper
│   └── appendix_code.tex     # Code appendix
├── output/
│   ├── figures/              # Generated figures (PDF)
│   └── tables/               # Generated tables (CSV)
├── logs/
│   └── analysis_log.md       # Analysis log
├── data/raw/                 # Raw data (not committed)
├── README.md
└── CLAUDE.md
```

## Execution Order

Run scripts in order (each is self-contained and independently runnable):

```bash
python scripts/01_data_prep.py
python scripts/02_visualization.py
python scripts/03_dml_estimation.py
python scripts/04_robustness.py
python scripts/05_diagnostics.py
python scripts/06_heterogeneity_analysis.py
```

## Requirements

- Python 3.11
- Key packages: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, xgboost
- LaTeX (TeX Live or MiKTeX) for paper compilation

## Results Summary

| Specification | ATE | SE | p-value |
|--------------|-----|----|---------|
| Baseline (RF, K=5) | 0.0059 | 0.0062 | 0.342 |
| LASSO (CV) | 0.0004 | 0.0064 | 0.948 |
| XGBoost | 0.0042 | 0.0065 | 0.519 |
| Strict D (full only) | 0.0557 | 0.0256 | 0.030 |

All 4/4 diagnostic tests pass: common support, placebo test (0% false positive rate), covariate balance (mean |corr(D_res, X)| = 0.017), sensitivity to unobserved confounding (critical ρ > 0.6).

## References

- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Acemoglu, D., & Restrepo, P. (2018). The race between man and machine. *American Economic Review*, 108(6), 1488–1542.
- Syverson, C. (2011). What determines productivity? *Journal of Economic Literature*, 49(2), 326–365.
