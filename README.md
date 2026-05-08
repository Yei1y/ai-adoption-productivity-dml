# AI Technology Adoption and Firm Productivity: A Double/Debiased Machine Learning Analysis

AI 技术采纳对企业劳动生产率的因果效应——基于双重机器学习的实证分析

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)

## Overview

This project investigates the causal effect of AI technology adoption on firm labor productivity using Double/Debiased Machine Learning (DML). The analysis is based on the [Global AI Adoption & Workforce Impact Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/global-ai-adoption-and-workforce-impact-dataset) (150,000 firm-level observations, 2023–2026).

**Core finding**: After controlling for firm characteristics (industry, size, automation rate, innovation capacity, etc.) using DML with cross-fitting, the causal effect of AI adoption on labor productivity is 0.006 (p = 0.342) and not statistically significant. This null average effect, however, masks substantial heterogeneity: Causal Forest analysis reveals that ~46% of firms experience positive effects while ~35% experience negative effects, driven by complementarity assets such as automation base, remote work ratio, and firm age.

## Methodology

- **Model**: Partial Linear Model
  - $Y_i = \theta_0 D_i + g_0(X_i) + \varepsilon_i$
  - $D_i = m_0(X_i) + \nu_i$
- **Method**: Double/Debiased Machine Learning (Chernozhukov et al., 2018)
  - Neyman orthogonal score
  - Cross-fitting (K=5, with robustness checks at K=2 and K=10)
- **ML nuisance models**: Random Forest (baseline), LASSO (CV), XGBoost
- **Inference**: Eicker-Huber-White robust standard errors
- **Heterogeneity analysis**: Causal Forest (Wager & Athey, 2018) via `CausalForestDML`

## Variable Design

| Type | Variable | Construction |
|------|----------|-------------|
| D (Treatment) | AI adoption | Binarized adoption stage (partial/full → 1, none/pilot → 0) |
| Y (Outcome) | Labor productivity | log(Annual Revenue / Employees) |
| X (Covariates) | Firm characteristics | Industry, country, region, size, age, automation rate, R&D, governance, etc. (64 dimensions after one-hot) |

## Key Results

| Specification | ATE | SE | p-value |
|--------------|-----|----|---------|
| Baseline (RF, K=5) | 0.0059 | 0.0062 | 0.342 |
| LASSO (CV) | 0.0004 | 0.0064 | 0.948 |
| XGBoost | 0.0042 | 0.0065 | 0.519 |
| Strict D (full only) | 0.0557 | 0.0256 | 0.030 |

**Diagnostic tests**: All 4/4 passed — common support, placebo test (0% false positive), covariate balance (mean |corr(D_res, X)| = 0.017), sensitivity to unobserved confounding (critical ρ > 0.6).

## Repository Structure

```
ai-adoption-productivity-dml/
├── scripts/                      # Python analysis scripts
│   ├── 01_data_prep.py           # Data loading and variable construction
│   ├── 02_visualization.py       # Exploratory data analysis figures
│   ├── 03_dml_estimation.py      # Core DML estimation
│   ├── 04_robustness.py          # Robustness checks (alternative ML, K-folds, outcome)
│   ├── 05_diagnostics.py         # Endogeneity diagnostics (overlap, placebo, sensitivity)
│   └── 06_heterogeneity_analysis.py  # Causal Forest for heterogeneous treatment effects
├── report/
│   ├── paper.tex                 # Main LaTeX paper (Chinese)
│   └── appendix_code.tex         # Code appendix
├── output/
│   ├── figures/                  # Generated figures (PDF)
│   │   ├── fig1_ai_adoption_by_industry.pdf
│   │   ├── fig2_productivity_by_adoption.pdf
│   │   ├── fig3_adoption_by_firm_size.pdf
│   │   ├── fig4_correlation_heatmap.pdf
│   │   ├── fig5_overlap.pdf
│   │   ├── fig6_covariate_balance.pdf
│   │   ├── fig7_sensitivity.pdf
│   │   ├── fig8_cate_distribution.pdf
│   │   └── fig9_feature_importance.pdf
│   └── tables/                   # Generated tables (CSV)
│       ├── dml_results.csv
│       ├── robustness_results.csv
│       ├── diagnostics_results.csv
│       ├── sensitivity_analysis.csv
│       ├── cate_summary.csv
│       ├── causal_forest_results.csv
│       └── multivalue_results.csv
├── logs/
│   └── analysis_log.md           # Analysis log with interpretations
├── data/
│   └── raw/                      # Raw data (not committed, download from Kaggle)
├── CLAUDE.md                     # Project instructions for Claude Code
└── README.md
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
- Key packages: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, xgboost, econml
- LaTeX (TeX Live or MiKTeX) for paper compilation

## Data

The dataset is from Kaggle: [Global AI Adoption & Workforce Impact Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/global-ai-adoption-and-workforce-impact-dataset). Download it and place it in `data/raw/` before running the scripts.

## References

- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Acemoglu, D., & Restrepo, P. (2018). The race between man and machine. *American Economic Review*, 108(6), 1488–1542.
- Syverson, C. (2011). What determines productivity? *Journal of Economic Literature*, 49(2), 326–365.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242.

## Acknowledgments

This project was developed with the assistance of [Claude Code](https://claude.ai/code), an AI-powered coding tool by Anthropic. The full research process — from project planning and data analysis to paper writing and repository management — was conducted collaboratively through Claude Code's interactive development environment.
