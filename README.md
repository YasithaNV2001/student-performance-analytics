# Student Performance Analytics

Predicting student outcomes and uncovering learner profiles with data mining across three educational datasets.

MSc Data Science & AI coursework, **University of Moratuwa** (2026). Full write-up in IEEE format: [report.pdf](report.pdf) (revised September 2026 to correct errors in the original submission; changes are listed in its appendix, and the [original](paper/report-original-2026-04.pdf) is kept for reference). The notebook also extends the report with baseline comparisons, early-warning models and SHAP explanations ([see below](#beyond-the-report-can-we-really-predict-early)).

![Model performance summary](figures/fig9_all_models_summary.png)

## Overview

Schools collect grades, attendance and learning-platform activity, but rarely use it to spot struggling students early. This project applies a complete data mining pipeline to three datasets to answer:

1. Can we predict which students will pass or fail, and what drives that prediction?
2. Do students fall into natural behavioural groups?
3. Which combinations of behaviours are linked to success or failure?

## Datasets

| Dataset | Students | Target |
|---|---|---|
| UCI Student Performance – Mathematics | 395 | Pass/fail (final grade ≥ 10/20) |
| UCI Student Performance – Portuguese | 649 | Pass/fail (final grade ≥ 10/20) |
| xAPI-Edu-Data (learning-platform behaviour) | 480 | Low / Medium / High performance |

382 students are enrolled in both UCI courses, which allows a direct cross-subject comparison.

## Methods

- **Data quality audit and cleaning:** checks for missing values, duplicates and out-of-range values; label encoding; engineered features such as `grade_trend` and `study_absence_ratio`
- **Exploratory analysis:** grade distributions, cross-subject comparison, correlation heatmaps, engagement patterns
- **Classification:** Random Forest (Math and Portuguese pass/fail), Gradient Boosting (xAPI 3-class), evaluated on a held-out test set and with stratified 5-fold cross-validation
- **Baselines and early warning:** majority class, a one-line grade rule and Logistic Regression; models retrained without interim grades
- **Explainability:** SHAP values for overall drivers and for individual students
- **Clustering:** K-Means with PCA visualisation and silhouette analysis
- **Association rule mining:** Apriori (mlxtend) with support, confidence and lift
- **Statistical testing:** paired t-test and correlation across subjects

## Results

| Model | Test accuracy | AUC-ROC | 5-fold CV accuracy |
|---|---|---|---|
| Random Forest – Math pass/fail | **86.1%** | 0.942 | 90.9% ± 2.5% |
| Random Forest – Portuguese pass/fail | **87.7%** | 0.936 | 92.5% ± 0.9% |
| Gradient Boosting – xAPI 3-class | **76.0%** | 0.920 | 73.9% ± 6.3% |

The xAPI result is in line with the 73–79% range reported by Amrieh et al. (2016) on the same task.

### Key findings

- **Earlier grades dominate pass/fail prediction.** First- and second-period grades (G1, G2) are the top features for both subjects, followed by past failures. How much the models add beyond those grades is tested [below](#beyond-the-report-can-we-really-predict-early).
- **Portuguese grades average 2.13 points higher than Math** for the same students (paired t-test, p < 10⁻²⁰), with a moderate correlation (r = 0.48) between subjects.
- **Three learner profiles** emerge from engagement data: *Active*, *Moderate* and *Passive* (silhouette = 0.36). They closely match actual performance classes.
- **Strongest rules:** high absence → low performance (lift 2.31); past failures + high alcohol use → fail (lift 2.06).

![xAPI learner clusters](figures/fig11_xapi_clustering.png)

## Beyond the Report: Can We Really Predict Early?

The 86–88% pass/fail accuracy uses the first- and second-period grades. G2 alone correlates at r = 0.90 with the final grade, so I tested what the models add beyond it, and how well they work **before any grades exist** (notebook Sections 6.7–6.9).

**1. A one-line rule matches the models.** Predicting *pass if G2 ≥ 10* is at least as good as Random Forest when G2 is available:

| Math | Accuracy | AUC | Fail recall |
|---|---|---|---|
| Always predict *Pass* | 67.1% | 0.500 | 0% |
| Rule: G2 ≥ 10 | **87.3%** | **0.958** | **96%** |
| Logistic Regression (all features) | 86.1% | 0.945 | 92% |
| Random Forest (all features) | 86.1% | 0.942 | 92% |

Portuguese shows the same pattern: rule 90.8% vs Random Forest 87.7%.

**2. Without grades, prediction is much harder.** With only demographic, family and behavioural features, Random Forest AUC drops to **0.63 (Math)** and **0.66 (Portuguese)**. In Portuguese, where 85% of students pass, the early model reaches 78% accuracy but catches only 30% of the students who fail. Accuracy alone would hide this.

**3. The first-period grade recovers most of the signal.** A model using G1 (but not G2) reaches AUC **0.87 (Math)** and **0.83 (Portuguese)**, early enough in the year to intervene.

![Baselines vs early-warning models](figures/fig14_baselines_early_warning.png)

**4. What drives risk before grades exist (SHAP).** Past failures, the school attended and whether the student plans to go to university are the strongest drivers, followed by parental education and alcohol use. Several of these describe a student's background rather than their effort, so a system like this should be used to offer support, not to label students.

| What drives risk across students | Why one student was flagged |
|---|---|
| ![SHAP summary](figures/fig15_shap_summary.png) | ![SHAP single student](figures/fig16_shap_single_student.png) |

**Takeaway:** for early warning, a school gets more from acting on the first-period grade and from collecting engagement data (the xAPI model reaches AUC 0.92 without grades, on a different population) than from a more complex model on background data.

<details>
<summary>More figures</summary>

![Cross-subject comparison](figures/fig2_cross_subject.png)
![Feature importance](figures/fig7_feature_importance_comparison.png)
![Association rules](figures/fig12_association_rules.png)

</details>

## Project Structure

```
├── analysis.ipynb        # Full analysis, with outputs
├── report.pdf            # IEEE-format paper (revised)
├── paper/
│   ├── report.tex        # LaTeX source of the revised paper
│   ├── report_figures/   # Figures used in the paper
│   └── report-original-2026-04.pdf
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned datasets (generated by the notebook)
├── figures/              # All 16 figures (generated by the notebook)
└── requirements.txt
```

## Running It

```bash
git clone https://github.com/YasithaNV2001/student-performance-analytics.git
cd student-performance-analytics
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

All models use `random_state=42`, so results are reproducible. The notebook in this repo was last run with Python 3.11 and scikit-learn 1.8; the AUC values for two models differ from the report by less than 0.003, most likely because of newer library versions.

## Tech Stack

Python · pandas · NumPy · scikit-learn · SHAP · mlxtend · SciPy · Matplotlib · seaborn · Jupyter

## Data Sources

- P. Cortez and A. Silva, "Using data mining to predict secondary school student performance," *FUBUTEC*, 2008. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)
- E. A. Amrieh, T. Hamtini and I. Aljarah, "Mining educational data to predict student's academic performance using ensemble methods," *Int. J. Database Theory and Application*, 2016. [Kaggle: xAPI-Edu-Data](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data)

## Author

**Yasitha Priyashan** – MSc Data Science & AI, University of Moratuwa – [GitHub](https://github.com/YasithaNV2001)
