# Titanic - Survival Prediction using Random Forest

An end-to-end Machine Learning project predicting passenger survival on the Titanic dataset from Kaggle, achieving an evaluation score of **0.77990 (~78% accuracy)**.

## Project Overview
The goal of this project is to build a predictive model that determines whether a passenger survived the Titanic shipwreck based on passenger data (such as age, gender, ticket class, fare, and family relationships).

## Methodology & Workflow
1. **Exploratory Data Analysis (EDA):**
   - Analyzed key survival drivers across socio-economic and demographic features (`Sex`, `Pclass`, `Age`).
2. **Missing Value Imputation:**
   - Handled missing numeric fields (`Age`, `Fare`) systematically using median baseline imputation to prevent skewed distributions.
3. **Feature Engineering:**
   - Created the composite feature `FamilySize` (`SibSp` + `Parch` + 1) to distinguish solo travelers from family units.
4. **Model Architecture & Regularization:**
   - Implemented a `RandomForestClassifier` with constrained tree growth (`max_depth=4`, `min_samples_leaf=3`) to penalize noise and eliminate training-set overfitting.

## Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Platform:** Kaggle Notebooks, Git/GitHub

## Results
- **Validation Score:** 0.77990 (Top baseline accuracy on Kaggle Public Leaderboard)
- **Model:** Regularized Random Forest

## Repository Structure
- `*.ipynb`: Full step-by-step notebook containing preprocessing, feature engineering, and modeling.
- `submission.csv`: Final prediction file submitted to the Kaggle competition.
- `README.md`: Project summary and technical breakdown.
