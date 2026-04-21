# Fairness and Explainability in Machine Learning  
### Adult Income Prediction using Logistic Regression, Fairlearn, SHAP, and LIME

## Project Overview

This project focuses on building a machine learning model for income prediction using the Adult Income dataset, with a strong emphasis on **ethical AI principles**, including:

- Fairness analysis across demographic groups  
- Model explainability using SHAP and LIME  
- Transparent and interpretable machine learning  

The goal is not only to build a predictive model but also to evaluate its **bias, fairness, and interpretability**.

## Objectives

- Train a Logistic Regression model for binary income classification  
- Evaluate model performance using standard metrics  
- Perform fairness analysis using **Fairlearn**  
- Analyze group disparities using:
  - Selection Rate  
  - False Positive Rate (FPR)  
  - True Positive Rate (TPR)  
- Apply explainability techniques:
  - SHAP (global + local explanations)  
  - LIME (local explanations)

## Dataset

- **Dataset:** Adult Income Dataset  
- **Source:** UCI Machine Learning Repository  
- **Task:** Predict whether income is `>50K` or `<=50K`  
- **Sensitive Attribute:** Gender (`sex`)

## Technologies Used

- Python  
- Pandas & NumPy  
- Scikit-learn  
- Fairlearn  
- SHAP  
- LIME
- Matplotlib & Seaborn 

## Project Workflow

### 1. Data Preprocessing
- Handled missing values  
- One-hot encoding for categorical variables  
- Defined target and sensitive attributes  

### 2. Model Training
- Logistic Regression classifier  
- Train-test split (70/30)  
- Model trained using Scikit-learn  

### 3. Model Evaluation
- Accuracy score  
- Confusion matrix  
- Classification report  

### 4. Fairness Analysis (Fairlearn)
- Computed group-wise metrics using MetricFrame  
- Evaluated:
  - Selection Rate  
  - False Positive Rate  
  - True Positive Rate  
- Compared results across gender groups  

### 5. Explainability
- **SHAP**
  - Global feature importance (summary plot)  
  - Local explanation (waterfall plot)  
- **LIME**
  - Instance-level prediction explanation  

## Key Results

- The model achieved strong predictive accuracy  
- Fairness analysis revealed **group disparities across gender**  
- Important predictive features include:
  - Education  
  - Hours per week  
  - Occupation  
- SHAP and LIME improved interpretability of predictions  

## Fairness Insights

- Differences observed in selection rates between groups  
- Variations in FPR and TPR indicate potential bias  
- Bias likely originates from historical data patterns  
- Highlights the importance of fairness-aware ML systems  

## Explainability Insights

- **SHAP (Global):**
  - Identified most influential features in predictions  
- **SHAP (Local):**
  - Explained individual prediction decisions  
- **LIME:**
  - Provided human-interpretable local explanations  

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/kadiwala1234/Assignment-14-Ethical-AI.git
cd Assignment-14-Ethical-AI

**## Install Dependencies**

pip install pandas numpy scikit-learn matplotlib seaborn fairlearn shap lime

**## Run Notebook ##**
**Jupyter Notebook or Google Colab**

Ethical AI.ipynb

**# Ethical Considerations**

Machine learning models may reflect bias present in data
Gender-based disparities were observed in predictions
Fairness evaluation is essential for responsible AI
Explainability improves transparency and trust

**Future Improvements**

Apply fairness mitigation techniques (Fairlearn reductions)
Remove proxy features correlated with sensitive attributes
Experiment with advanced models (Random Forest, XGBoost)
Improve dataset balancing
Deploy fairness-aware pipelines

**Conclusion**

This project demonstrates that high accuracy does not guarantee fairness. By combining Fairlearn, SHAP, and LIME, we achieve a more transparent and ethical machine learning pipeline.
