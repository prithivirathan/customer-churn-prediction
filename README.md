# 📉 Customer Churn Prediction — Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-green)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-teal)
![Internship](https://img.shields.io/badge/Thiranex-Internship-purple)

A machine learning project that predicts whether a telecom customer will **churn (leave) or stay**, built as part of the **Thiranex Data Science Internship**.  
Three classification models are trained, evaluated, and compared — Logistic Regression, Decision Tree, and Random Forest.

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
│
├── churn.csv                          # Dataset (download from Kaggle)
├── churn_model.py                     # Main Python script
│
├── chart1_model_accuracy.png          # Model accuracy comparison
├── chart2_confusion_matrix.png        # Confusion matrix (Random Forest)
├── chart3_churn_distribution.png      # Churn vs No Churn count
├── chart4_feature_importance.png      # Top 10 important features
├── chart5_monthly_charges.png         # Monthly charges by churn status
├── chart6_tenure_distribution.png     # Tenure distribution by churn
│
├── requirements.txt                   # Required Python libraries
└── README.md                          # Project documentation
```

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,043 |
| **Columns** | 21 |
| **Target** | `Churn` (Yes / No) |
| **Domain** | Telecom / Customer Analytics |

### Key Columns
| Column | Description |
|--------|-------------|
| `tenure` | Months the customer has been with the company |
| `MonthlyCharges` | Amount charged per month |
| `TotalCharges` | Total amount charged |
| `Contract` | Month-to-month, One year, Two year |
| `PaymentMethod` | How the customer pays |
| `InternetService` | DSL, Fiber optic, No |
| `Churn` | Target — Yes (churned) / No (stayed) |

---

## 🤖 Models Built

| Model | Description |
|-------|-------------|
| Logistic Regression | Simple, fast baseline classifier |
| Decision Tree | Easy to interpret, visualizable |
| Random Forest | Ensemble model — best accuracy |

---

## 📈 Visualizations

| # | Chart | Insight |
|---|-------|---------|
| 1 | Model Accuracy Comparison | Random Forest performs best |
| 2 | Confusion Matrix | Shows true vs predicted churn |
| 3 | Churn Distribution | ~26% customers churned |
| 4 | Feature Importance | Tenure & MonthlyCharges are top predictors |
| 5 | Monthly Charges by Churn | Churned customers pay more monthly |
| 6 | Tenure by Churn | New customers churn more often |

---

## 🔍 Key Insights

- 📉 Around **26%** of customers churned
- 💰 Customers with **higher monthly charges** are more likely to churn
- 📅 **New customers** (low tenure) churn significantly more than long-term ones
- 📋 **Month-to-month contracts** have the highest churn rate
- 🌐 **Fiber optic** internet users churn more than DSL users
- 🏆 **Random Forest** achieves the best accuracy among the three models

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas** — data loading and manipulation
- **Scikit-learn** — ML models and evaluation
- **Matplotlib & Seaborn** — visualizations

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), rename it to `churn.csv`, and place it in the project folder.

### 4. Run the script
```bash
python churn_model.py
```

---

## 📜 License

This project is for educational purposes as part of the **Thiranex Data Science Internship**.

---

## 🙋 Author

**Your Name**  
Thiranex Data Science Intern  
[GitHub](https://github.com/your-username) • [LinkedIn](https://linkedin.com/in/your-profile)
