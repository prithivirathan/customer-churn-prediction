# ============================================================
#   Predictive Modeling — Customer Churn Prediction
#   Tools  : Python, Scikit-learn, Matplotlib, Seaborn
#   Dataset: Telco Customer Churn (Kaggle)
#   Target : Predict whether a customer will churn (Yes/No)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

sns.set_theme(style="whitegrid")

# ============================================================
# STEP 1 — Load Dataset
# ============================================================
df = pd.read_csv("churn.csv")
print("✅ Dataset loaded!")
print(f"   Rows: {df.shape[0]}  |  Columns: {df.shape[1]}\n")

# ============================================================
# STEP 2 — Data Cleaning
# ============================================================
# Drop customerID (not useful for prediction)
df.drop(columns=["customerID"], inplace=True)

# TotalCharges has spaces — convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows with missing values
df.dropna(inplace=True)

# Convert target: Yes → 1, No → 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("✅ Data cleaned!")
print(f"   Churn rate: {df['Churn'].mean()*100:.1f}%\n")

# ============================================================
# STEP 3 — Encode Categorical Columns
# ============================================================
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print(f"✅ Encoded {len(cat_cols)} categorical columns\n")

# ============================================================
# STEP 4 — Feature & Target Split
# ============================================================
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"✅ Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}\n")

# ============================================================
# STEP 5 — Train 3 Models
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "accuracy": acc}
    print(f"✅ {name}: Accuracy = {acc*100:.2f}%")

print()

# ============================================================
# CHART 1 — Model Accuracy Comparison (Bar Chart)
# ============================================================
plt.figure(figsize=(8, 5))
names = list(results.keys())
accs  = [results[n]["accuracy"]*100 for n in names]
bars  = plt.bar(names, accs, color=["#4C72B0","#55A868","#C44E52"], edgecolor="white", width=0.5)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
plt.ylabel("Accuracy (%)")
plt.ylim(70, 100)
plt.tight_layout()
plt.savefig("chart1_model_accuracy.png", dpi=150)
plt.show()
print("✅ Chart 1 saved: chart1_model_accuracy.png")

# ============================================================
# CHART 2 — Confusion Matrix (Best Model = Random Forest)
# ============================================================
best_preds = results["Random Forest"]["preds"]
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn","Churn"])
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Random Forest", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("chart2_confusion_matrix.png", dpi=150)
plt.show()
print("✅ Chart 2 saved: chart2_confusion_matrix.png")

# ============================================================
# CHART 3 — Churn Distribution (Count Plot)
# ============================================================
plt.figure(figsize=(6, 5))
ax = sns.countplot(x="Churn", data=df, palette="Set2")
ax.set_xticklabels(["No Churn", "Churn"])
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.title("Churn Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("chart3_churn_distribution.png", dpi=150)
plt.show()
print("✅ Chart 3 saved: chart3_churn_distribution.png")

# ============================================================
# CHART 4 — Feature Importance (Random Forest)
# ============================================================
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(9, 5))
sns.barplot(x=top10.values, y=top10.index, palette="viridis")
plt.title("Top 10 Feature Importances — Random Forest", fontsize=14, fontweight="bold")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("chart4_feature_importance.png", dpi=150)
plt.show()
print("✅ Chart 4 saved: chart4_feature_importance.png")

# ============================================================
# CHART 5 — Monthly Charges: Churn vs No Churn (Box Plot)
# ============================================================
plt.figure(figsize=(7, 5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="Set1")
plt.xticks([0, 1], ["No Churn", "Churn"])
plt.title("Monthly Charges: Churn vs No Churn", fontsize=14, fontweight="bold")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges ($)")
plt.tight_layout()
plt.savefig("chart5_monthly_charges.png", dpi=150)
plt.show()
print("✅ Chart 5 saved: chart5_monthly_charges.png")

# ============================================================
# CHART 6 — Tenure: Churn vs No Churn (Histogram)
# ============================================================
plt.figure(figsize=(8, 5))
df[df["Churn"]==0]["tenure"].plot(kind="hist", bins=30, alpha=0.6,
                                   color="steelblue", label="No Churn", edgecolor="white")
df[df["Churn"]==1]["tenure"].plot(kind="hist", bins=30, alpha=0.6,
                                   color="tomato", label="Churn", edgecolor="white")
plt.title("Tenure Distribution: Churn vs No Churn", fontsize=14, fontweight="bold")
plt.xlabel("Tenure (Months)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("chart6_tenure_distribution.png", dpi=150)
plt.show()
print("✅ Chart 6 saved: chart6_tenure_distribution.png")

# ============================================================
# STEP 6 — Print Classification Report (Best Model)
# ============================================================
print("\n📋 Classification Report — Random Forest:")
print(classification_report(y_test, best_preds, target_names=["No Churn", "Churn"]))

# ============================================================
# DONE
# ============================================================
print("\n🎉 All charts and models completed successfully!")
print("📁 Files saved:")
for c in [
    "chart1_model_accuracy.png",
    "chart2_confusion_matrix.png",
    "chart3_churn_distribution.png",
    "chart4_feature_importance.png",
    "chart5_monthly_charges.png",
    "chart6_tenure_distribution.png"
]:
    print(f"   ✅ {c}")
