import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc, f1_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import os

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("data/Toddler Autism dataset July 2018.csv")
df.rename(columns={"Class/ASD Traits ": "ASD_Traits"}, inplace=True)
df.columns = df.columns.str.replace('"', '').str.strip().str.replace(' ', '_')
df.drop(columns=["Case_No", "Who_completed_the_test", "Qchat-10-Score"], inplace=True)
df["ASD_Traits"] = df["ASD_Traits"].map({"Yes": 1, "No": 0})

# Normalize string fields
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.lower())

# Binary encoding
df['Sex'] = df['Sex'].map({'m': 1, 'f': 0})
df['Jaundice'] = df['Jaundice'].map({'yes': 1, 'no': 0})
df['Family_mem_with_ASD'] = df['Family_mem_with_ASD'].map({'yes': 1, 'no': 0})

# Features and target
X = df.drop("ASD_Traits", axis=1)
y = df["ASD_Traits"]
categorical = ['Ethnicity']

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

# Models
models = {
    "Random Forest (App)": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Store metrics
metrics = {}
fpr_dict = {}
tpr_dict = {}

# Create folder for figures
os.makedirs("figures/autism_comparison", exist_ok=True)

# Evaluation
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n📊 Results for {name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    metrics[name] = {'accuracy': acc, 'roc_auc': roc, 'f1': f1}
    fpr_dict[name], tpr_dict[name], _ = roc_curve(y_test, y_proba)

    # Confusion matrix plot
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"figures/autism_comparison/conf_matrix_{name.replace(' ', '_')}.png")
    plt.close()

# ROC curve plot
plt.figure(figsize=(6, 5))
for name in fpr_dict:
    plt.plot(fpr_dict[name], tpr_dict[name], label=f"{name} (AUC = {metrics[name]['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Autism Models")
plt.legend()
plt.tight_layout()
plt.savefig("figures/autism_comparison/roc_curve.png")
plt.close()

# Bar plot of accuracy, ROC AUC, F1
df_metrics = pd.DataFrame(metrics).T
df_metrics.plot(kind="bar", figsize=(8, 5))
plt.title("Model Comparison (Autism)")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("figures/autism_comparison/performance_comparison.png")
plt.close()
