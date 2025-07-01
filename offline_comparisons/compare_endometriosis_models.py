import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)

warnings.filterwarnings("ignore")

# --- Load and preprocess data ---
df = pd.read_csv("data/structured_endometriosis_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Feature engineering
df['age_binned'] = pd.cut(df['age'], bins=[17, 25, 35, 50], labels=[0, 1, 2])
df['bmi_binned'] = pd.cut(df['bmi'], bins=[14, 18.5, 25, 30, 40], labels=[0, 1, 2, 3])
df['pain_binned'] = pd.cut(df['chronic_pain_level'], bins=[-1, 3, 7, 10], labels=[0, 1, 2])
df['irregular_pain_combo'] = df['menstrual_irregularity'] * df['chronic_pain_level']
df['infertility_hormone_interact'] = df['infertility'] * df['hormone_level_abnormality']

df.drop(columns=['age', 'bmi', 'chronic_pain_level'], inplace=True)

for col in df.select_dtypes(include='category').columns:
    df[col] = df[col].astype(int)

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# --- Scaling and feature selection ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(score_func=f_classif, k=min(15, X_train.shape[1]))
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)

# --- Classifier definitions ---
models = {
    "SVM (App)": SVC(probability=True, class_weight="balanced", kernel='rbf', C=0.1, gamma=0.001, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

results = {}

# --- Evaluation ---
for name, model in models.items():
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    y_proba = model.predict_proba(X_test_sel)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 Results for {name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    results[name] = {
        'model': model,
        'accuracy': acc,
        'roc_auc': roc,
        'conf_matrix': cm,
        'y_proba': y_proba,
        'y_pred': y_pred
    }

# --- Plot ROC Curves ---
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Endometriosis")
plt.legend()
plt.tight_layout()
plt.savefig("offline_comparisons/roc_curve_endometriosis.png")
plt.close()

# --- Plot Accuracy & AUC Bar Chart ---
plt.figure(figsize=(8, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
aucs = [results[name]['roc_auc'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35
plt.bar(x - width/2, accuracies, width, label='Accuracy')
plt.bar(x + width/2, aucs, width, label='ROC AUC')
plt.xticks(x, model_names, rotation=15)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison - Endometriosis")
plt.legend()
plt.tight_layout()
plt.savefig("offline_comparisons/performance_comparison_endometriosis.png")
plt.close()

# --- Confusion Matrix Heatmaps ---
for name, res in results.items():
    cm = res['conf_matrix']
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    filename = f"offline_comparisons/conf_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_endometriosis.png"
    plt.savefig(filename)
    plt.close()
