import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    classification_report, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# --- Load and Preprocess Dataset ---
file_path = 'data/PCOS_data_without_infertility.xlsx'
df = pd.read_excel(file_path, sheet_name='Full_new')

# Drop unnecessary columns
cols_to_drop = [
    'Patient File No.', 'Case No', 'Age at menarche', 'FSH', 'LH', 'FSH/LH Ratio',
    'Waist:Hip Ratio', 'Blood Group', 'Hb', 'TSH', 'AMH', 'PRL', 'Vit D3',
    'PRG', 'Prolactin', 'Beta HCG', 'RBS', 'Sl. No', 'Cycle(R/I)', 'Hip(inch)',
    'Waist(inch)', '  I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)',
    'LH(mIU/mL)', 'FSH/LH', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)',
    'PRG(ng/mL)', 'RBS(mg/dl)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)',
    'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)',
    'Endometrium (mm)', 'Unnamed: 44'
]
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# Clean target column
df['PCOS (Y/N)'] = df['PCOS (Y/N)'].astype(str).str.strip().replace({'0': 0, '1': 1}).astype(int)
df = df.dropna(subset=['PCOS (Y/N)'])

# Fill missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)
    mode_series = df[col].mode()
    df[col] = df[col].fillna(mode_series[0] if not mode_series.empty else 'Unknown')
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=['PCOS (Y/N)'])
y = df['PCOS (Y/N)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Classifiers to compare ---
models = {
    "XGBoost (App)": joblib.load("models/pcos_xgb_model.pkl"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

# --- Evaluate models ---
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 Results for {name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc_score:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    results[name] = {
        'model': model,
        'accuracy': acc,
        'auc': auc_score,
        'conf_matrix': cm,
        'y_proba': y_proba,
        'y_pred': y_pred
    }

# --- Plot ROC Curves ---
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - PCOS")
plt.legend()
plt.tight_layout()
plt.savefig("offline_comparisons/roc_curve_pcos.png")
plt.close()

# --- Plot Accuracy & AUC Bar Chart ---
plt.figure(figsize=(8, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
aucs = [results[name]['auc'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35
plt.bar(x - width/2, accuracies, width, label='Accuracy')
plt.bar(x + width/2, aucs, width, label='ROC AUC')
plt.xticks(x, model_names, rotation=15)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison - PCOS")
plt.legend()
plt.tight_layout()
plt.savefig("offline_comparisons/performance_comparison_pcos.png")
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
    plt.savefig(f"offline_comparisons/conf_matrix_{name.replace(' ', '_')}_pcos.png")
    plt.close()
