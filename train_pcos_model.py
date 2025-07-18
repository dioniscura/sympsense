import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap

# --- Step 1: Load Data ---
file_path = 'data/PCOS_data_without_infertility.xlsx'
df = pd.read_excel(file_path, sheet_name='Full_new')

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

df['PCOS (Y/N)'] = df['PCOS (Y/N)'].astype(str).str.strip().replace({'0': 0, '1': 1}).astype(int)
df.dropna(subset=['PCOS (Y/N)'], inplace=True)

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    mode_series = df[col].mode()
    df[col] = df[col].fillna(mode_series[0] if not mode_series.empty else 'Unknown')
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(columns=['PCOS (Y/N)'])
y = df['PCOS (Y/N)']

# --- Step 1b: Class Imbalance Visualization ---
plt.figure(figsize=(5, 4))
sns.countplot(x=y, palette='Set2')
plt.title("PCOS Class Distribution")
plt.xlabel("Class (0 = No PCOS, 1 = PCOS)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- Step 2: Train XGBoost ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

xgb = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='recall',
    cv=3,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

train_sample = X_train.copy()
train_sample['PCOS (Y/N)'] = y_train.values
train_sample.head(10).to_csv("annex/pcos_train_sample.csv", index=False)

test_sample = X_test.copy()
test_sample['PCOS (Y/N)'] = y_test.values
test_sample.head(10).to_csv("annex/pcos_test_sample.csv", index=False)

# --- Step 3: Evaluate Model ---
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["No PCOS", "PCOS"], yticklabels=["No PCOS", "PCOS"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - PCOS Detection")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title("ROC Curve - PCOS Detection")
plt.show()

# --- Step 4: Feature Importance ---
plt.figure(figsize=(10, 8))
plot_importance(best_model, max_num_features=15, importance_type='gain')
plt.title("Feature Importance (Gain) - PCOS Detection")
plt.tight_layout()
plt.show()

importance_dict = best_model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.to_csv('models/pcos_feature_importance.csv', index=False)

# --- Step 5: PCA ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y.values

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Target', palette='Set1', alpha=0.7)
plt.title("PCA - PCOS Dataset Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Step 6: SHAP Analysis ---
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_train)

plt.figure()
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - PCOS")
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values, X_train)

# --- Step 7: Save Model and Artifacts ---
joblib.dump(best_model, 'models/pcos_xgb_model.pkl')
joblib.dump(list(X.columns), 'models/pcos_feature_names.pkl')
print("✅ Model and feature names saved successfully.")
