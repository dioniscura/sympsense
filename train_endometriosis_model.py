import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
import seaborn as sns

# --- Load and preprocess data ---
df = pd.read_csv("data/structured_endometriosis_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

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

# --- Class imbalance visualization ---
class_counts = y.value_counts().sort_index()
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='pastel')
plt.title("Class Distribution - Endometriosis Diagnosis")
plt.xlabel("Diagnosis (0 = No, 1 = Yes)")
plt.ylabel("Number of Samples")
plt.xticks([0, 1])
plt.tight_layout()
plt.show()

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Scale and select features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(score_func=f_classif, k=15)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)

# --- GridSearchCV for SVM ---
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

svm = SVC(probability=True, class_weight='balanced', random_state=42)

grid_search = GridSearchCV(
    svm,
    param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_sel, y_train)
best_model = grid_search.best_estimator_

# --- Evaluate ---
y_pred = best_model.predict(X_test_sel)
y_proba = best_model.predict_proba(X_test_sel)[:, 1]

print("\n✅ Grid search complete.")
print(f"Best Params: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion matrix heatmap ---
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Endometriosis")
plt.tight_layout()
plt.show()

# --- ROC Curve ---
RocCurveDisplay.from_estimator(best_model, X_test_sel, y_test)
plt.title("ROC Curve - Endometriosis")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Save artifacts ---
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/endometriosis_svm_model.pkl")
joblib.dump(scaler, "models/endometriosis_scaler.pkl")
joblib.dump(selector, "models/endometriosis_selector.pkl")
joblib.dump(list(X.columns), "models/endometriosis_features.pkl")
print("✅ Model and preprocessing artifacts saved.")

# --- PCA Visualization ---
X_scaled_full = scaler.transform(X)
X_selected_full = selector.transform(X_scaled_full)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_selected_full)

plt.figure(figsize=(8, 6))
for label, color in zip([0, 1], ['red', 'blue']):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=f'Target {label}', alpha=0.5, c=color)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Endometriosis Dataset Projection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Feature Importance (ANOVA F-scores) ---
mask = selector.get_support()
selected_features = X.columns[mask]
f_scores = selector.scores_[mask]

importance_df = pd.DataFrame({
    'Feature': selected_features,
    'F-score': f_scores
}).sort_values(by='F-score', ascending=False)

print("\nTop Important Features by F-score:")
print(importance_df.head(10))

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:15][::-1], importance_df['F-score'][:15][::-1], color='skyblue')
plt.xlabel("ANOVA F-score")
plt.title("Top Important Features - Endometriosis")
plt.tight_layout()
plt.show()

# Save feature importance
importance_df.to_csv("models/endometriosis_feature_importance.csv", index=False)
