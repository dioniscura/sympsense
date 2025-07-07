import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Load and prepare dataset ---
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

X = df.drop("ASD_Traits", axis=1)
y = df["ASD_Traits"]
categorical = ['Ethnicity']

# --- Class Imbalance Visualization ---
class_counts = y.value_counts().sort_index()
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='pastel')
plt.title("Class Distribution - ASD Traits")
plt.xlabel("Class (0 = No, 1 = Yes)")
plt.ylabel("Number of Samples")
plt.xticks([0, 1])
plt.tight_layout()
plt.show()

# --- Preprocessing and pipeline ---
onehot = OneHotEncoder(handle_unknown="ignore")
preprocessor = ColumnTransformer([
    ("onehot", onehot, categorical)
], remainder="passthrough")

clf = RandomForestClassifier(n_estimators=100, random_state=42)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", clf)
])

# --- Train/test split and training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# --- Metrics ---
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"✅ Model trained with accuracy: {accuracy:.2%}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Confusion Matrix:")
print(cm)
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")

# --- Confusion Matrix Heatmap ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["No ASD", "ASD"], yticklabels=["No ASD", "ASD"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Autism Detection")
plt.tight_layout()
plt.show()

# --- Save model and feature names ---
Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/autism_pipeline.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

# --- PCA Plot ---
X_scaled = StandardScaler().fit_transform(preprocessor.fit_transform(X))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for label, color in zip([0, 1], ['red', 'blue']):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=f'Target {label}', alpha=0.6, c=color)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Autism Dataset Projection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Feature Importance ---
encoded_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical)
other_features = [col for col in X.columns if col not in categorical]
final_feature_names = list(encoded_feature_names) + other_features

importances = clf.feature_importances_
importances_df = pd.DataFrame({
    'Feature': final_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Label mapping for interpretability
label_map = {
    "A1": "Eye contact",
    "A2": "Social smile",
    "A3": "Response to name",
    "A4": "Imitation",
    "A5": "Shared enjoyment",
    "A6": "Points to interest",
    "A7": "Pretend play",
    "A8": "Finger movements",
    "A9": "Interest in peers",
    "A10": "Emotion response"
}

plot_df = importances_df.copy()
plot_df['Feature'] = plot_df['Feature'].replace(label_map)

print("\nTop 10 Features by Importance:")
print(plot_df.head(10))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(plot_df['Feature'][:15][::-1], plot_df['Importance'][:15][::-1], color='skyblue')
plt.xlabel("Feature Importance (Gini)")
plt.title("Top 15 Important Features - Autism")
plt.tight_layout()
plt.show()

# Save importance CSV
importances_df.to_csv("models/autism_feature_importance.csv", index=False)

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Autism Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
