import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Loading and preparing data
df = pd.read_csv("data/Toddler Autism dataset July 2018.csv")
df.rename(columns={"Class/ASD Traits ": "ASD_Traits"}, inplace=True)
df.columns = df.columns.str.replace('"', '').str.strip().str.replace(' ', '_')
df.drop(columns=["Case_No", "Who_completed_the_test", "Qchat-10-Score"], inplace=True)
df["ASD_Traits"] = df["ASD_Traits"].map({"Yes": 1, "No": 0})

# Normalizing string fields (only object columns)
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.lower())

# Explicit binary encoding
df['Sex'] = df['Sex'].map({'m': 1, 'f': 0})
df['Jaundice'] = df['Jaundice'].map({'yes': 1, 'no': 0})
df['Family_mem_with_ASD'] = df['Family_mem_with_ASD'].map({'yes': 1, 'no': 0})

# Preparing features and target
X = df.drop("ASD_Traits", axis=1)
y = df["ASD_Traits"]
categorical = ['Ethnicity']  # Only ethnicity needs one-hot encoding

# Pipeline
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Training and evaluating
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy:.2%}")

# Saving model
Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/autism_pipeline.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
