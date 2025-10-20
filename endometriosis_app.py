def main():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib

    # Load trained model and preprocessing tools
    model = joblib.load("models/endometriosis_svm_model.pkl")
    scaler = joblib.load("models/endometriosis_scaler.pkl")
    selector = joblib.load("models/endometriosis_selector.pkl")
    features = joblib.load("models/endometriosis_features.pkl")

    st.set_page_config(page_title="Endometriosis Risk Detector", page_icon="🌸", layout="wide")
    st.title("🌸 Endometriosis Risk Detector")
    st.write("Estimate your **risk of Endometriosis** based on symptoms and medical history.")

    st.header("👩 Basic and Lifestyle Information")

    # Collect input data
    age = st.slider("Age (years)", 18, 50, 25)
    weight = st.slider("Weight (kg)", 30, 120, 60)
    height = st.slider("Height (cm)", 130, 190, 160)
    bmi = round(weight / ((height / 100) ** 2), 2)
    st.markdown(f"**BMI (kg/m²):** `{bmi}`")

    st.header("🩺 Symptom & Medical History")

    menstrual_irregularity = 1 if st.radio("Do you experience menstrual irregularity?", ["Yes", "No"], horizontal=True) == "Yes" else 0
    infertility = 1 if st.radio("Have you been diagnosed with or experienced infertility?", ["Yes", "No"], horizontal=True) == "Yes" else 0
    chronic_pain_level = st.slider("On a scale from 0 to 10, how severe is your chronic pelvic pain?", 0, 10, 5)
    hormone_level_abnormality = 1 if st.radio("Have you been told or suspect a hormonal level abnormality?", ["Yes", "No"], horizontal=True) == "Yes" else 0
    family_history = 1 if st.radio("Do you have a family history of endometriosis?", ["Yes", "No"], horizontal=True) == "Yes" else 0
    painful_intercourse = 1 if st.radio("Do you experience pain during or after sexual intercourse (dyspareunia)?", ["Yes", "No"], horizontal=True) == "Yes" else 0

    if st.button("Check Endometriosis Risk"):
        # Manually construct the input DataFrame with engineered features
        df = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "menstrual_irregularity": menstrual_irregularity,
            "infertility": infertility,
            "chronic_pain_level": chronic_pain_level,
            "hormone_level_abnormality": hormone_level_abnormality,
            "family_history": family_history,
            "painful_intercourse": painful_intercourse
        }])

        # Feature engineering
        df['age_binned'] = pd.cut(df['age'], bins=[17, 25, 35, 50], labels=[0, 1, 2]).astype(int)
        df['bmi_binned'] = pd.cut(df['bmi'], bins=[14, 18.5, 25, 30, 40], labels=[0, 1, 2, 3]).astype(int)
        df['pain_binned'] = pd.cut(df['chronic_pain_level'], bins=[-1, 3, 7, 10], labels=[0, 1, 2]).astype(int)
        df['irregular_pain_combo'] = df['menstrual_irregularity'] * df['chronic_pain_level']
        df['infertility_hormone_interact'] = df['infertility'] * df['hormone_level_abnormality']
        df['pain_irregularity_ratio'] = df['chronic_pain_level'] / (df['menstrual_irregularity'] + 1)
        df['age_times_bmi'] = df['age'] * df['bmi']

        df.drop(columns=['age', 'bmi', 'chronic_pain_level'], inplace=True)

        # Ensure all features present and ordered.
        for col in features:
            if col not in df:
                df[col] = 0.0
        df = df[features]

        try:
            scaled = scaler.transform(df)
            selected = selector.transform(scaled)
            probs = model.predict_proba(selected)[0]
            risk_prob = probs[1] * 100  # Probability of positive class

            if risk_prob < 40:
                st.success(f"✅ Low risk of Endometriosis ({risk_prob:.1f}% confidence)")
            elif risk_prob < 70:
                st.warning(f"⚠️ Moderate risk of Endometriosis ({risk_prob:.1f}% confidence)")
                st.info("Some signs suggest endometriosis, but further clinical evaluation is recommended.")
            else:
                st.error(f"❗ High risk of Endometriosis ({risk_prob:.1f}% confidence)")
                st.info("Please consult a gynecologist for testing and diagnosis.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")


if __name__ == "__main__":
    main()
