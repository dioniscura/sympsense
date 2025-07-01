def main():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib

    # Load model and features
    model = joblib.load("models/pcos_xgb_model.pkl")
    features = joblib.load("models/pcos_feature_names.pkl")

    st.set_page_config(page_title="PCOS Risk Detector", page_icon="🌼")
    st.title("🌼 PCOS Risk Detector")
    st.write("Estimate the likelihood of Polycystic Ovary Syndrome (PCOS) based on basic symptoms and indicators.")

    st.header("👩 Basic and Lifestyle Information")

    # Define UI based on cleaned feature list
    input_data = {}

    for feature in features:
        clean_feat = feature.strip().lower()

        if "age" in clean_feat:
            input_data[feature] = st.slider("Age (years)", 13, 50, 25)
        elif "weight" in clean_feat and "gain" not in clean_feat:
            input_data[feature] = st.slider("Weight (kg)", 30, 120, 60)
        elif "height" in clean_feat:
            input_data[feature] = st.slider("Height (cm)", 130, 190, 160)
        elif "bmi" in clean_feat:
            input_data[feature] = st.number_input("BMI (kg/m²)", value=22.0)
        elif "cycle length" in clean_feat:
            input_data[feature] = st.slider("Cycle length (days)", 21, 45, 28)
        elif "pregnant" in clean_feat:
            input_data[feature] = 1 if st.radio("Are you currently pregnant?", ["Yes", "No"], horizontal=True) == "Yes" else 0
        elif "aborptions" in clean_feat:
            input_data[feature] = st.selectbox("Number of past abortions (if any)", list(range(0, 6)))
        elif "weight gain" in clean_feat:
            input_data[feature] = 1 if st.radio("Unusual weight gain?", ["Yes", "No"], horizontal=True) == "Yes" else 0
        elif "hair growth" in clean_feat:
            input_data[feature] = 1 if st.radio("Excessive hair growth (hirsutism)?", ["Yes", "No"], horizontal=True) == "Yes" else 0
        elif "skin darkening" in clean_feat:
            input_data[feature] = 1 if st.radio("Skin darkening in neck/armpit area?", ["Yes", "No"], horizontal=True) == "Yes" else 0
        elif "hair loss" in clean_feat:
            input_data[feature] = 1 if st.radio("Hair thinning or loss?", ["Yes", "No"], horizontal=True) == "Yes" else 0
        elif "pimples" in clean_feat:
            input_data[feature] = 1 if st.radio("Frequent pimples/acne?", ["Yes", "No"], horizontal=True) == "Yes" else 0
        elif "fast food" in clean_feat:
            input_data[feature] = 1 if st.radio("Do you often eat fast food?", ["Yes", "No"], horizontal=True) == "Yes" else 0
        elif "exercise" in clean_feat:
            input_data[feature] = 1 if st.radio("Do you exercise regularly?", ["Yes", "No"], horizontal=True) == "Yes" else 0
        else:
            input_data[feature] = st.number_input(f"{feature} (numeric)", value=0.0)

    # Predict button
    if st.button("Check PCOS Risk"):
        input_df = pd.DataFrame([input_data])

        try:
            probs = model.predict_proba(input_df)[0]
            risk_prob = probs[1] * 100  # Probability of PCOS class

            # Define risk categories based on probability thresholds
            if risk_prob < 40:
                st.success(f"✅ Low risk of PCOS ({risk_prob:.1f}% confidence)")
            elif risk_prob < 70:
                st.warning(f"⚠️ Moderate risk of PCOS ({risk_prob:.1f}% confidence)")
                st.info("Some signs suggest PCOS, but further clinical evaluation and testing by a specialist are needed to confirm the diagnosis.")
            else:
                st.error(f"❗ High risk of PCOS ({risk_prob:.1f}% confidence)")
                st.info("Please consult a gynecologist or endocrinologist for further testing.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
