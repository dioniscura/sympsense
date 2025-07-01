def main():

    import streamlit as st
    import pandas as pd
    import joblib

    st.set_page_config(page_title="Autism Risk Detector", page_icon="👶", layout="wide")

    st.title("👶 Autism Risk Detector")
    st.write("Check the likelihood of autism traits in toddlers based on behavioral indicators and basic information.")

    # Loading model
    model = joblib.load("models/autism_pipeline.pkl")

    # 👶 Basic Information
    st.header("🧒 Basic Information")
    age = st.slider("Age in Months", 12, 60, 36)
    sex = st.selectbox("Sex", ["M", "F"])
    ethnicity = st.selectbox("Ethnicity", [
        "White European", "Latino", "Others", "Black", "Asian",
        "Middle Eastern", "Pasifika", "South Asian", "Hispanic", "Turkish"
    ])
    jaundice = st.radio("Was the child born with jaundice?", ["Yes", "No"], horizontal=True, index=1)
    family_asd = st.radio("Is there a family member with ASD?", ["Yes", "No"], horizontal=True, index=1)

    # 🧳 Screening Questions in Two Columns
    st.header("📋 Toddler Screening Questionnaire")
    question_texts = {
        "A1": "Does your child look at you when you call his/her name?",
        "A2": "Does your child smile back when you smile at him/her?",
        "A3": "Does your child respond to your facial expressions?",
        "A4": "Does your child make eye contact while interacting with others?",
        "A5": "Does your child engage in pretend play (e.g., talking on a toy phone)?",
        "A6": "Does your child point to share interest (e.g., pointing at an airplane)?",
        "A7": "Does your child bring objects to show you?",
        "A8": "Does your child imitate actions like clapping or waving?",
        "A9": "Does your child show interest in other children?",
        "A10": "Does your child use simple gestures (like pointing)?"
    }

    col1, col2 = st.columns(2)
    screening = {}
    questions = list(question_texts.items())

    for i, (key, question) in enumerate(questions):
        with (col1 if i < 5 else col2):
            answer = st.radio(f"👶 {question}", ["Yes", "No"], horizontal=True, key=key)
            screening[key] = 0 if answer == "Yes" else 1

    # 🧠 Prediction
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="text-align:center">', unsafe_allow_html=True)
    if st.button("Check Autism Risk"):
        input_data = {
            **screening,
            "Age_Mons": age,
            "Sex": sex.lower(),
            "Ethnicity": ethnicity.lower().strip(),
            "Jaundice": jaundice.lower(),
            "Family_mem_with_ASD": family_asd.lower()
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        input_df['Sex'] = input_df['Sex'].map({'m': 1, 'f': 0})
        input_df['Jaundice'] = input_df['Jaundice'].map({'yes': 1, 'no': 0})
        input_df['Family_mem_with_ASD'] = input_df['Family_mem_with_ASD'].map({'yes': 1, 'no': 0})

        try:
            probs = model.predict_proba(input_df)[0]
            pred = probs.argmax()
            confidence = probs[pred] * 100

            if pred == 1:
                st.error(f"⚠️ High risk of autism traits ({confidence:.1f}% confidence)")
                st.info("Please consult a pediatrician or child psychologist.")
            else:
                st.success(f"✅ Low risk of autism traits ({confidence:.1f}% confidence)")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
