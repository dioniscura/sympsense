import streamlit as st
import autism_app
import pcos_app
import endometriosis_app
import sympsense_assistant

st.set_page_config(page_title="SympSense App", page_icon="🩺", layout="wide")

# CSS Styling
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 4rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        #MainMenu, footer {visibility: hidden;}
        header {visibility: hidden;}  /* Completely hide default Streamlit header */

        .stApp > div:first-child {
            margin-top: -50px;
            padding-top: 0;
        }

        /* Top Navbar */
        .top-navbar {
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: white;
            padding: 1.2rem 3rem 0.2rem 3rem;
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            margin-bottom: 0;
        }

        .top-navbar .stButton > button {
            background: white;
            border: 1.8px solid #ddd;
            padding: 10px 24px;
            border-radius: 25px;
            font-size: 15px;
            font-weight: 500;
            color: #333;
            transition: 0.3s ease;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        }

        .top-navbar .stButton > button:hover {
            background: #e91e63 !important;
            color: white !important;
            border-color: #e91e63 !important;
            box-shadow: 0 4px 16px rgba(233, 30, 99, 0.2);
        }

        /* Divider */
        .navbar-divider {
            margin-top: 0.2rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid #eee;
        }

        /* Card Styling */
        .card {
            background-color: #fdfdfd;
            border: 1px solid #eee;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
            margin-top: 2rem;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        .card-icon {
            font-size: 40px;
            margin-bottom: 10px;
        }
        .card-title {
            font-weight: 600;
            margin-bottom: 10px;
            font-size: 18px;
        }
        .card-desc {
            font-size: 14px;
            color: #666;
        }

        .disclaimer {
            margin-top: 4rem;
            color: #666;
            font-style: italic;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Session state
if "nav_mode" not in st.session_state:
    st.session_state.nav_mode = "🏠 Home"

# Top Navbar
st.markdown('<div class="top-navbar">', unsafe_allow_html=True)
cols = st.columns([1, 1, 1, 1, 1])
with cols[0]:
    if st.button("🏠 Home"):
        st.session_state.nav_mode = "🏠 Home"
        st.rerun()
with cols[1]:
    if st.button("👶 Autism"):
        st.session_state.nav_mode = "👶 Autism Risk Detector"
        st.rerun()
with cols[2]:
    if st.button("🌼 PCOS"):
        st.session_state.nav_mode = "🌼 PCOS Risk Detector"
        st.rerun()
with cols[3]:
    if st.button("🌸 Endometriosis"):
        st.session_state.nav_mode = "🌸 Endometriosis Risk Detector"
        st.rerun()
with cols[4]:
    if st.button("🪄 Assistant"):
        st.session_state.nav_mode = "🪄 SympSense Assistant"
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Divider
st.markdown('<div class="navbar-divider"></div>', unsafe_allow_html=True)

# Pages
app_mode = st.session_state.nav_mode

if app_mode == "🏠 Home":
    st.markdown("""
        <h1 style="text-align: center; margin-top: 2rem;">👋 <strong>Welcome to <span style='color:#e91e63;'>SympSense</span></strong></h1>
        <p style="text-align: center; font-size: 18px;">
            SympSense is a smart health web app designed to help raise awareness around <strong>syndrome-related risks</strong><br>
            through early symptom screening and supportive guidance for women and families.
        </p>
    """, unsafe_allow_html=True)

    cols = st.columns(4)

    with cols[0]:
        st.markdown("""
            <div class="card">
                <div class="card-icon">👶</div>
                <div class="card-title">Autism Risk Detector</div>
                <div class="card-desc">For toddlers showing early behavioral signs.</div>
            </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown("""
            <div class="card">
                <div class="card-icon">🌼</div>
                <div class="card-title">PCOS Risk Detector</div>
                <div class="card-desc">For women with symptoms related to PCOS.</div>
            </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown("""
            <div class="card">
                <div class="card-icon">🌸</div>
                <div class="card-title">Endometriosis Risk Detector</div>
                <div class="card-desc">For chronic pain and menstrual irregularities.</div>
            </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        st.markdown("""
            <div class="card">
                <div class="card-icon">🪄</div>
                <div class="card-title">SympSense Assistant</div>
                <div class="card-desc">AI assistant for symptom-related advice.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="disclaimer">
            ⚠️ This is not a diagnostic tool. Please consult a medical professional for any concerns.
        </div>
    """, unsafe_allow_html=True)

elif app_mode == "👶 Autism Risk Detector":
    autism_app.main()
elif app_mode == "🌼 PCOS Risk Detector":
    pcos_app.main()
elif app_mode == "🌸 Endometriosis Risk Detector":
    endometriosis_app.main()
elif app_mode == "🪄 SympSense Assistant":
    sympsense_assistant.main()
