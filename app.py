
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NestEstimate | House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"
DATA_PATH = BASE_DIR / "data" / "cleaned_house_prices.csv"

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #172C28;
    --green: #52796F;
    --sage: #DCE8DF;
    --cream: #F7F5EF;
    --muted: #68766F;
    --white: #FFFFFF;
}

.stApp {
    background: #F7F5EF;
    color: #172C28;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #172C28 !important;
    letter-spacing: -0.8px;
}

p, label, span {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Header */
.brand {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1.45rem;
    color: #172C28;
    letter-spacing: -1px;
}

.brand-icon {
    background: #172C28;
    color: white;
    border-radius: 12px;
    padding: 7px 10px;
    margin-right: 9px;
}

.eyebrow {
    color: #52796F;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero {
    background: linear-gradient(120deg, #E7EEE5 0%, #F7F5EF 75%);
    border: 1px solid #E1E7DD;
    border-radius: 26px;
    padding: 38px 40px;
    margin: 24px 0 28px 0;
}

.hero h1 {
    font-size: clamp(2.1rem, 4vw, 3.45rem);
    line-height: 1.12;
    margin: 12px 0;
    font-weight: 800;
}

.hero p {
    font-size: 1.04rem;
    line-height: 1.8;
    color: #68766F;
    max-width: 640px;
    margin-bottom: 0;
}

.tag {
    display: inline-block;
    background: #FFFFFF;
    border: 1px solid #DCE5DB;
    border-radius: 100px;
    padding: 7px 12px;
    font-size: 0.76rem;
    font-weight: 600;
    color: #52796F;
}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E8E9E1;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 8px 28px rgba(23, 44, 40, 0.035);
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: #172C28;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #68766F;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 12px !important;
    border-color: #E0E5DD !important;
    background: #FCFCF9 !important;
}

label[data-testid="stWidgetLabel"] p {
    color: #344B42 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

div[data-testid="stNumberInput"] button {
    color: #52796F !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: #172C28 !important;
    color: #FFFFFF !important;
    border: 1px solid #172C28 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-weight: 700 !important;
    min-height: 48px;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #31574C !important;
    border-color: #31574C !important;
    transform: translateY(-1px);
}

/* Result */
.result-card {
    background: linear-gradient(145deg, #172C28, #25483D);
    border-radius: 22px;
    padding: 28px;
    color: #FFFFFF;
    min-height: 250px;
    margin-bottom: 18px;
}

.result-label {
    font-size: 0.75rem;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #C2D7C8;
    font-weight: 700;
}

.result-price {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2.1rem, 4vw, 3rem);
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #FFFFFF;
    margin: 16px 0 8px;
    line-height: 1.15;
    overflow-wrap: anywhere;
}

.result-note {
    color: #D4E0D7;
    font-size: 0.88rem;
    line-height: 1.7;
}

.result-divider {
    border: 0;
    border-top: 1px solid rgba(255,255,255,0.17);
    margin: 22px 0;
}

.detail-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin: 12px 0;
    font-size: 0.88rem;
}

.detail-label {
    color: #C2D7C8;
}

.detail-value {
    color: #FFFFFF;
    font-weight: 600;
    text-align: right;
}

.info-card {
    background: #E7EEE5;
    border: 1px solid #D8E4D8;
    border-radius: 16px;
    padding: 17px;
    color: #344B42;
    line-height: 1.7;
    font-size: 0.87rem;
}

.stat-card {
    background: #FFFFFF;
    border: 1px solid #E8E9E1;
    border-radius: 16px;
    padding: 20px;
}

.stat-number {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1.55rem;
    color: #172C28;
}

.stat-label {
    color: #68766F;
    font-size: 0.83rem;
    margin-top: 4px;
}

.footer {
    border-top: 1px solid #E1E5DC;
    margin-top: 36px;
    padding-top: 20px;
    color: #68766F;
    font-size: 0.8rem;
    line-height: 1.8;
}

@media (max-width: 700px) {
    .hero {
        padding: 25px 22px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 15px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL & DATA ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_locations():
    df = pd.read_csv(DATA_PATH)
    return sorted(df["location"].dropna().unique().tolist())


try:
    model = load_model()
    locations = load_locations()
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.info(
        "Check that the models and data folders are inside "
        "the same project folder as app.py."
    )
    st.stop()


# ---------------- HEADER ----------------
header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<div class="brand"><span class="brand-icon">⌂</span>'
        'NestEstimate</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div style="text-align:right;padding-top:10px;">'
        '<span class="tag">✦ ML POWERED</span></div>',
        unsafe_allow_html=True
    )


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">BENGALURU PROPERTY INSIGHTS</div>
    <h1>Find your home's<br>estimated value.</h1>
    <p>
        A smarter starting point for your next move.
        Enter your property details to get an instant
        machine-learning estimate.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- INPUT AND RESULT ----------------
left, right = st.columns([1.08, 0.92], gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">Property details</div>'
            '<div class="section-subtitle">'
            'Tell us a little about the home you have in mind.'
            '</div>',
            unsafe_allow_html=True
        )

        with st.form("prediction_form"):
            location = st.selectbox(
                "📍 Property location",
                options=locations,
                index=(
                    locations.index("Electronic City Phase II")
                    if "Electronic City Phase II" in locations
                    else 0
                )
            )

            total_sqft = st.number_input(
                "Area (square feet)",
                min_value=300.0,
                max_value=30000.0,
                value=1200.0,
                step=50.0
            )

            col_bhk, col_bath = st.columns(2)

            with col_bhk:
                bhk = st.selectbox(
                    "Bedrooms",
                    options=list(range(1, 11)),
                    index=1,
                    format_func=lambda x: f"{x} BHK"
                )

            with col_bath:
                bath = st.selectbox(
                    "Bathrooms",
                    options=list(range(1, 11)),
                    index=1,
                    format_func=lambda x: f"{x} bath"
                )

            st.markdown("<div style='height:8px'></div>",
                        unsafe_allow_html=True)

            submitted = st.form_submit_button(
                "Calculate estimate  →",
                use_container_width=True
            )

with right:
    st.markdown(
        '<div class="section-title" style="margin-bottom:15px;">'
        'Your estimate</div>',
        unsafe_allow_html=True
    )

    if "predicted_price" not in st.session_state:
        st.markdown("""
        <div class="result-card">
            <div class="result-label">YOUR ESTIMATED VALUE</div>
            <div class="result-price">₹ —</div>
            <div class="result-note">
                Your personalized property estimate will appear here
                once you enter the details and calculate.
            </div>
            <hr class="result-divider">
            <div class="detail-row">
                <span class="detail-label">Location</span>
                <span class="detail-value">Bengaluru</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Prediction</span>
                <span class="detail-value">Ready when you are</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        price = st.session_state["predicted_price"]
        details = st.session_state["prediction_details"]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">YOUR ESTIMATED VALUE</div>
            <div class="result-price">₹ {price:,.2f} Lakh</div>
            <div class="result-note">
                Indicative machine-learning estimate · Bengaluru
            </div>
            <hr class="result-divider">
            <div class="detail-row">
                <span class="detail-label">Location</span>
                <span class="detail-value">{details["location"]}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Area</span>
                <span class="detail-value">{details["area"]:,.0f} sq ft</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Configuration</span>
                <span class="detail-value">{details["bhk"]} BHK · {details["bath"]} Bath</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <strong>✦ A helpful starting point</strong><br>
        This estimate is generated by a machine-learning model.
        Actual property prices can vary with condition, amenities,
        building age and market conditions.
    </div>
    """, unsafe_allow_html=True)


# ---------------- PROCESS FORM ----------------
# The submitted values are stored so the result stays visible
# when Streamlit reruns the app.
if submitted:
    if total_sqft / bhk < 300:
        st.warning(
            "This property has less than 300 sq ft per bedroom. "
            "Try adjusting the area or bedroom count."
        )
    else:
        input_data = pd.DataFrame([{
            "location": location,
            "total_sqft": total_sqft,
            "bhk": bhk,
            "bath": bath
        }])

        try:
            prediction = float(model.predict(input_data)[0])

            if prediction <= 0:
                st.session_state.pop("predicted_price", None)
                st.warning(
                    "The model returned an unrealistic estimate. "
                    "Try different property details."
                )
            else:
                st.session_state["predicted_price"] = prediction
                st.session_state["prediction_details"] = {
                    "location": location,
                    "area": total_sqft,
                    "bhk": bhk,
                    "bath": bath
                }
                st.rerun()

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ---------------- BOTTOM STATS ----------------
st.markdown("<div style='height:20px'></div>",
            unsafe_allow_html=True)

st.markdown('<div class="eyebrow">BUILT FOR A BETTER MOVE</div>',
            unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">12,319</div>
        <div class="stat-label">Cleaned property records</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">4</div>
        <div class="stat-label">Property input features</div>
    </div>
    """, unsafe_allow_html=True)

with s3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">Instant</div>
        <div class="stat-label">Model-generated estimate</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <strong>NestEstimate</strong> · MCA Machine Learning Project<br>
    Powered by Random Forest Regression. Estimates are for
    educational purposes and are not formal property valuations.
</div>
""", unsafe_allow_html=True)
