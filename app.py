import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Custom CSS for a clean, professional look
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #f7f9fc; }

        .block-container { padding-top: 2rem; padding-bottom: 3rem; }

        h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #1f2937; }

        .app-header {
            background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(30, 58, 138, 0.25);
        }
        .app-header h1 { color: white; margin-bottom: 0.25rem; }
        .app-header p { color: #dbeafe; font-size: 1.05rem; margin: 0; }

        .metric-card {
            background: white;
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border: 1px solid #e5e7eb;
        }

        .result-card {
            border-radius: 16px;
            padding: 1.75rem 2rem;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
            margin-top: 1rem;
        }
        .result-stay { background: #ecfdf5; border: 1px solid #10b981; }
        .result-churn { background: #fef2f2; border: 1px solid #ef4444; }

        .stButton>button {
            background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 1rem;
            width: 100%;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
            transform: translateY(-1px);
        }

        section[data-testid="stSidebar"] {
            background-color: #111827;
        }
        section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

FEATURES = [
    "credit_score", "country", "gender", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member", "estimated_salary",
]

COUNTRY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Female": 0, "Male": 1}

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📊 Customer Churn Predictor</h1>
        <p>Estimate the likelihood that a bank customer will churn, powered by a Random Forest model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar — About / Info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About this app")
    st.write(
        "This tool uses a trained **Random Forest Classifier** to predict whether "
        "a bank customer is likely to churn based on their profile and account activity."
    )
    st.markdown("### 🧠 Model details")
    st.write(f"- Algorithm: `RandomForestClassifier`")
    st.write(f"- Trees: `{model.n_estimators}`")
    st.write(f"- Features used: `{len(FEATURES)}`")
    st.markdown("---")
    st.caption(
        "⚠️ Country and gender are encoded as France=0, Germany=1, Spain=2 and "
        "Female=0, Male=1. Adjust the mapping in the code if your training "
        "encoding was different."
    )

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
st.markdown("### Enter Customer Details")

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        age = st.slider("Age", 18, 100, 35)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)

    with col2:
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=500.0)
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=500.0)
        products_number = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)

    with col3:
        country = st.selectbox("Country", list(COUNTRY_MAP.keys()))
        gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
        credit_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        active_member = st.radio("Active Member?", ["Yes", "No"], horizontal=True)

    submitted = st.form_submit_button("🔍 Predict Churn")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    input_dict = {
        "credit_score": credit_score,
        "country": COUNTRY_MAP[country],
        "gender": GENDER_MAP[gender],
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": 1 if credit_card == "Yes" else 0,
        "active_member": 1 if active_member == "Yes" else 0,
        "estimated_salary": estimated_salary,
    }
    input_df = pd.DataFrame([input_dict], columns=FEATURES)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # probability of class 1 (churn)

    st.markdown("### Prediction Result")

    result_col, gauge_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-card result-churn">
                    <h2>⚠️ Likely to Churn</h2>
                    <p style="font-size:1.1rem;">This customer shows a high risk of leaving the bank.</p>
                    <h1 style="color:#ef4444;">{probability*100:.1f}%</h1>
                    <p>Estimated churn probability</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-stay">
                    <h2>✅ Likely to Stay</h2>
                    <p style="font-size:1.1rem;">This customer shows a low risk of leaving the bank.</p>
                    <h1 style="color:#10b981;">{probability*100:.1f}%</h1>
                    <p>Estimated churn probability</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with gauge_col:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Churn Risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 33], "color": "#dcfce7"},
                        {"range": [33, 66], "color": "#fef9c3"},
                        {"range": [66, 100], "color": "#fee2e2"},
                    ],
                },
            )
        )
        fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 View input data sent to the model"):
        st.dataframe(input_df, use_container_width=True)

else:
    st.info("Fill in the customer details above and click **Predict Churn** to see the result.")

st.markdown("---")
st.caption("Built with Streamlit · Random Forest model · For demonstration purposes only.")
