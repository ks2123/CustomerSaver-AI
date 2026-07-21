"""
CustomerSaver AI - churn prediction app

Loads a trained XGBoost model + scaler and predicts whether a customer
is likely to churn based on some basic account info.

To run:
    streamlit run app.py

Needs these files to exist (from the training script):
    data/processed/model.pkl
    data/processed/scaler.pkl
    data/processed/feature_names.pkl
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="CustomerSaver AI", layout="wide")

MODEL_PATH = Path("data/processed/model.pkl")
SCALER_PATH = Path("data/processed/scaler.pkl")
FEATURES_PATH = Path("data/processed/feature_names.pkl")

RISK_CUTOFF = 50  # % - above this we call it "high risk"

GENDERS = ["Male", "Female", "Other"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


# a little bit of CSS just so the metric numbers don't look like default
# streamlit text. nothing fancy.
st.markdown("""
<style>
.metric-box {
    border: 1px solid #444;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: center;
}
.metric-box .label {
    font-size: 0.75rem;
    color: #888;
}
.metric-box .value {
    font-size: 1.3rem;
    font-weight: 600;
}
.result-high {
    border: 1px solid #a33;
    background-color: rgba(170, 51, 51, 0.12);
    border-radius: 6px;
    padding: 16px 18px;
}
.result-low {
    border: 1px solid #2a7;
    background-color: rgba(34, 170, 119, 0.10);
    border-radius: 6px;
    padding: 16px 18px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load model/scaler/feature list. Returns (model, scaler, features, error_msg)."""
    missing = [str(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not p.exists()]
    if missing:
        return None, None, None, "Missing file(s): " + ", ".join(missing)

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
    except Exception as e:
        return None, None, None, f"Couldn't load artifacts - {e}"

    return model, scaler, features, None


def run_prediction(input_dict, model, scaler, feature_names):
    """
    Takes the raw form inputs, does the same preprocessing the model was
    trained on (one-hot encode -> reindex to match training columns ->
    scale), then returns (predicted_class, churn_probability).
    """
    df = pd.DataFrame([input_dict])

    # one-hot encode categorical columns the same way training did
    df = pd.get_dummies(df)

    # line up columns with what the model actually expects - anything
    # missing (e.g. a category that wasn't in this particular row) gets
    # filled with 0
    df = df.reindex(columns=feature_names, fill_value=0)

    scaled = scaler.transform(df)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]  # probability of class "churn"

    return pred, prob, df


# ---------------------------------------------------------------------
# load everything up front so we can bail early if something's missing
# ---------------------------------------------------------------------
model, scaler, feature_names, err = load_artifacts()

st.title("CustomerSaver AI")
st.caption("Churn risk predictor - XGBoost model")

if err:
    st.error(err)
    st.write(
        "Make sure you've run the training script first and that the "
        "`data/processed/` folder has model.pkl, scaler.pkl, and "
        "feature_names.pkl in it."
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------
# input form (sidebar keeps the main area clean)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Customer info")

    gender = st.selectbox("Gender", GENDERS)
    age = st.slider("Age", 18, 100, 35)

    contract = st.selectbox("Contract type", CONTRACTS)
    payment_method = st.selectbox("Payment method", PAYMENT_METHODS)
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    monthly_charges = st.number_input("Monthly charges ($)", 10.0, 150.0, 65.0, step=1.0)

    # total charges = tenure * monthly, but let the user override it
    # (sometimes a customer's actual total doesn't match the clean math,
    # e.g. mid-cycle plan changes)
    default_total = round(tenure * monthly_charges, 2)
    total_charges = st.number_input("Total charges ($)", 0.0, value=default_total, step=1.0)

    predict_btn = st.button("Predict", type="primary", use_container_width=True)


# ---------------------------------------------------------------------
# quick summary of what was entered
# ---------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip(
    [c1, c2, c3, c4],
    ["Age", "Tenure", "Monthly", "Total"],
    [f"{age}", f"{tenure} mo", f"${monthly_charges:,.2f}", f"${total_charges:,.2f}"],
):
    col.markdown(
        f'<div class="metric-box"><div class="label">{label}</div>'
        f'<div class="value">{val}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------
if predict_btn:
    raw_input = {
        "Age": age,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Gender": gender,
        "Contract": contract,
        "PaymentMethod": payment_method,
    }

    try:
        pred, prob, model_input = run_prediction(raw_input, model, scaler, feature_names)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Debug - raw input was:", raw_input)
        st.stop()

    risk_pct = round(prob * 100, 1)

    if risk_pct > RISK_CUTOFF:
        st.markdown(
            f"""
            <div class="result-high">
            <h4>High churn risk - {risk_pct}%</h4>
            This customer is likely to leave soon based on their contract
            and billing pattern.<br><br>
            <b>Suggested action:</b> reach out with a 20% retention discount
            and try to move them onto a longer contract (they're currently
            on {contract.lower()}).
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-low">
            <h4>Low churn risk - {risk_pct}%</h4>
            Account looks stable, no red flags in the current profile.<br><br>
            <b>Suggested action:</b> nothing urgent - maybe a quick
            check-in email in a few months to keep engagement up.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("See model input (after encoding)"):
        st.dataframe(model_input)

else:
    st.write("Fill in the details on the left and hit Predict.")