import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Startup Success Predictor", layout="centered")
st.title("Startup Success Chances")
st.write("Will your startup get acquired or shut down? Let’s find out.")

st.header("Enter Startup Details")

funding_total_usd = st.number_input("Total Funding (USD)", min_value=0, step=10000)
funding_rounds      = st.number_input("Number of Funding Rounds", min_value=0)
milestones          = st.number_input("Milestones Achieved", min_value=0)
relationships       = st.number_input("Number of Founder/Team Relationships", min_value=0)
avg_participants    = st.number_input("Avg. Investors per Round", min_value=0.0, step=0.1)

category = st.selectbox("Startup Category", [
    "software", "ecommerce", "biotech", "gamesvideo",
    "mobile", "consulting", "advertising", "web", "enterprise"
])
location = st.selectbox("Primary Location", [
    "California", "New York", "Massachusetts", "Texas", "Other"
])

has_VC     = st.checkbox("Has Venture Capital Funding?")
has_angel  = st.checkbox("Has Angel Investor?")
has_roundA = st.checkbox("Has Round A?")
has_roundB = st.checkbox("Has Round B?")
has_roundC = st.checkbox("Has Round C?")
has_roundD = st.checkbox("Has Round D?")

category_map = {
    "software":   17, "ecommerce": 7,  "biotech": 0,  "gamesvideo": 5,
    "mobile":     10, "consulting": 3, "advertising": 2, "web":        18,
    "enterprise": 6
}
category_code = category_map[category]

is_CA         = 1 if location == "California"     else 0
is_NY         = 1 if location == "New York"       else 0
is_MA         = 1 if location == "Massachusetts"  else 0
is_TX         = 1 if location == "Texas"          else 0
is_otherstate = 1 if location == "Other"          else 0

input_data = np.array([[
    funding_total_usd, funding_rounds, milestones, relationships, avg_participants,
    has_VC, has_angel, has_roundA, has_roundB, has_roundC, has_roundD,
    is_CA, is_NY, is_MA, is_TX, is_otherstate,
    1 if category == "software" else 0,
    1 if category == "web"      else 0,
    1 if category == "mobile"   else 0,
    1 if category == "enterprise" else 0,
    1 if category == "ecommerce"  else 0,
    category_code
]])

try:
    model  = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Missing rf_model.pkl or scaler.pkl in this folder.")
    st.stop()

input_scaled = scaler.transform(input_data)
pred = model.predict(input_scaled)[0]
conf = model.predict_proba(input_scaled)[0][pred]

st.header("Prediction Result")
if pred == 1:
    st.success(f"Likely to be Acquired (confidence: {conf:.1%})")
else:
    st.warning(f"Likely to be Closed   (confidence: {conf:.1%})")

st.markdown(
    "—\n"
    "[GitHub Repository](https://github.com/kushu30/startupsuccess) • "
    "Built by Kushagra Shukla • "
    "[LinkedIn](https://linkedin.com/in/kushu30) • "
    "[GitHub](https://github.com/kushu30) • "
    "[Portfolio](https://kushu.vercel.app)"
)

