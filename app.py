import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="SME Digital Payment Adoption Predictor",
    page_icon="🇰🇪",
    layout="wide"
)

st.title("🇰🇪 SME Digital Payment Adoption Predictor")
st.markdown("""
**Built for Absa Bank Kenya & Airtel Money Kenya**  
This tool predicts which SMEs are most likely to adopt 
digital paybill payments via the Absa-Airtel partnership.
""")

st.divider()

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 2000

    counties = [
        "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
        "Kiambu", "Machakos", "Meru", "Nyeri", "Kakamega",
        "Kisii", "Kilifi", "Garissa", "Kitui", "Bungoma",
        "Uasin Gishu", "Muranga", "Embu", "Kericho", "Migori"
    ]
    county_weights = [
        0.20, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05,
        0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03,
        0.03, 0.03, 0.03, 0.02, 0.02, 0.02
    ]
    county_weights = [w / sum(county_weights) for w in county_weights]
    sectors = ["Retail", "Food & Beverage", "Transport", "Wholesale",
               "Salon/Barbershop", "Electronics", "Clothing", "Hardware"]

    df = pd.DataFrame({
        "county": np.random.choice(counties, n, p=county_weights),
        "sector": np.random.choice(sectors, n),
        "business_age_years": np.random.randint(1, 20, n),
        "monthly_revenue_ksh": np.random.choice(
            [5000, 15000, 35000, 75000, 150000, 300000],
            n, p=[0.15, 0.25, 0.30, 0.18, 0.08, 0.04]
        ),
        "uses_mpesa": np.random.choice([0, 1], n, p=[0.25, 0.75]),
        "has_bank_account": np.random.choice([0, 1], n, p=[0.40, 0.60]),
        "mobile_money_freq_per_month": np.random.randint(0, 60, n),
        "owns_smartphone": np.random.choice([0, 1], n, p=[0.35, 0.65]),
        "km_to_nearest_bank": np.random.exponential(scale=8, size=n).clip(0.5, 80),
        "num_employees": np.random.choice([1, 2, 3, 5, 10], n, p=[0.40, 0.25, 0.20, 0.10, 0.05]),
        "urban": np.random.choice([0, 1], n, p=[0.38, 0.62]),
    })

    adoption_score = (
        0.30 * (df["mobile_money_freq_per_month"] / 60) +
        0.20 * df["owns_smartphone"] +
        0.15 * df["has_bank_account"] +
        0.15 * df["urban"] +
        0.10 * (df["monthly_revenue_ksh"] / 300000) +
        0.05 * df["uses_mpesa"] +
        0.05 * (1 - df["km_to_nearest_bank"] / 80)
    )
    noise = np.random.normal(0, 0.08, n)
    df["will_adopt"] = ((adoption_score + noise).clip(0, 1) > 0.45).astype(int)

    le_county = LabelEncoder()
    le_sector = LabelEncoder()
    df["county_enc"] = le_county.fit_transform(df["county"])
    df["sector_enc"] = le_sector.fit_transform(df["sector"])

    features = ["county_enc", "sector_enc", "business_age_years",
                "monthly_revenue_ksh", "uses_mpesa", "has_bank_account",
                "mobile_money_freq_per_month", "owns_smartphone",
                "km_to_nearest_bank", "num_employees", "urban"]

    X = df[features]
    y = df["will_adopt"]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    return rf, le_county, le_sector, df, features

rf, le_county, le_sector, df, features = train_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Enter SME Details")

    county = st.selectbox("County", sorted([
        "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
        "Kiambu", "Machakos", "Meru", "Nyeri", "Kakamega",
        "Kisii", "Kilifi", "Garissa", "Kitui", "Bungoma",
        "Uasin Gishu", "Muranga", "Embu", "Kericho", "Migori"
    ]))
    sector = st.selectbox("Business Sector", [
        "Retail", "Food & Beverage", "Transport", "Wholesale",
        "Salon/Barbershop", "Electronics", "Clothing", "Hardware"
    ])
    business_age = st.slider("Business Age (years)", 1, 20, 5)
    revenue = st.selectbox("Monthly Revenue (KSH)",
        [5000, 15000, 35000, 75000, 150000, 300000], index=2)
    uses_mpesa = st.radio("Does the SME use M-Pesa?", ["Yes", "No"])
    has_bank = st.radio("Does the SME have a bank account?", ["Yes", "No"])
    mm_freq = st.slider("Mobile Money Transactions per Month", 0, 60, 15)
    smartphone = st.radio("Does the owner own a smartphone?", ["Yes", "No"])
    km_bank = st.slider("Distance to Nearest Bank (km)", 1, 80, 10)
    employees = st.selectbox("Number of Employees", [1, 2, 3, 5, 10])
    urban = st.radio("Location Type", ["Urban", "Rural"])
    predict_btn = st.button("🔍 Predict Adoption Likelihood", type="primary")

with col2:
    st.subheader("🎯 Prediction Result")

    if predict_btn:
        county_enc = le_county.transform([county])[0]
        sector_enc = le_sector.transform([sector])[0]

        input_data = pd.DataFrame([{
            "county_enc": county_enc,
            "sector_enc": sector_enc,
            "business_age_years": business_age,
            "monthly_revenue_ksh": revenue,
            "uses_mpesa": 1 if uses_mpesa == "Yes" else 0,
            "has_bank_account": 1 if has_bank == "Yes" else 0,
            "mobile_money_freq_per_month": mm_freq,
            "owns_smartphone": 1 if smartphone == "Yes" else 0,
            "km_to_nearest_bank": km_bank,
            "num_employees": employees,
            "urban": 1 if urban == "Urban" else 0
        }])

        prob = rf.predict_proba(input_data)[0][1]
        prediction = "Will Adopt" if prob >= 0.5 else "Won't Adopt"

        st.metric(label="Adoption Probability", value=f"{prob:.0%}")

        if prob >= 0.7:
            st.success(f"✅ {prediction} — High priority SME. Target immediately.")
        elif prob >= 0.5:
            st.warning(f"⚠️ {prediction} — Medium priority. Nurture with incentives.")
        else:
            st.error(f"❌ {prediction} — Low priority. Focus resources elsewhere.")

        st.progress(float(prob))

        st.divider()
        st.markdown("#### 📊 What Drives This Prediction?")

        feat_imp = pd.Series(
            rf.feature_importances_,
            index=["County", "Sector", "Business Age", "Revenue",
                   "Uses M-Pesa", "Has Bank Account", "MM Frequency",
                   "Owns Smartphone", "Distance to Bank", "Employees", "Urban"]
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        feat_imp.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info("👈 Fill in the SME details on the left and click Predict to see results.")

st.divider()
st.subheader("📍 Adoption Rate by County")
st.markdown("Which counties have the highest SME digital payment adoption potential?")

county_adoption = df.groupby("county")["will_adopt"].mean().sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(12, 4))
county_adoption.plot(kind="bar", ax=ax2, color="steelblue")
ax2.set_title("Predicted Adoption Rate by County")
ax2.set_ylabel("Adoption Rate")
ax2.set_xlabel("County")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig2)

st.caption("Built by Ashley Nyaboke Kibwogo | SME Digital Payment Adoption Predictor | Absa & Airtel Kenya")
