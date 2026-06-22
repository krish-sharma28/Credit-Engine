import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('features.pkl')
explainer = joblib.load('explainer.pkl')

st.title("Credit Engine")

def decision(score):
    if score >= 750:
        return "Approve"
    elif score >= 600:
        return "Review"
    else:
        return "Reject"

uploaded = st.file_uploader("Upload borrower data (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded is not None:
    if uploaded.name.endswith('.csv'):
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_excel(uploaded)

    st.subheader("Uploaded data")
    st.dataframe(raw)

    X = raw.drop(columns=['default', 'nbfc'], errors='ignore')
    X = pd.get_dummies(X, columns=['loan_type', 'employment_type'])

    X = X.reindex(columns=feature_names, fill_value=0)

    X_scaled = scaler.transform(X)

    default_proba = model.predict_proba(X_scaled)[:, 1]
    scores = ((1 - default_proba) * 1000).astype(int)
    decisions = [decision(s) for s in scores]

    results = pd.DataFrame({
        'NBFC': raw['nbfc'],
        'Loan Type': raw['loan_type'],
        'Loan Amount': raw['loan_amount_inr'],
        'Score': scores,
        'Decision': decisions
    })

    st.subheader("Results")

    approved = (results['Decision'] == 'Approve').sum()
    review = (results['Decision'] == 'Review').sum()
    rejected = (results['Decision'] == 'Reject').sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(results))
    c2.metric("Approved", approved)
    c3.metric("Review", review)
    c4.metric("Rejected", rejected)

    def color_decision(val):
        colors = {'Approve': 'Green', 'Review': "Yellow", 'Reject': 'Red'}
        return f'background-color: {colors.get(val, "")}; color: white'

    styled = results.style.map(color_decision, subset=['Decision'])
    st.dataframe(styled)
    
    st.subheader("Why did a borrower get their score?")
    idx = st.selectbox("Pick a borrower (row number)", results.index)

    shap_values = explainer.shap_values(X_scaled)
    borrower_shap = shap_values[idx]

    contributions = sorted(
        zip(feature_names, borrower_shap),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    st.write(f"**Score: {scores[idx]} → {decisions[idx]}**")
    st.write("Top factors behind this decision:")
    for name, val in contributions[:5]:
        direction = "increased risk" if val > 0 else "lowered risk"
        st.write(f"- {name}: {direction} ({val:+.2f})")