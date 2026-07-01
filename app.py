import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go


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
    
    st.subheader("Map your columns")
    st.write("Match your CSV columns to the fields the model needs.")

    model_fields = [
        'age', 'monthly_income_inr', 'cibil_score', 'avg_bank_balance_inr',
        'gst_filing_regular', 'upi_txn_per_month', 'nach_bounces_12m',
        'existing_loans', 'employment_years', 'loan_amount_inr',
        'loan_tenure_months', 'loan_type', 'employment_type'
    ]

    user_cols = ['-- skip --'] + raw.columns.tolist()
    mapping = {}

    cols = st.columns(2)
    for i, field in enumerate(model_fields):
        default_idx = user_cols.index(field) if field in user_cols else 0
        selected = cols[i % 2].selectbox(f"{field}", user_cols, index=default_idx, key=field)
        if selected != '-- skip --':
            mapping[selected] = field

    if st.button("Run Scoring"):
        raw = raw.rename(columns=mapping)
        X = raw.drop(columns=['default', 'nbfc'], errors='ignore')
        X['cibil_available'] = (X['cibil_score'] != -1).astype(int)
        X['cibil_score'] = X['cibil_score'].replace(-1, np.nan)
        X['gst_available'] = (X['gst_filing_regular'] != -1).astype(int)
        X['gst_filing_regular'] = X['gst_filing_regular'].replace(-1, np.nan)

        X['loan_to_income_ratio'] = X['loan_amount_inr'] / X['monthly_income_inr']
        X['emi_to_income_ratio'] = (X['loan_amount_inr'] / X['loan_tenure_months']) / X['monthly_income_inr']
        X['balance_to_loan_ratio'] = X['avg_bank_balance_inr'] / X['loan_amount_inr']
        X = X.drop(columns=['avg_bank_balance_inr'], errors='ignore')

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
        top = contributions[:5]
        names = [c[0] for c in top]
        vals = [c[1] for c in top]
        colors = ['red' if v > 0 else 'green' for v in vals]

        fig = go.Figure(go.Bar(
            x=vals,
            y=names,
            orientation='h',
            marker_color=colors
        ))
        fig.update_layout(
            title="Top factors behind this score",
            xaxis_title="Impact on risk",
            height=300
        )
        st.plotly_chart(fig)

        top_feature = contributions[0][0]
        top_val = contributions[0][1]
        direction_word = "high risk" if top_val > 0 else "low risk"
        st.info(f"The biggest factor in this decision was **{top_feature}**, \
                which signals **{direction_word}** for this borrower.")