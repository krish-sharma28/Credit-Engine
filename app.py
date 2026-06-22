import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import xgboost as xgb
import shap
import numpy as np



df = pd.read_csv('data/nbfc_credit_data.csv')

y = df['default']
X = df.drop(columns=['default', 'nbfc'])
X = pd.get_dummies(X, columns=['loan_type', 'employment_type'])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,
                                                    random_state=28, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))

xgb_model = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=2.6)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
print("XGBoost results:")
print(classification_report(y_test, xgb_pred))

default_proba = xgb_model.predict_proba(X_test)[:, 1]
scores = ((1 - default_proba) * 1000).astype(int)

print(scores[:5])

def decision(score):
    if score >= 750:
        return "Approve"
    elif score >= 600:
        return "Review"
    else:
        return "Reject"
    
for s in scores[:5]:
    print(s, "->", decision(s))

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

print(shap_values.shape)


feature_names = X.columns
borrower_shap = shap_values[0]

contributions = sorted(zip(feature_names, borrower_shap), key=lambda x: abs(x[1]), reverse=True)

print("Why borrower 0 got their score:")
for name, val in contributions[:5]:
    direction = "increased risk" if val > 0 else "lowered risk"
    print(f"  {name}: {direction} ({val:+.2f})")

