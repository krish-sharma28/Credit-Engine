import pandas as pd

df = pd.read_csv('data/nbfc_credit_data.csv')

y = df['default']
X = df.drop(columns=['default', 'nbfc'])

X = pd.get_dummies(X, columns=['loan_type', 'employment_type'])
print(X.columns.tolist())
print(X.shape)
