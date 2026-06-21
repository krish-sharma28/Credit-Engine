import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


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
