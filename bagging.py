import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

uploaded_file_path = 'diabetes.csv'
df = pd.read_csv(uploaded_file_path)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_tree = DecisionTreeClassifier(random_state=16)
bagging_model = BaggingClassifier(base_tree, n_estimators=3, random_state=16)

bagging_model.fit(X_train, y_train)

y_pred_bagging = bagging_model.predict(X_test)
acc_bagging = accuracy_score(y_test, y_pred_bagging)
report_bagging = classification_report(y_test, y_pred_bagging)

print("Bagging with 3 Decision Trees - Accuracy:", acc_bagging)
print("Bagging with 3 Decision Trees - Classification Report:\n", report_bagging)
