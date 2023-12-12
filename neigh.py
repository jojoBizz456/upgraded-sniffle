import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

uploaded_file_path = 'diabetes.csv'
df = pd.read_csv(uploaded_file_path)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test).round()
y_pred_knn = knn_model.predict(X_test)
y_pred_nb = nb_model.predict(X_test)

ensemble_predictions = (y_pred_lr + y_pred_knn + y_pred_nb) >= 2

acc_ensemble = accuracy_score(y_test, ensemble_predictions)
report_ensemble = classification_report(y_test, ensemble_predictions)

print("Ensemble (Linear Regression, KNN, Naive Bayes) - Accuracy:", acc_ensemble)
print("Ensemble (Linear Regression, KNN, Naive Bayes) - Classification Report:\n", report_ensemble)
