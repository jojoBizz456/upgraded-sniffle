import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the red wine quality dataset
data = pd.read_csv('winequality_red.csv')

# Separate features (X) and labels (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear SVM
linear_svm = SVC(kernel='linear', C=1.0, random_state=42)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

# Non-linear SVM with RBF kernel
rbf_svm = SVC(kernel='rbf', C=1.0, random_state=42)
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

# Polynomial SVM
poly_svm = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
poly_svm.fit(X_train, y_train)
y_pred_poly = poly_svm.predict(X_test)

# Evaluate Linear SVM
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print("Linear SVM:")
print(f"Accuracy: {accuracy_linear:.2f}")
print(classification_report(y_test, y_pred_linear))

# Evaluate Non-linear SVM with RBF kernel
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print("\nNon-linear SVM with RBF Kernel:")
print(f"Accuracy: {accuracy_rbf:.2f}")
print(classification_report(y_test, y_pred_rbf))

# Evaluate Polynomial SVM
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print("\nPolynomial SVM:")
print(f"Accuracy: {accuracy_poly:.2f}")
print(classification_report(y_test,y_pred_poly))
