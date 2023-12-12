import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv('/content/winequality-red.csv') ;
data

X= data.drop('quality', axis=1);
Y= data['quality'] ;

# Scaling of Data
scaler = StandardScaler() ;
X_scale= scaler.fit_transform(X) ;

# Split X and Y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size= 0.20, random_state=0) ;


pca = PCA(n_components= 11) ;
X_train_pca= pca.fit_transform(X_train) ;
X_test_pca= pca.transform(X_test) ;

model = LogisticRegression(random_state=0)
model.fit(X_train_pca, Y_train)

Y_pred = model.predict(X_test_pca)

accuracy = accuracy_score(Y_test, Y_pred)
confusion_mat = confusion_matrix(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion_mat}')
print(f'Classification Report:\n{classification_rep}')


# Read the file csv
# Extract Features <X, Y> (Input, Output)
# Apply Scaling on the features to make the features on the same level
# To Train test sets (80% and 20%)
# Apply PCA(Principal Component Analysis)
# Use Logistic Regression
# Evaluate the Model
