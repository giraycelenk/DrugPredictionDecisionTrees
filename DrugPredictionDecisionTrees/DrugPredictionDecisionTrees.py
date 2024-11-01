# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("drug200.csv")

label_encoders = {}
for column in ['Sex', 'BP', 'Cholesterol']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
    
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42,max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

new_patient = pd.DataFrame([[30, label_encoders['Sex'].transform(['F'])[0], label_encoders['BP'].transform(['HIGH'])[0], label_encoders['Cholesterol'].transform(['HIGH'])[0], 15.5]],
                           columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])

predicted_drug = model.predict(new_patient)
print("Predicted medicine of the new patient:", predicted_drug[0])

