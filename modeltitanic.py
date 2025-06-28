# Re-run the Titanic ML project code after kernel reset

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import joblib

# Reload the uploaded files after kernel reset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 2: Fill missing values
# Step 2: Fill missing values safely
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Step 3: Convert categorical to numeric
for df in [train_df, test_df]:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Step 4: Drop unnecessary columns
drop_cols = ['Name', 'Ticket', 'Cabin']
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# Step 5: Prepare training data
X = train_df.drop(['Survived', 'PassengerId'], axis=1)
y = train_df['Survived']

# Step 6: Split data for evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

# Step 9: Predict on test set
X_test = test_df.drop(['PassengerId'], axis=1)
test_preds = model.predict(X_test)

# Step 10: Prepare submission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_preds
})

# Save submission to CSV
submission_path = "my_submission.csv"
submission.to_csv(submission_path, index=False)

# Output evaluation metrics and path to submission
acc, report, submission_path
print(acc)
print(report)



# After training
joblib.dump(model, "model.pkl")
