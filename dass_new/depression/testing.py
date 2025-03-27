import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = r'C:\Users\VAISHNAVI\OneDrive\Desktop\Projects\Major_Project\dass_new\depression\depression_dataset.csv'
df = pd.read_csv(file_path)

# Define answer mapping for Q1 to Q7 encoding (same as in training)
answer_mapping = {
    "Did not apply to me at all": 0,
    "Applied to me to some degree": 1,
    "Applied to me a considerable degree": 2,
    "Applied to me very much": 3
}

# Apply encoding to the questions (Q1 to Q7)
for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']:
    df[col] = df[col].map(answer_mapping)

# Load the trained model and label encoder
model = joblib.load('depression_model.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Prepare test data (features and labels)
X_test = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']]
y_test = target_encoder.transform(df['Target'])

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Generate a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
