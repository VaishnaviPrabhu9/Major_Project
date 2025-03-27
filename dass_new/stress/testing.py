import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model and target encoder
model = joblib.load('stress_model.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Load the test dataset (replace with the actual file path)
df = pd.read_csv(r'C:\Users\VAISHNAVI\OneDrive\Desktop\Projects\Major_Project\dass_new\stress\stress_dataset.csv')

# Encode the input variables (Q1 to Q7) using the same encoding as training
encoder = {  # Ensure encoding is consistent
    "Did not apply to me at all": 0,
    "Applied to me to some degree": 1,
    "Applied to me a considerable degree": 2,
    "Applied to me very much": 3
}

for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']:
    df[col] = df[col].map(encoder)

# Separate features and target
X_test = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']]  # Features
y_test = target_encoder.transform(df['Target'])  # Encode target labels

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Test with a single sample (first row)
sample_data = X_test.iloc[0].values.reshape(1, -1)
predicted_class = model.predict(sample_data)
predicted_label = target_encoder.inverse_transform(predicted_class)

print(f"\nPredicted Class for Sample 0: {predicted_label[0]}")
print(f"Original Class for Sample 0: {df['Target'].iloc[0]}")  
