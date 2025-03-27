import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model and target encoder
model = joblib.load('anxiety_model.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Load the test dataset (replace with the actual file path)
df = pd.read_csv(r'C:\Users\VAISHNAVI\OneDrive\Desktop\Projects\Major_Project\dass_new\anxiety\anxiety_dataset.csv')

# Define answer encoding
answer_mapping = {
    "Did not apply to me at all": 0,
    "Applied to me to some degree": 1,
    "Applied to me a considerable degree": 2,
    "Applied to me very much": 3
}

# Apply encoding to the features (Q1 to Q7)
for col in df.columns[:-1]:  # Exclude the target column
    df[col] = df[col].map(answer_mapping)

# Separate features and target
X_test = df.drop(columns=['Target'])  # Features (Q1 to Q7)
y_test = target_encoder.transform(df['Target'])  # Encode target

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
