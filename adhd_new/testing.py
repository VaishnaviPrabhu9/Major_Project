import pandas as pd
import joblib  # To load the trained model

# Load the trained model from the file
model_filename = 'adhd_impulsivity_model.pkl'
loaded_model = joblib.load(model_filename)

# Load the CSV file for testing
file_path = r'C:\Users\job01\Downloads\ManoShakti-main\ManoShakti-main\balanced_adhd_assessment_data.csv'  # Adjust path to your test CSV file
df = pd.read_csv(file_path)

# Separate features (Q1 to Q18) and target (Assuming 'Target' is the label column)
X_test = df.iloc[:, :-1]  # All columns except the last one (questions 1 to 18)
y_test = df['Target']  # Assuming 'Target' is the column to predict

# Encode the target variable if it's categorical (e.g., 'High Impulsivity/Restlessness')
y_test = pd.Categorical(y_test).codes

# Make predictions on the entire test set
y_pred = loaded_model.predict(X_test)

# Evaluate the model's performance on the test set
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Print the classification report to evaluate the model's precision, recall, and F1-score
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example: Testing a single sample from the test set
sample_data = X_test.iloc[0].values.reshape(1, -1)  # Example using the first test sample
predicted_class = loaded_model.predict(sample_data)

print(f"Predicted Class for Sample 0: {predicted_class}")

# You can also access the original target class (e.g., 'High Impulsivity/Restlessness') for the sample
original_class = y_test[0]
print(f"Original Class for Sample 0: {original_class}")
