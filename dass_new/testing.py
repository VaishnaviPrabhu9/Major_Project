import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the saved model and label encoder
model_filename = 'stress_model.pkl'
encoder_filename = 'label_encoder.pkl'

model = joblib.load(model_filename)
label_encoder = joblib.load(encoder_filename)

# Load the dataset for testing (replace with the actual file path)
test_file_path = 'stress_test_dataset.csv'  # Replace with your test dataset
df_test = pd.read_csv(test_file_path)

# Define features and target
X_test = df_test.iloc[:, :-1]  # All columns except the last one (features)
y_test = df_test.iloc[:, -1]   # The last column (target)

# Encode the target labels (if necessary)
y_test_encoded = label_encoder.transform(y_test)

# Make predictions
y_pred_encoded = model.predict(X_test)

# Decode predictions back to original labels
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Evaluate model performance
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred_encoded, target_names=label_encoder.classes_))

# Optional: Compare predictions with actual values
df_test['Predicted'] = y_pred
print("Comparison of Actual vs Predicted:")
print(df_test[['target', 'Predicted']].head())
