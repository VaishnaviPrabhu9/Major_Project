import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib  # To save and load the model

# Load the CSV file
file_path = 'C:/Users/VAISHNAVI/OneDrive/Desktop/agri_last1/adhd_new/balanced_adhd_assessment_data.csv'  # Adjust path
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(df.head())

# Separate features (Q1 to Q18) and target (Assuming 'Target' is the label column)
X = df.iloc[:, :-1]  # All columns except the last one (questions 1 to 18)
y = df['Target']  # Assuming 'Target' is the column to predict

# Encode the target variable using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Save the model to a file using joblib
model_filename = 'adhd_impulsivity_model.pkl'
joblib.dump(rf_classifier, model_filename)
print(f"Model saved as {model_filename}")

# Save the label encoder as a .pkl file
label_encoder_filename = 'label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_filename)
print(f"Label encoder saved as {label_encoder_filename}")

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Load the model and label encoder (for later predictions)
loaded_model = joblib.load(model_filename)
loaded_label_encoder = joblib.load(label_encoder_filename)

# Example: Making predictions with the loaded model and label encoder
sample_data = X_test.iloc[0].values.reshape(1, -1)  # Example using the first test sample
predicted_class_encoded = loaded_model.predict(sample_data)
predicted_class = loaded_label_encoder.inverse_transform(predicted_class_encoded)  # Convert back to original labels
print(f"Predicted Class: {predicted_class}")
