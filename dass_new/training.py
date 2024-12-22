import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset from the CSV file
file_path = 'stress_dataset.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Check column names
print("Columns in dataset:", df.columns)

# Define features and target directly
X = df.iloc[:, :-1]  # All columns except the last one (assumed to be the target)
y = df.iloc[:, -1]   # The last column (assumed to be the target)

# Encode the target column if it contains categorical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file using joblib
model_filename = 'stress_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# (Optional) Save the LabelEncoder
encoder_filename = 'label_encoder.pkl'
joblib.dump(label_encoder, encoder_filename)
print(f"Label encoder saved as {encoder_filename}")
