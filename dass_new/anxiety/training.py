import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
file_path = 'D:/project_10/dass_new/anxiety/anxiety_dataset.csv'  # Replace with the path to your dataset
df = pd.read_csv(file_path)

# Create LabelEncoder for responses and target
response_encoder = LabelEncoder()
target_encoder = LabelEncoder()

# Encode the responses for each question (assuming your columns are questions)
for col in df.columns[:-1]:  # Excluding the 'Target' column
    df[col] = response_encoder.fit_transform(df[col])

# Encode the target column (for example: 'Anxious' / 'Not Anxious')
df['Target'] = target_encoder.fit_transform(df['Target'])

# Separate features (X) and target (y)
X = df.drop(columns=['Target'])
y = df['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Output model accuracy
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model and label encoders for future use
joblib.dump(model, 'anxiety_model.pkl')
joblib.dump(response_encoder, 'response_encoder.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

# Provide file paths for downloading (if needed)


print("Model and encoders saved successfully!")
