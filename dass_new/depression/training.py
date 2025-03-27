import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('C:\Users\VAISHNAVI\OneDrive\Desktop\Projects\Major_Project\dass_new\depression\depression_dataset.csv')

# Create LabelEncoder for responses and target
response_encoder = LabelEncoder()
target_encoder = LabelEncoder()

# Encode the responses for each question
for col in df.columns[:-1]:  # Excluding the 'Target' column
    df[col] = response_encoder.fit_transform(df[col])

# Encode the target column (Depressed/Not Depressed)
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

print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model and label encoders for future use
import joblib
joblib.dump(model, 'depression_model.pkl')
joblib.dump(response_encoder, 'response_encoder.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')
