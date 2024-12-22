import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your CSV file (replace 'your_data.csv' with the actual file path)
df = pd.read_csv('D:/project_10/dass_new/stress/stress_dataset.csv')

# Display the first few rows of the dataset to confirm it's loaded
print(df.head())

# Encode the target variable (Stress) into numeric values
target_encoder = LabelEncoder()
df['Target'] = target_encoder.fit_transform(df['Target'])

# Encode the input variables (Q1 to Q7) into numeric values
encoder = LabelEncoder()
for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']:
    df[col] = encoder.fit_transform(df[col])

# Separate features (Q1 to Q7) and target (Target)
X = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']]  # Features
y = df['Target']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check the accuracy of the model
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the model and the target encoder
joblib.dump(model, 'stress_model.pkl')  # Save the trained model
joblib.dump(target_encoder, 'target_encoder.pkl')  # Save the target encoder
