import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset from a CSV file
df = pd.read_csv('D:/project_10/dass_new/depression/depression_dataset.csv')
 # Update this with the path to your CSV file

# Print the first few rows to understand the data
print(df.head())

# Define answer mapping for Q1 to Q7 encoding
answer_mapping = {
    "Did not apply to me at all": 0,
    "Applied to me to some degree": 1,
    "Applied to me a considerable degree": 2,
    "Applied to me very much": 3
}

# Apply the encoding to the questions (Q1 to Q7)
for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']:
    df[col] = df[col].map(answer_mapping)

# Encode the target labels (Depressed / Not Depressed)
target_encoder = LabelEncoder()
df['Target'] = target_encoder.fit_transform(df['Target'])

# Separate features (Q1 to Q7) and target (Target)
X = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']]
y = df['Target']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (optional)
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model and label encoder to files
joblib.dump(model, 'depression_model.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

print("Model and target encoder saved successfully.")
