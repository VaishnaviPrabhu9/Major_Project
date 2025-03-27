import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset (replace with your actual CSV file path)
df = pd.read_csv('C:\Users\VAISHNAVI\OneDrive\Desktop\Projects\Major_Project\dass_new\anxiety\anxiety_dataset.csv')

# Preview the data
print("Dataset Preview:")
print(df.head())

# Step 2: Map the answers to numerical values
answer_mapping = {
    "Did not apply to me at all": 0,
    "Applied to me to some degree": 1,
    "Applied to me a considerable degree": 2,
    "Applied to me very much": 3
}

# Apply the answer encoding to each question (Q1 to Q7)
for col in df.columns[:-1]:  # Exclude the target column
    df[col] = df[col].map(answer_mapping)

# Preview the encoded data
print("\nEncoded Data Preview:")
print(df.head())

# Step 3: Separate features (Q1 to Q7) and target (Anxiety/Not Anxiety)
X = df.drop(columns=['Target'])  # Features (Q1 to Q7)
y = df['Target']  # Target (Anxiety or Not Anxiety)

# Step 4: Encode the target variable (Anxiety vs. Not Anxiety)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Save the target encoder for later use
joblib.dump(target_encoder, 'target_encoder.pkl')

# Step 5: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 6: Train the model (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Step 9: Save the trained model
joblib.dump(model, 'anxiety_model.pkl')

# Optionally, you can print out the feature importance for understanding
print("\nFeature Importances:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance}")
