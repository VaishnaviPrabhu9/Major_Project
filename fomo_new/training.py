import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
file_path = 'fomo_dataset.csv'  # Update this path if needed
df = pd.read_csv(file_path)

# Define features and target
X = df.iloc[:, :-1]  # All columns except the last one (features)
y = df.iloc[:, -1]   # The last column (target)

# Encode the target column if it contains categorical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and label encoder
model_filename = 'fomo_model.pkl'
encoder_filename = 'label_encoder.pkl'

joblib.dump(model, model_filename)
joblib.dump(label_encoder, encoder_filename)

print(f"Model saved as {model_filename}")
print(f"Label encoder saved as {encoder_filename}")
