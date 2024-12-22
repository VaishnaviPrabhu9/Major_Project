import pandas as pd
import random

# Define the options for responses
options = [
    "Did not apply to me at all",
    "Applied to me to some degree",
    "Applied to me a considerable degree",
    "Applied to me very much"
]

# Define the questions
questions = [
    "I found it hard to wind down.",
    "I tended to over-react to situations.",
    "I felt that I was using a lot of nervous energy.",
    "I found myself getting agitated.",
    "I found it difficult to relax.",
    "I was intolerant of anything that kept me from getting on with what I was doing.",
    "I felt that I was rather touchy."
]

# Generate the dataset
data = []
for _ in range(1000):
    # Randomly assign responses to each question
    responses = [random.choice(options) for _ in questions]

    # Define the target as "Stressed" or "Not Stressed" based on a simple logic
    # Example: If 3 or more responses are "Applied to me a considerable degree" or "Applied to me very much", mark as Stressed
    stress_count = responses.count("Applied to me a considerable degree") + responses.count("Applied to me very much")
    target = "Stressed" if stress_count >= 3 else "Not Stressed"

    # Append the responses and target to the dataset
    data.append(responses + [target])

# Create a DataFrame
columns = questions + ["Target"]
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
df.to_csv("stress_dataset.csv", index=False)

print("Dataset created successfully! File saved as 'stress_dataset.csv'.")
