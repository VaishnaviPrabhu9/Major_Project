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
    "I couldn’t seem to experience any positive feeling at all",
    "I found it difficult to work up the initiative to do things",
    "I felt that I had nothing to look forward to",
    "I felt down-hearted and blue",
    "I was unable to become enthusiastic about anything",
    "I felt I wasn’t worth much as a person",
    "I felt that life was meaningless"
]

# Generate the dataset
data = []
for _ in range(1000):
    # Randomly assign responses to each question
    responses = [random.choice(options) for _ in questions]

    # Define the target as "Depressed" or "Not Depressed" based on a simple logic
    # Example: If 3 or more responses are "Applied to me a considerable degree" or "Applied to me very much", mark as Depressed
    depression_count = responses.count("Applied to me a considerable degree") + responses.count("Applied to me very much")
    depressed = "Depressed" if depression_count >= 3 else "Not Depressed"

    # Append the responses and target to the dataset
    data.append(responses + [depressed])

# Create a DataFrame
columns = questions + ["Target"]
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
df.to_csv("depression_dataset.csv", index=False)

print("Dataset created successfully! File saved as 'depression_dataset.csv'.")
