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
    "I was aware of dryness of my mouth.",
    "I experienced breathing difficulty.",
    "I experienced trembling (e.g., in the hands).",
    "I was worried about situations in which I might panic and make a fool of myself.",
    "I felt I was close to panic.",
    "I was aware of the action of my heart in the absence of physical exertion.",
    "I felt scared without any good reason."
]

# Generate the dataset
data = []
for _ in range(1000):
    # Randomly assign responses to each question
    responses = [random.choice(options) for _ in questions]

    # Define the target as "Anxiety" or "Not Anxiety" based on a simple logic
    # Example: If 3 or more responses are "Applied to me a considerable degree" or "Applied to me very much", mark as Anxiety
    anxiety_count = responses.count("Applied to me a considerable degree") + responses.count("Applied to me very much")
    target = "Anxiety" if anxiety_count >= 3 else "Not Anxiety"

    # Append the responses and target to the dataset
    data.append(responses + [target])

# Create a DataFrame
columns = questions + ["Target"]
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
df.to_csv("anxiety_dataset.csv", index=False)

print("Dataset created successfully! File saved as 'anxiety_dataset.csv'.")
