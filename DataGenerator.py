import pandas as pd

# Create sample data
data = {
    'text': ['I love cats', 'Dogs are loyal', 'Birds can fly', 'Fish live in water'],
    'label': [1, 1, 0, 0]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('data.csv', index=False)