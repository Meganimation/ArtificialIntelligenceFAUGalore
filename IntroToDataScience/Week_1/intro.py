import pandas as pd

# Create a simple dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
    'Salary': [50000, 60000, 75000, 55000]
}

# Raw Data
print("Raw Data:", data)

# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print("DataFrame:")
print(df)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Filter data
print("\nPeople older than 28:")
print(df[df['Age'] > 28])

# Sort by salary
print("\nSorted by Salary:")
print(df.sort_values('Salary', ascending=False))

# Calculate average salary
print(f"\nAverage Salary: ${df['Salary'].mean():,.2f}")
