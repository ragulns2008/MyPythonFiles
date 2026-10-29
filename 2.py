import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Step 1: Create sample dataset
data = {
'Name': ['A', 'B', 'C', 'D', 'E', 'D'],
'Age': [20, 21, None, 23, 21, 23],
'Salary': [25000, 30000, 28000, None, 30000, None],
'Department': ['HR', 'IT', 'HR', 'Finance', 'IT', 'Finance']
}
df = pd.DataFrame(data)
print("Original Data:")
print(df)
# Step 2: Handling Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
# Step 3: Removing Duplicates
df.drop_duplicates(inplace=True)
print("\nOriginal Data after removing duplicates")
print(df)
# Step 4: Encoding Categorical Data
df = pd.get_dummies(df, columns=['Department'])
# Step 5: Feature Scaling (Normalization)
scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])
print("\nPreprocessed Data:")
print(df)
