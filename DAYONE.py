import pandas as pd
df = pd.read_csv("iris.csv")
print("Dataset Preview:")
print(df.head())
print("\nRows with Null Values:")
print(df[df.isnull().any(axis=1)])
print("\nColumns with Null Values:")
print(df.isnull().sum())
