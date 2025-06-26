print("Summary Statistics:")
print(df.describe())
print("\nSample count per species:")
print(df['species'].value_counts())
filtered_df = df[df['petal_length'] > 1.5]
print("\nFiltered Rows (petal_length > 1.5):")
print(filtered_df)
df['species_encoded'] = df['species'].astype('category').cat.codes
df['petal_ratio'] = df['petal_length'] / df['petal_width']
print("\nData with Encoded Species and Petal Ratio:")
print(df.head())
