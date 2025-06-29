from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(
    df_scaled[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
df_scaled.to_csv("cleaned_iris.csv", index=False)
def predict_species(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    pred = model.predict(input_scaled)
    return pred[0]
df['species'].value_counts().plot(kind='bar', title='Species Count')
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()
joblib.dump(tree_model, "iris_decision_tree.pkl")
joblib.dump(scaler, "iris_scaler.pkl")
