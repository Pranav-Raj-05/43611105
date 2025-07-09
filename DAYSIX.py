import pandas as pd
import json
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
boston = load_boston()
df_sklearn = pd.DataFrame(boston.data, columns=boston.feature_names)
df_sklearn['MEDV'] = boston.target
df_sklearn.columns = df_sklearn.columns.str.strip().str.upper()
df_kaggle = pd.read_csv("BostonHousing.csv")  # Ensure this file is in the same folder
df_kaggle.columns = df_kaggle.columns.str.strip().str.upper()
diff_columns = set(df_kaggle.columns).symmetric_difference(set(df_sklearn.columns))
print("🧾 Column differences:", diff_columns)
shape_diff = {
    "kaggle_shape": df_kaggle.shape,
    "sklearn_shape": df_sklearn.shape
}
print("📐 Shape differences:", shape_diff)
common_cols = list(set(df_kaggle.columns).intersection(set(df_sklearn.columns)))
stat_kaggle = df_kaggle[common_cols].describe().round(2)
stat_sklearn = df_sklearn[common_cols].describe().round(2)
stat_diff = (stat_kaggle - stat_sklearn).dropna(how='all')
print("📊 Summary Statistics Differences:\n", stat_diff)
differences = {
    "column_diff": list(diff_columns),
    "shape_diff": shape_diff,
    "stat_diff": stat_diff.to_dict()
}
with open("boston_dataset_differences.json", "w") as f:
    json.dump(differences, f, indent=4)
print("✅ Differences saved to 'boston_dataset_differences.json'")
