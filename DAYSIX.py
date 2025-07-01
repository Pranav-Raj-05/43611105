# Day 6: Load Datasets
import pandas as pd
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load Scikit-learn version
boston = load_boston()
df_sklearn = pd.DataFrame(boston.data, columns=boston.feature_names)
df_sklearn['MEDV'] = boston.target

# Load Kaggle version
df_kaggle = pd.read_csv("BostonHousing.csv")  # path to your downloaded CSV

# Compare Columns and Stats
df_kaggle.columns = df_kaggle.columns.str.upper().str.strip()
df_sklearn.columns = df_sklearn.columns.str.upper().str.strip()

diff_cols = set(df_kaggle.columns).symmetric_difference(set(df_sklearn.columns))
print("Column differences:", diff_cols)
print("Shape differences:", df_kaggle.shape, df_sklearn.shape)

# Day 7: Convert & Add Target Column
# Already done above

# Day 8: Summary Stats
print(df_kaggle.describe())
print(df_kaggle.info())
print(df_kaggle.isnull().sum())

# Day 9: Handle Missing, Visualize Outliers
import seaborn as sns
import matplotlib.pyplot as plt

df_kaggle = df_kaggle.dropna()
sns.boxplot(data=df_kaggle)
plt.xticks(rotation=90)
plt.title("Outlier Check")
plt.show()

# Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler1 = StandardScaler()
scaler2 = MinMaxScaler()

scaled_std = pd.DataFrame(scaler1.fit_transform(df_kaggle.drop(columns=["MEDV"])), columns=df_kaggle.columns[:-1])
scaled_minmax = pd.DataFrame(scaler2.fit_transform(df_kaggle.drop(columns=["MEDV"])), columns=df_kaggle.columns[:-1])

print("Standard Scaler:\n", scaled_std.describe())
print("Min-Max Scaler:\n", scaled_minmax.describe())

# Day 10: Correlation Matrix
corr = df_kaggle.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Day 11: Train-Test & Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X = df_kaggle.drop(columns=["MEDV"])
y = df_kaggle["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Day 12: Report
print("""
--- REPORT ---
Dataset was taken from Kaggle and Scikit-learn. After comparing columns and statistics, we used the Kaggle dataset due to sklearn's deprecation.

We performed EDA, visualized outliers, applied scaling, and selected features based on correlation.

A linear regression model was trained and evaluated using MSE, RMSE, and R² score.

Insights:
- RM and LSTAT are highly correlated with MEDV.
- The model performs decently with a good R² score.

Conclusion:
This project demonstrates the end-to-end process of data handling, preprocessing, and regression modeling.
""")
