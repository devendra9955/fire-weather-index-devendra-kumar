import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("Algerian_forest_fires_dataset.csv")
df.columns = df.columns.str.strip()
print(df.dtypes)

# Basic inspection
print("\nTop 10 rows:\n", df.head(10))
print("\nBottom 10 rows:\n", df.tail(10))
print("\nDescription:\n", df.describe())
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

# Duplicates
print("\nDuplicates before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates after:", df.duplicated().sum())

# Null values
print("\nNull mask:\n", df.isnull())

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)


numeric_cols = ['Temperature', 'RH', 'Ws', 'Rain',
                'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=numeric_cols, inplace=True)

print("\nNull values after cleaning:\n", df.isnull().sum())
print("\nNumeric columns used:", numeric_cols)

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Histograms
df[numeric_cols].hist(figsize=(12, 10), bins=15, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions")
plt.show()

# Boxplots
plt.figure(figsize=(12, 8))
df.boxplot(column=['Temperature', 'RH', 'Ws', 'Rain', 'FWI'])
plt.title("Boxplots for Numerical Features")
plt.show()


if 'Classes' in df.columns:
    df['Classes'] = df['Classes'].astype(str)

    sns.pairplot(
        df.sample(min(100, len(df))),
        vars=['Temperature', 'RH', 'Ws', 'Rain', 'FWI'],
        hue='Classes',
        diag_kind='hist'
    )
    plt.suptitle('Pairwise Feature Relationships', y=1.02)
    plt.show()

    avg_vals = df.groupby('Classes')[['Temperature', 'RH', 'Ws', 'Rain', 'FWI']].mean()
    avg_vals.plot(kind='bar', figsize=(10, 6))
    plt.title("Average Weather Metrics by Fire Class")
    plt.ylabel("Mean Values")
    plt.show()
else:
    print("\n'Classes' column not found. Skipping pairplot and class-wise averages.")

df.fillna(0, inplace=True)
print("\nNull values after final fill:\n", df.isnull().sum())

# Drop columns
for col in ["Classes", "class", "Region", "year", "month", "day"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Outlier detection using IQR
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

print("\nQ1:\n", Q1)
print("\nQ3:\n", Q3)
print("\nIQR:\n", IQR)

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("\nLower limit:\n", lower_limit)
print("\nUpper limit:\n", upper_limit)


mask_outliers = ((df[numeric_cols] < lower_limit) | (df[numeric_cols] > upper_limit)).any(axis=1)
outliers = df[mask_outliers]
print("\nOutliers:\n", outliers.head())

no_outliers = df[~mask_outliers]
print("\nDataframe without outliers shape:", no_outliers.shape)

# Histograms after outlier removal
no_outliers[numeric_cols].hist(figsize=(12, 10), bins=5, edgecolor='black')
plt.suptitle("Histograms after Outlier Removal")
plt.show()

mean_values = no_outliers[numeric_cols].mean()
print("\nMean values after outlier removal:\n", mean_values)

std_values = no_outliers[numeric_cols].std()
print("\nStd values after outlier removal:\n", std_values)

# Train-test split on cleaned data
X = no_outliers.drop(columns=["FWI"])
y = no_outliers["FWI"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lin_reg = LinearRegression().fit(x_train_scaled, y_train)
ridge_reg = Ridge(alpha=1.0).fit(x_train_scaled, y_train)
lasso_reg = Lasso(alpha=0.01).fit(x_train_scaled, y_train)

y_pred_lin = lin_reg.predict(x_test_scaled)
y_pred_ridge = ridge_reg.predict(x_test_scaled)
y_pred_lasso = lasso_reg.predict(x_test_scaled)

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

evaluate_model("Linear Regression", y_test, y_pred_lin)
evaluate_model("Ridge Regression", y_test, y_pred_ridge)
evaluate_model("Lasso Regression", y_test, y_pred_lasso)

results = pd.DataFrame({
    'Actual FWI': y_test.values,
    'Linear Predicted': y_pred_lin,
    'Ridge Predicted': y_pred_ridge,
    'Lasso Predicted': y_pred_lasso
})

print("\nSample Predictions (Top 10):")
print(results.head(10))

joblib.dump(lin_reg, "Linear_regression_models.pkl")
joblib.dump(scaler, "Scalers.pkl")

print("\nLinear Regression model and Scaler saved successfully!")
