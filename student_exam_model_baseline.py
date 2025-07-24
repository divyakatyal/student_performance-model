import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("StudentPerformanceFactors.csv")

X = df.drop(columns=["Exam_Score"])
y = df["Exam_Score"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Pipeline: preprocessing + feature selection + ridge regression
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", SelectKBest(score_func=f_regression)),
    ("regressor", Ridge())
])

# Grid search parameters
param_grid = {
    "feature_selection__k": [10, 20, "all"],
    "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
}

# Run GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate best model on test set
y_pred = grid_search.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Best Parameters:", grid_search.best_params_)
print(f"Test R² Score: {r2:.4f}")
print(f"Test RMSE: {rmse:.4f}")
