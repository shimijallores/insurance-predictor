from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "insurance.csv"
MODEL_PATH = BASE_DIR / "insurance_model.pkl"


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    smoker = (X["smoker"] == "yes").astype(float)
    X["smoker_bmi"] = smoker * X["bmi"]
    X["smoker_age"] = smoker * X["age"]
    X["age_sq"] = X["age"] ** 2
    X["bmi_sq"] = X["bmi"] ** 2
    X["obese"] = (X["bmi"] >= 30).astype(float)
    X["smoker_obese"] = smoker * X["obese"]
    X["age_bmi"] = X["age"] * X["bmi"]
    return X


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
    y = df["charges"]

    categorical_features = ["sex", "smoker", "region"]
    numeric_features = [
        "age", "bmi", "children",
        "smoker_bmi", "smoker_age",
        "age_sq", "bmi_sq",
        "obese", "smoker_obese", "age_bmi",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(engineer_features)),
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Linear Regression Evaluation Metrics")
    print("Using domain-specific feature engineering")
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()