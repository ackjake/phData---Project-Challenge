import json
import pathlib
import pickle
from typing import List, Tuple

import pandas
from sklearn import model_selection, pipeline, preprocessing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(
        sales_path, usecols=sales_column_selection, dtype={"zipcode": str}
    )
    demographics = pandas.read_csv(
        "data/zipcode_demographics.csv", dtype={"zipcode": str}
    )

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(
        columns="zipcode"
    )
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop("price")
    x = merged_data

    return x, y


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42, test_size=0.25
    )

    model = pipeline.make_pipeline(
        HistGradientBoostingRegressor(
            loss="gamma",
            max_depth=4,
            max_features=0.7,
            max_iter=1000,
            early_stopping=True,
            random_state=42,
        ),
    ).fit(x_train, y_train)

    y_preds = model.predict(x_test)
    rsq = r2_score(y_test, y_preds)
    mse = mean_squared_error(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)
    print(f"RSq2: {rsq:.2f}, MSE: {mse:.0f}, MAE: {mae:.0f}")

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model.pkl", "wb"))
    json.dump(list(x_train.columns), open(output_dir / "model_features.json", "w"))


# naive knn: RSq2=0.72, MSE=40666526377, MAE=102044
# gbm gamma: RSq2: 0.80, MSE: 29395686789, MAE: 8898

if __name__ == "__main__":
    main()
