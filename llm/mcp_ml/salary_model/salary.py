import numpy as np
import pandas as pd
import sqlite3
import os
from joblib import dump, load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate


ROOT_DIR = os.path.dirname(__file__)
# Path to the SQLite database file
DB_PATH = os.path.join(ROOT_DIR, "../data/data_jobs.db")
# Path to model
MODEL_PATH = os.path.join(ROOT_DIR, "../data/salary_model.dat")


def read_ads_from_db() -> pd.DataFrame:
    """
    Reads the 'ads' table from the SQLite database and returns it as a DataFrame.

    :return: A Pandas DataFrame containing the contents of the 'ads' table.
    """
    try:
        # Establish a connection to the database
        connection = sqlite3.connect(DB_PATH)

        # Read the 'ads' table into a Pandas DataFrame
        ads_df = pd.read_sql_query("SELECT * FROM ads", connection)

        return ads_df
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    finally:
        # Ensure the connection is closed
        if 'connection' in locals():
            connection.close()


def features_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_ads = read_ads_from_db()
    levels = {
        'Entry-level': 1,
        'Junior': 2,
        'Intermediate': 2.5,
        'Mid-level': 3,
        'Senior-level': 4,
        'Expert': 5,
        'Director': 6,
        'Executive-level': 7
    }

    def calc_level_num(val):
        return np.mean(
            [levels.get(v.strip(), 0) for v in val.split("/")]
        )

    df["level_num"] = df["level"].map(calc_level_num)

    norm_title_map = {x: idx for idx, x in enumerate(df_ads["norm_title"].unique()) if x}
    df["norm_title_cat"] = df["norm_title"].map(lambda x: norm_title_map.get(x, -1))

    return df


def get_feats() -> list:
    return ["norm_title_cat", "level_num"]


def train_model():
    df_ads = read_ads_from_db()
    df_new = features_engineering(df_ads)

    feats = get_feats()
    X = df_new[feats].values
    y = df_new["salary_usd_min"].values

    model = GradientBoostingRegressor(max_depth=7, n_estimators=100, learning_rate=0.3, random_state=0)

    result = cross_validate(model, X, y, return_estimator=True, cv=2, scoring="neg_mean_absolute_error")
    print("MAE", result["test_score"].mean(), "std: ", result["test_score"].std())

    model = result["estimator"][0]
    save_model(model)
    return model


def save_model(model) -> None:
    dump(model, MODEL_PATH)


def load_model():
    return load(MODEL_PATH)


def read_or_train_model():
    if os.path.isfile(MODEL_PATH):
        return load_model()
    else:
        return train_model()



def predict(sample):
    model = read_or_train_model()
    samples = [sample]

    df_new = features_engineering(pd.DataFrame(samples))
    feats = get_feats()
    X_new = df_new[feats].values

    y_pred = model.predict(X_new)
    return round(float(y_pred[0]), 2)


if __name__ == "__main__":
    print('path: ', DB_PATH)
    if os.path.isfile(DB_PATH):
        print('The database exists')
    else:
        print('The database does not exit')

    ads_df = read_ads_from_db()
    print(ads_df.head(10))

    prediction = predict({'norm_title': 'Software Developer', 'level': 'Mid-level'})
    print(f'Predicted salary: {prediction * 1000} USD yearly')

