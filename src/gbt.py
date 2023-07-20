import numpy as np
import pandas as pd
import optuna.integration.lightgbm as lgb
from optuna.integration.lightgbm import Dataset


class OptunaLGBMRegressor:
    """
    A wrapper class for the LightGBM Regressor with Optuna for hyperparameters tuning
    """

    def __init__(
            self,
            n_estimators: int,
            learning_rate: float = 0.01,
            metric: str = 'rmse',
            cat_columns: str = 'auto',
            seed: int = 42
    ):
        """
        Initializes a new instance of the OptunaLGBMRegressor class
        """
        self.params = {
            "n_estimators": n_estimators,
            "objective": "regression",
            "verbosity": -1,
            "metric": metric,
            "learning_rate": learning_rate,
            "boosting_type": 'gbdt',
            "random_state": seed
        }
        self.cat_columns = cat_columns
        self.model = None
        self.features = None
        self.is_fitted_ = False

    def _to_datasets(
            self, x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray
    ) -> (Dataset, Dataset):
        """
        Converts Pandas DataFrames to LightGBM Datasets
        """
        self.features = list(x_train.columns)
        X_val = x_val[self.features].copy()
        dtrain = Dataset(x_train, label=y_train, categorical_feature=self.cat_columns)
        dval = Dataset(X_val, label=y_val, categorical_feature=self.cat_columns)

        return dtrain, dval

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> None:
        dtrain, dval = self._to_datasets(X_train, y_train, X_val, y_val)

        self.model = lgb.tuner.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dval],
            verbose_eval=0,
            early_stopping_rounds=150
        )

        self.is_fitted_ = True

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self.is_fitted_, 'Model is not fitted!'
        return self.model.predict(X_test[self.features], num_iteration=self.model.best_iteration)

