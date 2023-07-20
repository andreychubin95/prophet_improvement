import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.model_selection import train_test_split

from .prophets import BaselineProphet, ProphetsEnsemble
from .gbt import OptunaLGBMRegressor


class UnivariateForecaster:
    def __init__(self, prophet_module: Union[BaselineProphet, ProphetsEnsemble], gbt_params: dict):
        self.prophet_ = prophet_module
        self.gbt = OptunaLGBMRegressor(**gbt_params)
        self.is_fitted_ = False
        self.prophet_forecast = None

    def fit(self, X: pd.DataFrame, periods: int) -> None:
        self.prophet_.fit(X)
        forecast = self.prophet_.forecast(periods)
        self.prophet_forecast = forecast.iloc[-periods:].copy()
        data = X.merge(forecast, on='ds', how='left')
        train_gbt, val_gbt = train_test_split(data, test_size=0.15, random_state=42)
        self.gbt.fit(train_gbt.drop(['ds', 'y'], 1), train_gbt.y.values, val_gbt.drop(['ds', 'y'], 1), val_gbt.y.values)
        self.is_fitted_ = True

    def forecast(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        assert self.is_fitted_, 'Forecaster is not fitted!'
        if X is not None:
            data = X.merge(self.prophet_forecast, on='ds', how='left')
        else:
            data = self.prophet_forecast.copy()

        yhat = self.gbt.predict(data.drop(['ds'], axis=1))
        return yhat
