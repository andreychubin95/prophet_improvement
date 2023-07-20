import prophet
import pandas as pd
from .utils import suppress_stdout_stderr


class BaselineProphet:
    def __init__(self, freq: str):
        self.freq = freq
        self.prophet_ = prophet.Prophet()
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame) -> None:
        with suppress_stdout_stderr():
            self.prophet_.fit(X)
        self.is_fitted_ = True

    def forecast(self, periods: int) -> pd.DataFrame:
        predictions = self.prophet_.make_future_dataframe(periods=periods, freq=self.freq)
        forecast = self.prophet_.predict(predictions)
        return forecast
