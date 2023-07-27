import re
import prophet
import pandas as pd
from typing import Optional
from itertools import product
from tqdm import tqdm
from holidays.holiday_base import HolidayBase

from .utils import suppress_stdout_stderr


class ProphetsEnsemble:
    """An ensemble of Prophet models with different aggregation functions and frequencies."""

    def __init__(self, freq: str, levels: list, agg_fn: list, holidays_getter: HolidayBase = None):
        """Initializes an ensemble of Prophet models."""
        self.freq = freq
        self.levels = ['_'.join(x) for x in product(levels, agg_fn)]
        self.h_getter = holidays_getter
        self.prophets_ = dict()
        self.is_fitted_ = False

    @staticmethod
    def _resample(data: pd.DataFrame, freq: str, how: str) -> pd.DataFrame:
        """Resamples a time series DataFrame."""
        if how not in ['median', 'mean', 'sum']:
            raise NotImplementedError(f'Unknown function {how}. Only [median, mean, sum] are supported.')
        return data.set_index('ds').resample(freq).agg(how).reset_index(drop=False)

    @staticmethod
    def _merge_key_gen(x, level: str) -> str:
        """Generates a key for merging DataFrames based on the frequency."""
        freq = re.sub('[\d]', '', level.split('_')[0])
        if freq == 'H':
            return f'{x.year}-{x.month}-{x.day}-{x.hour}'
        elif freq in ['D', 'M']:
            return f'{x.year}-{x.month}-{x.day}' if freq == 'D' else f'{x.year}-{x.month}'
        elif freq == 'W':
            return f'{x.isocalendar().year}-{x.isocalendar().week}'
        raise NotImplementedError(f'Only [H, D, W, M] are supported. {freq} was recieved as input!')

    def _get_holidays(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extracts holidays from the data."""
        if self.h_getter is None:
            return None
        holidays = data[['ds']].copy()
        holidays['holiday'] = holidays['ds'].apply(self.h_getter.get)
        return holidays.dropna()

    def _fit_level(self, data: pd.DataFrame, level: str) -> None:
        """Fits a Prophet model for a specific aggregation level."""
        resampled = self._resample(data, *level.split('_')) if level != self.freq else data.copy()
        fb = prophet.Prophet(holidays=self._get_holidays(resampled))
        with suppress_stdout_stderr():
            fb.fit(resampled)
        self.prophets_[level] = fb

    def _predict_level(self, periods: int, level: str) -> pd.DataFrame:
        """Makes predictions for a specific aggregation level."""
        fb = self.prophets_[level]
        df = fb.make_future_dataframe(periods=periods, freq=level.split('_')[0])
        forecasts = fb.predict(df)
        forecasts.columns = [f'{x}_{level}' for x in forecasts.columns]
        return forecasts

    def _combine_levels(self, base_df: pd.DataFrame, data: pd.DataFrame, level: str) -> pd.DataFrame:
        """Combines predictions from different aggregation levels."""
        key = lambda x: self._merge_key_gen(x, level)
        return (
            base_df.assign(key=base_df['ds'].apply(key))
                .merge(data.assign(key=data[f'ds_{level}'].apply(key)), on='key', how='left')
                .drop(['key', f'ds_{level}'], axis=1)
        )

    @staticmethod
    def _drop_redundant(data: pd.DataFrame) -> pd.DataFrame:
        """Drops redundant features from the DataFrame."""
        redundant = [col for col in data.columns if col != 'ds' and 'yhat' not in col and len(data[col].unique()) == 1]
        return data.drop(redundant, axis=1)

    def fit(self, data: pd.DataFrame) -> None:
        """Fits the Prophet models for all aggregation levels."""
        for level in tqdm([self.freq] + self.levels, 'Fitting prophets...'):
            self._fit_level(data, level)
        self.is_fitted_ = True

    def forecast(self, periods: int) -> pd.DataFrame:
        """Makes forecasts for all aggregation levels and combines them."""
        assert self.is_fitted_, 'Model is not fitted'
        forecasts = [self._predict_level(periods, level) for level in tqdm([self.freq] + self.levels, 'Forecasting...')]

        forecast = forecasts[0].rename(columns={f'ds_{self.freq}': 'ds'})
        for level, fore in zip(self.levels, forecasts[1:]):
            forecast = self._combine_levels(forecast, fore, level)

        return self._drop_redundant(forecast)
