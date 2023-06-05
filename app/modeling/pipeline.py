import pandas as pd
from sklearn.pipeline import Pipeline as _Pipeline


class Pipeline(_Pipeline):
    def fit_transform(self, *args, **kwargs) -> pd.DataFrame:
        x = super().fit_transform(*args, **kwargs)
        return pd.DataFrame(x)
