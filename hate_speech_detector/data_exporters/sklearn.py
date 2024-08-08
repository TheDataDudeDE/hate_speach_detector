from typing import Dict, Tuple, Union

from pandas import Series, DataFrame
from sklearn.base import BaseEstimator


if "data_exporter" not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def register(
    settings: Tuple[
        DataFrame,
        Series,
        BaseEstimator,
    ],
    **kwargs,
) -> Tuple[BaseEstimator, Dict[str, str]]:
    X, y, model = settings

    model.fit(X, y)

    return model
