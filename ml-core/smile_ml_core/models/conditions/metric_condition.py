from typing import Any

import numpy as np
import pandas as pd

from smile_ml_core.classes.scoring import MetricName
from smile_ml_core.data.structures.dict_data_frame import DictDataFrame
from smile_ml_core.models import BaseModel


class MetricCondition(BaseModel):
    """
    Checking the quality of result metrics of models.
    The model returns True if metric condition satisfied, False otherwise.
    """

    def get_extra_parameters(self, name: str, input_features: Any, **kwargs: Any) -> list[dict[str, Any]]:
        access_scoring_list = MetricName.surrogate_modeling_names()

        return [
            {
                'model': name,
                'name': 'metric',
                'input_type': 'select',
                'types': {'select': access_scoring_list},
                'default': access_scoring_list[0] if access_scoring_list else '',
                'description': 'Metric which value to track during cyclic process',
            },
            {
                'model': name,
                'name': 'metric_threshold',
                'input_type': 'number',
                'default': 0.01,
                'description': 'Desirable threshold value of selected metric',
            },
        ]

    def _fit(self, X: DictDataFrame, y: pd.Series | np.ndarray | None = None, **kwargs) -> BaseModel:
        return self

    def _predict(self, X: DictDataFrame, **kwargs) -> DictDataFrame:
        return X

    def stopping_condition(self, data: pd.DataFrame) -> bool:
        metric_name = self._parameters.get('metric')
        metric_threshold = self._parameters.get('metric_threshold')

        metric_value = data[metric_name].mean()
        k = -1 if metric_name == MetricName.R2.name.lower() else 1

        return k * (metric_threshold - metric_value) >= 0
