from typing import Any

import numpy as np
import pandas as pd

from smile_ml_core.data.structures.dict_data_frame import DictDataFrame
from smile_ml_core.models import BaseModel


class BaseSampling(BaseModel):
    """
    Perform sampling of input data according to selected method.
    """

    def get_extra_parameters(self, name: str, input_features: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                'model': name,
                'name': 'nt',
                'input_type': 'number',
                'default': 75,
                'description': 'Number of sampling points',
            }
        ]

    def _fit(self, X: DictDataFrame, y: pd.Series | np.ndarray | None = None, **kwargs) -> BaseModel:
        return self

    def _predict(self, X: DictDataFrame, **kwargs) -> DictDataFrame:
        nt = self._parameters.pop('nt')

        df: pd.DataFrame = X.view()
        columns = df['variable_name'].tolist()
        xlimits = df[['lower', 'upper']].to_numpy()

        self._model = self.model_cls(xlimits=xlimits, **self._parameters)
        x: np.ndarray = self._model(nt)
        df_out = pd.DataFrame(x, columns=columns)

        return DictDataFrame(df=df_out)
