import numpy as np
import pandas as pd

from smile_ml_core.data.structures.dict_data_frame import DictDataFrame
from smile_ml_core.models._base_model import BaseModel


class StreamRecallingModel(BaseModel):
    """Stub class."""

    def get_docs(self) -> str:
        return self.__doc__ or ''

    def _fit(self, X: DictDataFrame, y: pd.Series | np.ndarray | None = None, **kwargs) -> BaseModel:
        return self

    def _predict(self, X: DictDataFrame, **kwargs) -> DictDataFrame:
        return X
