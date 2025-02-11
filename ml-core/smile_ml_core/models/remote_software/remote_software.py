import json
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests import RequestException

from smile_ml_core.data.structures.dict_data_frame import DictDataFrame
from smile_ml_core.models._base_model import BaseModel


class RemoteSoftware(BaseModel):
    """
    Send incoming dataset by URL to remote server with installed industrial
    software and return results in the form of dataset.
    """

    def get_docs(self) -> str:
        return self.__doc__ or ''

    def get_extra_parameters(self, name: str, input_features: Any, **kwargs) -> list[dict[str, Any]]:
        return [
            {
                'model': name,
                'name': 'url',
                'input_type': 'str',
                'default': 'http://127.0.0.1:5000/api/v1/func/rozenbrock/',
                'description': 'Required. The url of the request',
            },
            {
                'model': name,
                'name': 'username',
                'input_type': 'str',
                'default': None,
                'description': 'Username for HTTP authentication',
            },
            {
                'model': name,
                'name': 'password',
                'input_type': 'str',
                'default': None,
                'description': 'Password for HTTP authentication',
            },
            {
                'model': name,
                'name': 'timeout',
                'input_type': 'number',
                'default': None,
                'description': 'A number indicating how many seconds to wait for the client to make a connection and/or send a response',
            },
            {
                'model': name,
                'name': 'proxy',
                'input_type': 'str',
                'default': None,
                'description': 'Optional. A dictionary of the protocol to the proxy url',
            },
        ]

    def _fit(self, X: DictDataFrame, y: pd.Series | np.ndarray | None = None, **kwargs) -> BaseModel:
        return self

    def _predict(self, X: DictDataFrame, **kwargs) -> DictDataFrame:
        if self._parameters.get('url') is None:
            raise ValueError('Not valid URL')

        df: pd.DataFrame = X.view()
        df_out = self.call_software(df, **self._parameters)

        return DictDataFrame(df=df_out)

    def call_software(
        self,
        df: pd.DataFrame,
        url: str,
        username: str | None = None,
        password: str | None = None,
        timeout: int | None = None,
        proxy: str | None = None,
    ) -> pd.DataFrame:
        data = json.dumps({'data': df.to_dict('index')})
        auth = (username, password) if username and password else None
        proxies = {'http': proxy, 'https': proxy} if proxy else None

        response = requests.post(url, data=data, auth=auth, timeout=timeout, proxies=proxies)
        response.raise_for_status()

        if response.status_code == 200:
            data_out: dict = response.json()['data']
            data_out = {int(k): v for k, v in data_out.items()}
            df_out = pd.DataFrame.from_dict(data_out, orient='index')

            return df_out
        else:
            raise RequestException(response.text)
