import timeit
from typing import Any

import pandas as pd
from requests.exceptions import RequestException
from smile_ml_core.data.structures import DictDataFrame

from graph.core.models._abstract_module import (
    AbstractModule,
    get_params,
    process_arguments,
)
from graph.core.node.data_mix import DataElement
from graph.core.patameters.types import ParametersType
from graph.exceptions import RemoteException


class RemoteSoftware(AbstractModule):
    """
    Send incoming dataset by URL to remote server with installed industrial software
    and return results in the form of dataset
    """

    def __init__(
        self,
        smile_model_path: str,
        model_class_path: str | None = None,
    ):
        super().__init__(
            smile_model_path=smile_model_path,
            model_class_path=model_class_path,
        )

        self.time: float = 0
        self.columns: list[str] | None = None

    def get_extra_features(
        self, node, columns=None, properties: ParametersType = None, **kwargs
    ) -> dict:
        output_feat_str: str = properties.get(f"{self.get_id()}.output_feat_str", "")
        self.columns = list(map(str.strip, output_feat_str.split(",")))

        if self.columns is None:
            return {}

        features = dict.fromkeys(self.columns, self.get_default_meta())

        return self.form_features_by_list(node.id, features)

    def get_extra_params(
        self,
        extra_params: list[dict[str, Any]] = None,
        node=None,
        prop_values: dict[str, Any] = None,
        **kwargs,
    ) -> list:
        parameters = get_params(self.get_name(), prop_values)

        extra_params = [
            {
                "model": self.get_id(),
                "name": "output_feat_str",
                "input_type": "string",
                "default": "f(x)",
                "description": "Output features list",
            }
        ]

        params = self.smile_model(
            logger=self.logger, parameters=parameters
        ).get_extra_parameters(name=self.get_id(), input_features=node.features)

        params.extend(extra_params)

        return super().get_extra_params(
            extra_params=params, prop_values=prop_values, **kwargs
        )

    @process_arguments
    def call(self, data: DataElement, model_parameters: dict[str, Any], **kwargs):
        parameters = get_params(self.get_name(), model_parameters)
        _ = parameters.pop("output_feat_str")
        model = self.smile_model(logger=self.logger, parameters=parameters)

        start_time = timeit.default_timer()

        data_out = []
        for data_el in data:
            ddf = DictDataFrame.from_dict(data_el, clean_names=True)

            try:
                ddf_out = model.fit(ddf).predict(ddf)
            except RequestException as e:
                raise RemoteException(
                    f"Remote software calculation failed: {e!s}"
                ) from e

            data_out.append({"name": data_el["name"], "data": {"df": ddf_out.view()}})
            self.columns = ddf_out.columns.tolist()

        self.time = timeit.default_timer() - start_time

        return data_out

    def get_json(self, **kwargs) -> list[dict[str, str | dict]]:
        df = pd.DataFrame([self.time], columns=["Время работы узла, с"])
        label = "Время работы узла запуска индустриального ПО, с"
        return self.build_table_by_df_json((label, df))

    def apply(self, data: list, **kwargs) -> None:
        pass
