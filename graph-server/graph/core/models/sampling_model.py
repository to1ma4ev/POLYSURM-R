from typing import Any

import pandas as pd
from smile_ml_core.data.structures import DictDataFrame

from graph.core.models._abstract_module import (
    AbstractModule,
    get_params,
    process_arguments,
)
from graph.core.patameters.types import ParametersType


class SamplingModel(AbstractModule):
    """
    Perform sampling of input data according to selected method
    """

    to_use_input = False
    var_names_column: str = "variable_name"

    def __init__(
        self,
        smile_model_path: str,
        model_class_path: str | None = None,
    ):
        super().__init__(
            smile_model_path=smile_model_path,
            model_class_path=model_class_path,
        )

        self.first_iteration: bool = True

        self.dfs_out: list = []
        self.columns: list[str] | None = None

        self.bounds: pd.DataFrame
        self.center_point: pd.DataFrame | None = None
        self.window_sizes: pd.Series

    @property
    def return_input(self) -> bool:
        return False

    def get_extra_features(
        self, node, columns=None, properties: ParametersType = None, **kwargs
    ) -> dict:
        var_names_str: str = properties.get(f"{self.get_id()}.var_names_str", "")
        self.columns = list(map(str.strip, var_names_str.split(",")))

        if self.columns is None:
            return {}

        features = dict.fromkeys(self.columns, self.get_default_meta())

        return self.form_features_by_list(node.id, features)

    def get_extra_params(
        self, extra_params=None, node=None, prop_values: dict[str, Any] = None, **kwargs
    ) -> list:
        parameters = get_params(self.get_name(), prop_values)

        extra_params = [
            {
                "model": self.get_id(),
                "name": "et",
                "input_type": "number",
                "default": 15,
                "description": "Number of extension points",
            },
            {
                "model": self.get_id(),
                "name": "window_size",
                "input_type": "number",
                "default": 0.1,
                "description": "Window size (0.0, 1.0]",
            },
            {
                "model": self.get_id(),
                "name": "var_names_str",
                "input_type": "string",
                "default": "x1,x2",
                "description": "Variable names list",
            },
        ]

        params = self.smile_model(
            logger=self.logger, parameters=parameters
        ).get_extra_parameters(name=self.get_id(), input_features=node.features)

        # порядок параметров: nt, et, window_size, model_params, var_names_str
        params = params[:1] + extra_params[:-1] + params[1:] + extra_params[-1:]

        return super().get_extra_params(
            extra_params=params, prop_values=prop_values, **kwargs
        )

    @process_arguments
    def call(self, data: list, model_parameters, **kwargs):
        parameters = get_params(self.get_name(), model_parameters)
        et = parameters.pop("et")
        window_size = parameters.pop("window_size")
        _ = parameters.pop("var_names_str")

        if self.first_iteration:
            self.first_iteration = False

            ddf = DictDataFrame.from_dict(data[0], clean_names=True)
            df: pd.DataFrame = ddf.view()

            self.bounds = df.set_index(self.var_names_column).sort_index()
            self.window_sizes = (
                (self.bounds["upper"] - self.bounds["lower"]) * window_size / 2
            )
        else:
            if self.center_point is None:
                raise ValueError("Center point is None. Reload Graph.")

            df = pd.concat(
                [
                    self.center_point - self.window_sizes,
                    self.center_point + self.window_sizes,
                ],
                axis=1,
                keys=["lower", "upper"],
            )
            df = df.clip(self.bounds["lower"], self.bounds["upper"], axis=0)
            df = df.fillna(self.bounds)
            df = df.reset_index(names=self.var_names_column)

            ddf = DictDataFrame(df=df)
            parameters.update(nt=et)

        model = self.smile_model(logger=self.logger, parameters=parameters)
        ddf_out = model.fit(ddf).predict(ddf)

        self.dfs_out.append(ddf_out.view())
        df_out = pd.concat(self.dfs_out, ignore_index=True)
        self.columns = df_out.columns

        data_out = [{"name": data[0]["name"], "data": {"df": df_out}}]

        return data_out

    def update_center_point(
        self, df_test: pd.DataFrame, target_column: str, predict_column: str
    ) -> None:
        point_index = ((df_test[predict_column] - df_test[target_column]) ** 2).argmax()
        columns = list(filter(lambda c: c in df_test.columns, self.columns))
        self.center_point = df_test[columns].iloc[point_index].sort_index()

    def apply(self, data: list, **kwargs):
        pass
