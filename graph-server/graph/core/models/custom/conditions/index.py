from typing import TYPE_CHECKING, Any

import pandas as pd

from graph.core.models._abstract_module import (
    AbstractModule,
    get_params,
    process_arguments,
)
from graph.exceptions import NotSupported

if TYPE_CHECKING:
    from smile_ml_core.models._base_model import BaseModel


class ConditionModel(AbstractModule):
    """
    A model for using with StreamRecallingModel for checking of results of a calculation
    of another model.
    """

    def __init__(
        self,
        smile_model_path: str,
        model_class_path: str | None = None,
    ):
        self.iters = 0

        super().__init__(
            smile_model_path=smile_model_path,
            model_class_path=model_class_path,
        )

    def get_extra_params(
        self, extra_params=None, node=None, prop_values: dict[str, Any] = None, **kwargs
    ) -> list:
        parameters = get_params(self.get_name(), prop_values)

        extra_params = [
            {
                "model": self.get_id(),
                "name": "count_iter",
                "input_type": "number",
                "default": 15,
                "description": "Maximum number of iterations during cyclic process",
            }
        ]

        params = self.smile_model(
            logger=self.logger, parameters=parameters
        ).get_extra_parameters(name=self.get_id(), input_features=node.features)

        extra_params.extend(params)

        return super().get_extra_params(
            extra_params=extra_params, prop_values=prop_values, **kwargs
        )

    def call(self, data, model_parameters, **kwargs):
        return self.fit(
            data_el=data,
            parameters=model_parameters,
        )

    def apply(self, data: list, calculation_id: str | None = None, **kwargs):
        raise NotSupported()

    @process_arguments
    def stopping_condition(self, data: pd.DataFrame, model_parameters: dict) -> bool:
        params = get_params(self.get_name(), model_parameters)
        max_iters = params.pop("count_iter")

        model: BaseModel = self.smile_model(logger=self.logger, parameters=params)

        self.iters += 1
        self.logger.info("Iteration %s of %s", self.iters, max_iters)

        if hasattr(model, "stopping_condition"):
            stopping_condition = model.stopping_condition(data)

        return self.iters >= max_iters or stopping_condition
