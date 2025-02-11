from typing import TYPE_CHECKING, Optional

import pandas as pd

from graph.core.models._abstract_module import AbstractModule, get_params
from graph.core.models._class_state import ClassState
from graph.core.models.custom.conditions.index import ConditionModel
from graph.core.models.iterations_tool import IterationsTool
from graph.core.models.sampling_model import SamplingModel
from graph.core.models.scorer import Scorer
from graph.core.node.node_feature import NodeFeature
from graph.exceptions import GraphIsNotCalledException

if TYPE_CHECKING:
    from graph.core.node.node_model import NodeModel


class StreamRecallingModel(AbstractModule):
    """
    Managing the iterative process of surrogate modeling.
    """

    def __init__(
        self,
        smile_model_path: str,
        model_class_path: str | None = None,
    ):
        # Текущий узел модели, необходим для того, чтобы повесить событие
        self.current_node_model: Optional["NodeModel"] = None

        # Узел генерации данных. Тот узел модели, где данные генерируются
        self.generation_node: Optional["NodeModel"] = None

        # Узел проверки условий. Узел проверяет, выполнилось ли условие после итерации
        self.condition_node: Optional["NodeModel"] = None

        # Узел триггер, какой узел модели необходимо слушать, чтобы перезапустить итерацию
        self.trigger_node: Optional["NodeModel"] = None

        super().__init__(
            smile_model_path=smile_model_path,
            model_class_path=model_class_path,
        )

    def get_extra_params(
        self,
        extra_params: Optional[list] = None,
        node: Optional["NodeModel"] = None,
        **kwargs,
    ) -> list:
        child_models = node.get_siblings(kind="models")
        models = {m.id: str(m) for m in child_models}
        extra_params = [
            {
                "model": self.get_name(),
                "name": "node_model",
                "input_type": "select",
                "types": {"select": models},
                "default": next(iter(models)) if models else None,
                "is_required": True,
                "description": "The model whose calculation completion will trigger the"
                "condition check.",
            }
        ]

        return super().get_extra_params(extra_params=extra_params, node=node, **kwargs)

    def call(self, data: dict, model_parameters: dict, **_kwargs):
        # Именно в call вешается событие на прослушивание триггера. То есть когда триггер меняется
        # то в этом узле (узле слушателе) запускается событие
        self.trigger_node.listeners.set_listener(listener_node=self.current_node_model)

        return data

    def stop_task(self):
        self.trigger_node.listeners.drop_listener(listener_node=self.current_node_model)

    def apply(self, data: list, calculation_id: str = None, **kwargs):
        pass

    def validate(self, node: "NodeModel", **kwargs):
        errors, warnings = super().validate(node, **kwargs)

        self.set_related_nodes(node=node)

        if self.generation_node is None:
            errors.append("The input nodes must have a data generation model.")

        if self.condition_node is None:
            errors.append("The input nodes must have a condition model.")

        properties = get_params(self.get_id(), node.get_property("properties"))
        trigger_node_id = properties.get("node_model")
        if trigger_node_id not in node.graph.nodes_map:
            errors.append("Selected node is not existed in graph")
        else:
            self.trigger_node = node.graph.nodes_map[trigger_node_id]

        return errors, warnings

    def set_related_nodes(self, node: "NodeModel") -> None:
        """
        The method finds and saves nodes with a condition and a data generation model.
        """
        self.current_node_model = node
        self.generation_node, self.condition_node = None, None

        condition_parents = node.get_parents(
            condition=lambda n: n.parent_node.is_model(ClassState.ConditionModel)
        )
        generation_parents = node.get_parents(
            condition=lambda n: not n.parent_node.is_feature()
            and not n.parent_node.is_model(ClassState.ConditionModel)
        )
        if len(condition_parents) == 1:
            self.condition_node = condition_parents[0]

        if len(generation_parents) == 1:
            self.generation_node = generation_parents[0]

    def on_listening_node_updating(self, listener_node=None) -> None:
        """
        Called when the calculation of the trigger_node is completed
        """

        scores: pd.DataFrame | None = None
        trigger_model: Scorer | IterationsTool = (
            self.trigger_node.title_property.instance
        )

        if self.trigger_node.was_called():
            if self.trigger_node.is_model(ClassState.Scorer):
                scores = trigger_model.get_scores()
                scores = scores.rename(
                    lambda x: x.replace(" ", "_").lower(), axis="columns"
                )
            elif self.trigger_node.is_model(ClassState.IterationsTool):
                scores = trigger_model.format_scores()
            else:
                raise NotImplementedError()
        else:
            raise GraphIsNotCalledException(self.trigger_node)

        condition_model: ConditionModel = self.condition_node.title_property.instance

        properties: dict = self.condition_node.get_property("properties")
        stopping_condition = condition_model.stopping_condition(scores, properties)

        if not stopping_condition:
            if self.generation_node.is_model(ClassState.SamplingModel):
                df_test_node: NodeFeature = self.trigger_node.get_parent()
                with df_test_node.data as data:
                    data_el = data.get("output", clean_names=True, return_one=True)
                df_test: pd.DataFrame = self.get_df(data_el)

                generation_model: SamplingModel = (
                    self.generation_node.title_property.instance
                )
                generation_model.update_center_point(
                    df_test,
                    trigger_model.target_column,
                    trigger_model.predict_column or "predict",
                )
            self.rerun_iteration(listener_node)
        else:
            self.logger.info(
                "Result evaluation by %s completed successfully", self.condition_node
            )

    def rerun_iteration(self, listener_node: "NodeModel") -> None:
        listener_node.graph.apply_chain_function_for_node(
            [self.generation_node], "drop_called"
        )
        listener_node.graph.run_node_with_parents(self.trigger_node)
