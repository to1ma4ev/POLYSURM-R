from typing import TYPE_CHECKING, Optional

import numpy as np
from django.utils.translation import gettext as _

if TYPE_CHECKING:
    from data.modules.machine_learning.node import NodeModel

from data.modules.machine_learning.models._abstract_module import (
    AbstractModule,
    get_params,
)
from data.modules.machine_learning.models._class_state import ClassState


class StreamRecallingModel(AbstractModule):
    def __init__(self, instance, plots=None):
        self.iters = 0
        self.max_iters = np.inf

        # Текущий узел модели, необходим для того, чтобы повесить событие
        self.current_node_model: Optional['NodeModel'] = None

        # Узел генерации данных. Тот узел модели, где данные генерируются
        self.generation_node: Optional['NodeModel'] = None

        # Узел проверки условий. Узел проверяет, выполнилось ли условие после итерации
        self.condition_node: Optional['NodeModel'] = None

        # Узел триггер, какой узел модели необходимо слушать, чтобы перезапустить итерацию
        self.trigger_node: Optional['NodeModel'] = None

        super().__init__(instance, plots)

    def get_extra_params(
        self, extra_params: Optional[list] = None, node: Optional['NodeModel'] = None, **kwargs
    ) -> list:
        child_models = node.get_siblings(kind='models')
        models = dict(map(lambda m: (m.id, str(m)), child_models))
        extra_params = [
            {
                'model': self.get_name(),
                'name': 'node_model',
                'input_type': 'select',
                'default': list(models.keys())[0] if models else None,
                'description': 'The model whose calculation completion will trigger the condition check.',
                'types': {'select': models},
                'is_required': True,
            }
        ]

        return super(StreamRecallingModel, self).get_extra_params(extra_params=extra_params, node=node, **kwargs)

    def call(self, data: dict, model_parameters: dict, **_kwargs):
        # Именно в call вешается событие на прослушивание триггера. То есть когда триггер меняется
        # то в этом узле (узле слушателе) запускается событие
        self.trigger_node.listeners.set_listener(listener_node=self.current_node_model)

        return data

    def stop_task(self):
        self.trigger_node.listeners.drop_listener(listener_node=self.current_node_model)

    def apply(self, data: list, calculation_id: str = None):
        pass

    def validate(self, node: 'NodeModel', **kwargs):
        errors, warning = [], []

        self.set_related_nodes(node=node)

        if self.generation_node is None:
            errors.append(_('The input nodes must have a data generation model.'))

        if self.condition_node is None:
            errors.append(_('The input nodes must have a condition model.'))

        properties = get_params(self.get_id(), node.get_property('properties'))
        trigger_node_id = properties.get('node_model')
        if trigger_node_id not in node.graph.nodes_map:
            errors.append(_('Selected node is not existed in graph'))
        else:
            self.trigger_node = node.graph.nodes_map[trigger_node_id]

        return errors, warning

    def set_related_nodes(self, node: 'NodeModel'):
        """
        The method finds and saves nodes with a condition and a data generation model.
        """
        self.current_node_model = node
        self.generation_node, self.condition_node = None, None

        condition_parents = node.get_parents(condition=lambda n: n.parent_node.is_model(ClassState.ConditionModel))
        generation_parents = node.get_parents(
            condition=lambda n: not n.parent_node.is_feature() and not n.parent_node.is_model(ClassState.ConditionModel)
        )
        if len(condition_parents) == 1:
            self.condition_node = condition_parents[0]
            self.max_iters = int(self.condition_node.get_property('properties')['MetricCondition.Iterations number'])

        if len(generation_parents) == 1:
            self.generation_node = generation_parents[0]

            if self.trigger_node is not None:
                trigger_model = self.trigger_node.title_property.instance
                generation_model = self.generation_node.title_property.instance

                if hasattr(trigger_model, 'target_column') and hasattr(generation_model, 'target_column'):
                    generation_model.target_column = trigger_model.target_column

    def on_listening_node_updating(self, listener_node=None):
        """
        Called when the calculation of the trigger_node is completed
        """

        condition_model = self.condition_node.title_property.instance.instance()
        condition_props = self.condition_node.get_properties('properties')

        scores_table = None
        if self.trigger_node.is_model(ClassState.Scorer):
            scores_table = self.trigger_node.title_property.instance.get_scores()
            scores_table = scores_table.rename(lambda x: x.replace(' ', '_').lower(), axis='columns')
        elif self.trigger_node.is_model(ClassState.IterationsTool):
            scores_table = self.trigger_node.title_property.instance.format_scores()

        self.logger.warning(
            f'Event source: {str(listener_node)}. Result evaluation by {str(self.condition_node)} started'
        )

        if self.iters < self.max_iters and not condition_model.fit(scores_table, condition_props):
            self.iters += 1
            self.logger.info(f'Iteration {self.iters} of {self.max_iters}')
            self.rerun_iteration(listener_node)
        else:
            self.logger.info(
                f'Event source: {str(listener_node)}. '
                f'Result evaluation by {str(self.condition_node)} completed successfully'
            )
            self.logger.info(f'Solution reached at iteration {self.iters} of {self.max_iters}')

    def rerun_iteration(self, listener_node: 'NodeModel'):
        listener_node.graph.apply_chain_function_for_node([self.generation_node], 'drop_called')
        listener_node.graph.run_node_with_parents(self.trigger_node)
