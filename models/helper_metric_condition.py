import uuid

from django.utils.translation import gettext as _

from data.modules.machine_learning.classes.scoring import MetricName
from data.modules.machine_learning.node.node_model import NodeModel
from helper.servives.helpers._base_helper import BaseHelper
from polygon.constants import COUNT_ITER_DEFAULT, METRIC_THRESHOLD_DEFAULT
from polygon.models import Competition
from project.models import Project


class HelperMetricCondition(BaseHelper):
    name = 'MetricCondition'
    group_name = 'metriccondition'
    description = 'проверка качества суррогатной модели'

    def __init__(self, *args, **kwargs):
        super(HelperMetricCondition, self).__init__(*args, **kwargs)

    def estimate(self, previous_helper=None) -> bool:
        self._refactor_actions = []

        self._refactor_actions.append(
            {
                'method': 'create_metric_condition_module',
                'done': False,
            }
        )

        self._refactor_actions.append(
            {
                'method': 'create_edge',
                'from': self._node_module,
                'to': self._last_data_node,
                'done': False,
            }
        )

        self.send_hint(message=_('To process something'))

        return self.is_valid

    def create_metric_condition_module(self, action: dict):
        if not action['done']:
            graph = self._graph

            model_name = 'MetricCondition'
            label = (model_name + self.postfix) if self.postfix else self.get_node_new_name(name=model_name)
            node = {
                'id': uuid.uuid4().hex,
                'type': 'methods',
                'position': {},
                'properties': {
                    NodeModel.select_field_name: model_name,
                    'label': label,
                },
            }
            graph.create_element(node=node)
            self._node_module = graph.get_node(node['id'])
            model = self._node_module.title_property.instance
            module = Project.objects.get(id=self._project_id)
            group = module.group
            params = Competition.objects.get(project_id=group.first().id).params_surrogate
            metric = params.competition.metric
            self._node_module.update_model_properties(
                {
                    'properties': {
                        f'{model.get_id()}.Metric': MetricName(metric).name,
                        f'{model.get_id()}.Metric value': self.parameters.get(
                            'metric_threshold', METRIC_THRESHOLD_DEFAULT
                        ),
                        f'{model.get_id()}.Iterations number': self.parameters.get('count_iter', COUNT_ITER_DEFAULT),
                    }
                }
            )
            self.sender.element_created(
                self._project_id,
                {'node': self._node_module.format(), 'message': f'Узел "{str(self._node_module)}" был создан'},
            )

    def create_edge(self, action: dict):
        if not action['done']:
            from_, to_ = action['from'], action['to']

            for from_node in from_ if isinstance(from_, list) else [from_]:
                from_node = self._node_module if from_node is None else from_node
                for to_node in to_ if isinstance(to_, list) else [to_]:
                    to_node = self._node_module if to_node is None else to_node

                    edge = {'id': uuid.uuid4().hex, 'type': 'defaultEdge', 'from': from_node.id, 'to': to_node.id}
                    self._graph.create_element(node=edge)
                    edge_created = self._graph.get_edge(edge['id'])

                    message = f'Ребро "{str(edge_created)}" было создано'
                    self.sender.element_created(
                        self._project_id, data={'edge': edge_created.format(), 'message': message}
                    )
