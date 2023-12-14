import uuid

from django.utils.translation import gettext as _

from data.modules.machine_learning.node.node_model import NodeModel
from helper.servives.helpers._base_helper import BaseHelper


class HelperStreamRecalling(BaseHelper):
    name = 'StreamRecalling'
    group_name = 'streamrecalling'
    description = 'циклический запуск графа'

    def __init__(self, *args, **kwargs):
        super(HelperStreamRecalling, self).__init__(*args, **kwargs)

    def __str__(self):
        return self.name

    def get_accessed_options(self):
        if self.last_node:
            return super(HelperStreamRecalling, self).get_accessed_options()
        return {}

    def detect_last_data_node(self, raise_exc: bool = True):
        return super(HelperStreamRecalling, self).detect_last_data_node(raise_exc=False)

    def estimate(self, previous_helper=None) -> bool:
        self._refactor_actions = []

        self._refactor_actions.append(
            {
                'method': 'create_stream_recalling_module',
                'done': False,
            }
        )

        self._refactor_actions.append(
            {
                'method': 'create_edge',
                'from': self._last_data_node,
                'to': self._node_module,
                'done': False,
            }
        )

        self.send_hint(message=_('To process something'))

        return self.is_valid

    def create_stream_recalling_module(self, action: dict):
        if not action['done']:
            graph = self._graph

            model_name = 'StreamRecallingModel'
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
