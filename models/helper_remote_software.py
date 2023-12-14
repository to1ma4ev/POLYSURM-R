import uuid

from django.utils.translation import gettext as _

from data.modules.machine_learning.models.remote_software import RemoteSoftware
from data.modules.machine_learning.node import NodeModel
from helper.servives.helpers._base_helper import BaseHelper
from polygon.models import Competition
from project.models import Project


class HelperRemoteSoftware(BaseHelper):
    name = 'Удаленное ПО'
    group_name = 'remote_software'
    description = 'Удаленное ПО'

    hint_message = 'To connect ...'

    def __init__(self, *args, **kwargs):
        super(HelperRemoteSoftware, self).__init__(*args, **kwargs)

    def estimate(self, previous_helper=None):
        self._refactor_actions.append(
            {
                'method': 'create_remote_software_module',
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

    def create_remote_software_module(self, action: dict):
        if not action['done']:
            graph = self._graph
            model_name = RemoteSoftware.__name__
            label = (model_name + self.postfix) if self.postfix else self.get_node_new_name(name=model_name)

            node = {
                'id': uuid.uuid4().hex,
                'type': 'methods',
                'position': {},
                'properties': {NodeModel.select_field_name: model_name, 'label': label},
            }
            graph.create_element(node=node)

            self._node_module: NodeModel = graph.get_node(node['id'])
            model = self._node_module.title_property.instance
            group = Project.objects.get(pk=graph.project_id).group.first()
            params_surrogate = Competition.objects.get(project=group).params_surrogate

            self._node_module.update_model_properties(
                {
                    'properties': {
                        f'{model.get_id()}.URL': params_surrogate.url,
                        f'{model.get_id()}.Proxy': params_surrogate.proxy or None,
                        f'{model.get_id()}.User': params_surrogate.username or None,
                        f'{model.get_id()}.Password': params_surrogate.password or None,
                        f'{model.get_id()}.Timeout': params_surrogate.timeout,
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

                    edge = {
                        'id': uuid.uuid4().hex,
                        'type': action.get('type', 'defaultEdge'),
                        'from': from_node.id,
                        'to': to_node.id,
                    }
                    self._graph.create_element(node=edge)
                    edge_created = self._graph.get_edge(edge['id'])

                    message = f'Ребро "{str(edge_created)}" было создано'
                    self.sender.element_created(
                        self._project_id, data={'edge': edge_created.format(), 'message': message}
                    )
