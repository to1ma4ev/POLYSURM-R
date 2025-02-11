import json

from helper.servives.helpers.models import BaseModelHelper
from helper.servives.structures import FeatureOption
from polygon.models import Competition


class HelperRemoteSoftware(BaseModelHelper):
    """ """

    name = 'Удаленное ПО'
    group_name = 'remote_software'
    description = 'Запуск удаленного ПО'

    hint_message = 'To run remote software'

    @property
    def model_name(self) -> str:
        return 'RemoteSoftware'

    @property
    def default_properties(self) -> dict:
        competition = Competition.objects.filter(baseline_module__id=self._project_id).first()
        params_surrogate = competition.params_surrogate

        features_limits = json.loads(competition.params_surrogate.features_limits)
        not_sampling_features = filter(lambda k: not features_limits[k]['sampling_input'], features_limits)
        output_feat_str = ','.join(not_sampling_features)

        return {
            f'{self.model_name}.url': params_surrogate.url,
            f'{self.model_name}.username': params_surrogate.username,
            f'{self.model_name}.password': params_surrogate.password,
            f'{self.model_name}.timeout': params_surrogate.timeout,
            f'{self.model_name}.proxy': params_surrogate.proxy,
            f'{self.model_name}.output_feat_str': output_feat_str,
        }

    def estimate_by_meta(self, features: dict[str, FeatureOption]):
        return True

    def create_refactor_events(self, *, done: bool = False):
        super().create_refactor_events(done=done)

        self._refactor_actions.append(
            {
                'method': 'create_data_node',
                'done': done,
            }
        )

        self._refactor_actions.append(
            {
                'method': 'create_output_edge',
                'done': done,
            }
        )

    def create_output_edge(self, action: dict):
        edge = self.create_edge(action | {'from': self._node_module.id, 'to': self._last_data_node.id})
        edge = self._graph.get_edge(edge.id)

        competition = Competition.objects.filter(baseline_module__id=self._project_id).first()
        features_train = json.loads(competition.features) + [self.target_column]

        features = list(filter(lambda v: v.split(':')[-1] in features_train, edge.feature_ids))
        self._graph.update_default_edge(edge.id, features=features)

    @property
    def last_node(self):
        return self._last_data_node
