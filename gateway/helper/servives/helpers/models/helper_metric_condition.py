from smile_ml_core.classes.scoring import MetricName

from helper.servives.helpers.models import BaseModelHelper
from helper.servives.structures import FeatureOption
from polygon.constants import COUNT_ITER_DEFAULT, METRIC_THRESHOLD_DEFAULT
from polygon.models import Competition


class HelperMetricCondition(BaseModelHelper):
    """ """

    name = 'Условие выхода из цикла'
    group_name = 'metric_condition'
    description = 'Проверка условия выхода из цикла'

    hint_message = 'To checking stopping condition'

    @property
    def model_name(self):
        return 'MetricCondition'

    @property
    def default_properties(self):
        competition = Competition.objects.filter(baseline_module__id=self._project_id).first()

        return {
            f'{self.model_name}.metric': MetricName(competition.metric).name.lower(),
            f'{self.model_name}.metric_threshold': self.parameters.get('metric_threshold', METRIC_THRESHOLD_DEFAULT),
            f'{self.model_name}.count_iter': self.parameters.get('count_iter', COUNT_ITER_DEFAULT),
        }

    def estimate_by_meta(self, features: dict[str, FeatureOption]):
        return True

    def create_refactor_events(self, *, done=False):
        self._refactor_actions.append(
            {
                'method': 'create_model',
                'done': done,
            }
        )

        self._refactor_actions.append(
            {
                'method': 'create_edge_to_stream_recalling_model',
                'done': done,
            }
        )

    def create_edge_to_stream_recalling_model(self, action: dict):
        """
        Создание ребра от модели из _node_module к stream_recalling_model
        """
        self.create_edge(action | {'from': self._node_module.id, 'to': self._last_data_node.id})
