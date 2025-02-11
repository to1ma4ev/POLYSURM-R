from helper.servives.helpers.models import BaseModelHelper, HelperMetricCondition
from helper.servives.structures import FeatureOption


class HelperStreamRecalling(BaseModelHelper):
    """ """

    name = 'Циклический перезапуск графа'
    group_name = 'stream_recalling'
    description = 'Управление итеративным процессом'

    hint_message = 'To manage iterative process'

    @property
    def model_name(self):
        return 'StreamRecallingModel'

    def estimate_by_meta(self, features: dict[str, FeatureOption]):
        return True

    def estimate(self, previous_helper=None) -> bool:
        super().estimate(previous_helper)
        self.estimate_metric_condition()

        return self.is_valid

    def estimate_metric_condition(self):
        helper_instance = HelperMetricCondition(
            sender=self.sender,
            project_id=self._project_id,
            previous_helper=self,
            target_column=self.target_column,
            send_to_client=self.send_to_client,
            parent_group=self.parent_group,
            task_type=self.task_type,
            params=self.parameters,
            parent_id=self.group,
        )
        helper_instance.estimate()

        return self.is_valid

    def refactor(self, groups: list | None = None) -> bool:
        super().refactor(groups)

        helper_instance = HelperMetricCondition(
            sender=self.sender,
            project_id=self._project_id,
            previous_helper=self,
            target_column=self.target_column,
            send_to_client=self.send_to_client,
            parent_group=self.parent_group,
            task_type=self.task_type,
            params=self.parameters,
            parent_id=self.group,
        )
        helper_instance.estimate()
        helper_instance.refactor(groups=groups)

        return self.is_valid
