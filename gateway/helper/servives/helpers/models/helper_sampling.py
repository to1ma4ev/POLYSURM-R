import json

from helper.servives.helpers.models import BaseModelHelper
from helper.servives.structures import FeatureOption
from polygon.constants import (
    COUNT_GEN_SAMPLING_POINTS_DEFAULT,
    COUNT_INC_SAMPLING_POINTS_DEFAULT,
    WINDOW_SIZE_DEFAULT,
)
from polygon.models import Competition


class HelperSampling(BaseModelHelper):
    """ """

    name = 'Сэмплирование'
    group_name = 'sampling'
    description = 'Сэмплирование'

    hint_message = 'To data sampling'

    @property
    def model_name(self) -> str:
        return self.parameters.get('sampling_method', 'LatinHypercubeSampling')

    @property
    def default_properties(self) -> dict:
        competition = Competition.objects.filter(baseline_module__id=self._project_id).first()

        features_limits = json.loads(competition.params_surrogate.features_limits)
        sampling_features = filter(lambda k: features_limits[k]['sampling_input'], features_limits)
        var_names_str = ','.join(sampling_features)

        return {
            f'{self.model_name}.nt': self._params.get('count_gen_sampling_points', COUNT_GEN_SAMPLING_POINTS_DEFAULT),
            f'{self.model_name}.et': self._params.get('count_inc_sampling_points', COUNT_INC_SAMPLING_POINTS_DEFAULT),
            f'{self.model_name}.window_size': self._params.get('window_size', WINDOW_SIZE_DEFAULT),
            f'{self.model_name}.var_names_str': var_names_str,
        }

    def estimate_by_meta(self, features: dict[str, FeatureOption]):
        return True
