from typing import Dict, Set

from data.modules.machine_learning.data import DictDataFrame
from data.modules.machine_learning.methods.sampling.sampling_methods import LHS
from data.modules.machine_learning.models._class_state import ClassState
from data.modules.machine_learning.node.features import FeatureMeta
from helper.servives.helpers.models import BaseModelHelper
from polygon.constants import (
    COUNT_GEN_SAMPLING_POINTS_DEFAULT,
    COUNT_INC_SAMPLING_POINTS_DEFAULT,
)


class HelperSampling(BaseModelHelper):
    name = 'Сэмплирование'
    group_name = 'sampling'
    description = 'Сэмплирование'

    hint_message = 'To sampling ...'

    def __init__(self, *args, **kwargs):
        super(HelperSampling, self).__init__(*args, **kwargs)

        nt = self._params.pop('count_gen_sampling_points', COUNT_GEN_SAMPLING_POINTS_DEFAULT)
        et = self._params.pop('count_inc_sampling_points', COUNT_INC_SAMPLING_POINTS_DEFAULT)
        self._params.update(nt=nt, et=et)

    @property
    def model_name(self) -> str:
        return self.parameters.get('sampling_method', LHS.__name__)

    @property
    def searching_modules(self) -> Set[ClassState]:
        return {
            ClassState.SamplingModel,
        }

    def estimate_by_meta(self, features: Dict[str, FeatureMeta]):
        return True

    def estimate_by_data(self, ddf: DictDataFrame) -> bool:
        return True

    def check_possibility_to_fix_existed_models(self, node_modules) -> bool:
        return False
