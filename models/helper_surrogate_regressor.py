from typing import Set

from data.modules.machine_learning.models._class_state import ClassState
from helper.servives.helpers.models import HelperFedot


class HelperRegressor(HelperFedot):
    name = 'Surrogate Regressor'
    group_name = 'baseline_surrogate'
    description = 'сэмплирование'

    def __init__(self, *args, **kwargs):
        super(HelperRegressor, self).__init__(*args, **kwargs)

    @property
    def model_name(self) -> str:
        return self.parameters.get('cv_model', 'XGBRegressor')

    @property
    def searching_modules(self) -> Set[ClassState]:
        return {
            ClassState.RegressionModel,
        }
