from typing import Optional

from helper.servives.helpers.models import HelperScorer, HelperTrainTestSplit
from helper.servives.helpers.models.helper_surrogate_regressor import HelperRegressor
from helper.servives.helpers.modules._base_module_helper import BaseModuleHelper


class PlainSurrogateHelper(BaseModuleHelper):
    name = 'Построение baseline для задачи суррогатного моделирования'
    group_name = 'plain'
    description = 'Построение baseline для задачи суррогатного моделирования'
    hint_message = 'Surrogate baseline'

    helpers_list = [
        HelperTrainTestSplit,
        HelperRegressor,
        HelperScorer,
    ]

    def refactor(self, groups: Optional[list] = None, helpers_list: Optional[list] = None, root_helper=None) -> bool:
        if self.parameters.get('include_auto_ml', False):
            return True

        return super().refactor(groups=groups, helpers_list=helpers_list, root_helper=root_helper)
