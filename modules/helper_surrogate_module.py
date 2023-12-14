from helper.servives.helpers.models import (
    HelperDataNode,
    HelperMetricCondition,
    HelperRemoteSoftware,
    HelperSampling,
    HelperStreamRecalling,
)
from helper.servives.helpers.modules._base_module_helper import BaseModuleHelper
from helper.servives.helpers.modules.helper_automl_module import AutoMLRegressionHelper
from helper.servives.helpers.modules.helper_plain_surrogate_model import (
    PlainSurrogateHelper,
)


class SurrogateBaselineHelper(BaseModuleHelper):
    name = 'Построение baseline для задачи суррогатного моделирования'
    group_name = 'baseline_surrogate'
    description = 'Построение baseline для задачи суррогатного моделирования'
    hint_message = 'Surrogate baseline'

    helpers_list = [
        HelperDataNode,
        HelperSampling,
        HelperStreamRecalling,
        [HelperMetricCondition],
        HelperRemoteSoftware,
        [AutoMLRegressionHelper, PlainSurrogateHelper],
    ]
