from helper.servives.helpers.models import HelperFedot


class HelperRegressor(HelperFedot):
    """ """

    name = 'Модель регрессии'
    group_name = 'regressor'
    description = 'Helper модели регрессии'

    hint_message = 'To create regression model node'

    @property
    def model_name(self) -> str:
        return self.parameters.get('cv_model', 'XGBRegressorModel')

    @property
    def default_properties(self) -> dict:
        return {}
