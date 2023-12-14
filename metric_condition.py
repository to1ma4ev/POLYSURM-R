from data.modules.machine_learning.classes.scoring import MetricName
from data.modules.machine_learning.models._abstract_module import get_params


class MetricCondition:
    """
    Checking the quality of result metrics of models.
    The model returns True if metric condition satisfied, False otherwise.
    """

    to_minimize_metric = [
        MetricName.mean_squared_error.name,
        MetricName.mean_absolute_percentage_error.name,
        MetricName.mean_absolute_error.name,
        MetricName.max_error.name,
    ]

    @classmethod
    def get_params(cls, name: str) -> list:
        return [
            {
                'model': name,
                'name': 'Metric',
                'input_type': 'select',
                'default': MetricName.surrogate_modeling_names[0],
                'description': 'Metric which value to track during cyclic process',
                'types': {'select': MetricName.surrogate_modeling_names},
            },
            {
                'model': name,
                'name': 'Metric value',
                'input_type': 'number',
                'default': 0.001,
                'description': 'Desirable threshold value of selected metric',
            },
            {
                'model': name,
                'name': 'Iterations number',
                'input_type': 'number',
                'default': 100,
                'description': 'Maximum number of iterations during cyclic process',
            },
        ]

    @classmethod
    def fit(cls, scores_table, model_parameters, *args, **kwargs) -> bool:
        params = get_params(model_parameters['models_title'], model_parameters['properties'])
        metric_name = params['Metric']
        metric_threshold = float(params['Metric value'])

        metric_value = scores_table[metric_name].mean()

        if metric_name in cls.to_minimize_metric:
            return metric_value <= metric_threshold
        else:
            return metric_value >= metric_threshold
