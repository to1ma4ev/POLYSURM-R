import json
from typing import List, Optional

import pandas as pd
from django.utils.translation import gettext as _
from smt.sampling_methods.sampling_method import SamplingMethod

from data.modules.machine_learning.models._abstract_module import (
    AbstractModule,
    get_params,
    process_arguments,
)
from data.modules.machine_learning.models._class_state import ClassState
from polygon.models import Competition
from project.models import Project


class SamplingModel(AbstractModule):
    """
    Perform sampling of input data according to selected method
    """

    to_use_input = False
    return_input = False
    bounds: pd.DataFrame
    window_sizes: pd.Series

    var_names_column: str = 'variable_name'

    def __init__(self, instance, plots=None):
        super().__init__(instance, plots)

        self.first_iteration: bool = True

        self.dfs_out: list = []
        self.columns: Optional[List[str]] = None
        self.df_before_scorer = None

        self.target_column = 'target'
        self.predict_column = 'predict'

    def get_extra_features(self, node, **kwargs):
        module = Project.objects.get(id=node.graph.project_id)
        group = module.group

        if competition := Competition.objects.filter(project_id=group.first().id).first():
            features_limits = json.loads(competition.params_surrogate.features_limits)
            self.columns = sorted(filter(lambda k: features_limits[k]['sampling_input'], features_limits))

        if self.columns is None:
            return {}

        child_models = node.get_siblings(kind='models')
        scorers = [n for n in child_models if n.is_model({ClassState.Scorer, ClassState.IterationsTool})]
        self.df_before_scorer = scorers[0].get_parent() if scorers else None

        features = dict.fromkeys(self.columns, self.get_default_meta())

        return self.form_features_by_list(node.id, features)

    def get_extra_params(self, extra_params=None, **kwargs) -> list:
        extra_params = [
            {
                'model': self.get_name(),
                'name': 'nt',
                'input_type': 'number',
                'default': 15,
                'description': 'Number of sampling points',
            },
            {
                'model': self.get_name(),
                'name': 'et',
                'input_type': 'number',
                'default': 10,
                'description': 'Number of extension points',
            },
        ]

        extra_params.extend(self.instance.get_params(self.get_name()))

        return super().get_extra_params(extra_params=extra_params, **kwargs)

    @process_arguments
    def call(self, data: list, model_parameters, **kwargs):
        if len(data) == 0:
            raise ValueError('No input parameters data')

        name = self.get_name()
        assert len(data) == 1, _(f'{name} works only with single DataFrame')

        params = get_params(name, model_parameters)
        nt = params.pop('nt')
        et = params.pop('et')

        data_out = []

        if self.first_iteration:
            self.first_iteration = False
            et = nt

            df = self.get_df(data[0]['data'], clean_names=True)
            bounds = df.set_index(self.var_names_column).sort_index()

            self.columns = bounds.index.to_list()
            self.window_sizes = (bounds['upper'] - bounds['lower']) / 10
            self.bounds = bounds
        else:
            data_el = self.df_before_scorer.data.get('output', return_one=True)
            df = self.get_df(data_el, clean_names=True)

            point_index = ((df[self.predict_column] - df[self.target_column]) ** 2).argmax()
            point = df[self.columns].iloc[point_index].sort_index()

            bounds = pd.concat([point - self.window_sizes, point + self.window_sizes], axis=1)
            bounds.clip(self.bounds['lower'], self.bounds['upper'], axis=0, inplace=True)

        limits = bounds.to_numpy()
        self.logger.info(f'New bounds of variables: {limits.round(3)}')

        sampling: SamplingMethod = self.instance(xlimits=limits, **params)
        df_out = pd.DataFrame(sampling(et), columns=self.columns)
        self.dfs_out.append(df_out)

        df_out = pd.concat(self.dfs_out, ignore_index=True)
        data_out = [{'name': name, 'data': {'df': df_out}}]

        return data_out

    def apply(self, data: list):
        pass
