import json
import timeit
from typing import List, Optional

import pandas as pd
from data.modules.machine_learning.models._abstract_module import (
    AbstractModule, get_params, process_arguments)
from polygon.models import Competition
from polygon.surrogate_modeling_services import call_software
from project.models import Project


class RemoteSoftware(AbstractModule):
    """
    Send incoming dataset by URL to remote server with installed industrial software
    and return results in the form of dataset
    """

    to_use_input = False
    return_input = False

    def __init__(self, instance, plots=None):
        super().__init__(instance, plots)

        self.time: float = 0
        self.columns: Optional[List[str]] = None

    def get_extra_features(self, node, **kwargs):
        module = Project.objects.get(id=node.graph.project_id)
        group = module.group

        if competition := Competition.objects.filter(project_id=group.first().id).first():
            features_limits = json.loads(competition.params_surrogate.features_limits)
            self.columns = list(features_limits)

        if self.columns is None:
            return {}

        features = dict.fromkeys(self.columns, self.get_default_meta())

        return self.form_features_by_list(node.id, features)

    def get_extra_params(self, extra_params=None, **kwargs) -> list:
        name = self.get_name()

        extra_params = [
            {
                'model': name,
                'name': 'URL',
                'input_type': 'str',
                'default': 'http://127.0.0.1:5000/API/values',
                'description': 'Required. The url of the request',
            },
            {
                'model': name,
                'name': 'Proxy',
                'input_type': 'str',
                'default': None,
                'description': 'Optional. A dictionary of the protocol to the proxy url',
            },
            {
                'model': name,
                'name': 'User',
                'input_type': 'str',
                'default': None,
                'description': 'Username for HTTP authentication',
            },
            {
                'model': name,
                'name': 'Password',
                'input_type': 'str',
                'default': None,
                'description': 'Password for HTTP authentication',
            },
            {
                'model': name,
                'name': 'Timeout',
                'input_type': 'number',
                'default': 100,
                'description': 'A number indicating how many seconds to wait '
                'for the client to make a connection and/or '
                'send a response',
            },
        ]

        return super().get_extra_params(extra_params=extra_params, **kwargs)

    @process_arguments
    def call(self, data: list, model_parameters, **kwargs):
        if len(data) == 0:
            raise ValueError('No input data')

        name = self.get_name()
        params = get_params(name, model_parameters)
        url = params['URL']
        username = params['User']
        password = params['Password']
        timeout = params['Timeout']
        proxy = params['Proxy']

        start_time = timeit.default_timer()

        data_out = []
        for data_el in data:
            if 'df' not in data_el['data']:
                continue

            df = self.get_df(data_el['data'], clean_names=True)
            df_out = call_software(df, url, username, password, proxy, timeout)
            data_out.append({'name': name, 'data': {'df': df.join(df_out, how='inner')}})

        self.time = timeit.default_timer() - start_time
        self.columns = data_out[0]['data']['df'].columns.values.tolist()

        return data_out

    def get_json(self):
        label = 'Время работы узла запуска индустриального ПО, сек'
        df = pd.DataFrame([int(self.time)], columns=['Время работы узла, сек'])
        return self.build_table_by_df_json(df, label=label)

    def apply(self, data: list, calculation_id: str = None):
        pass
