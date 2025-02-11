from typing import Any

from lazy_imports import try_import

from smile_ml_core.models.sampling_methods import BaseSampling

with try_import() as smt_import:
    from smt.sampling_methods import LHS


class LatinHypercubeSampling(BaseSampling):
    @property
    def model_cls(self) -> type[LHS]:
        smt_import.check()

        return LHS

    def get_extra_parameters(self, name: str, input_features: Any, **kwargs: Any) -> list[dict[str, Any]]:
        params: list = super().get_extra_parameters(name, input_features)

        criterion = [
            'center',
            'maximin',
            'centermaximin',
            'correlation',
            'ese',
        ]

        return params + [
            {
                'model': name,
                'name': 'criterion',
                'input_type': 'select',
                'types': {'select': criterion},
                'default': criterion[0],
                'description': 'Criterion used to construct the LHS design',
            },
            {
                'model': name,
                'name': 'random_state',
                'input_type': 'number',
                'default': None,
                'description': 'Seed number which controls random draws',
            },
        ]
