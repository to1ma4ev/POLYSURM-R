from typing import Any

from lazy_imports import try_import

from smile_ml_core.models.sampling_methods import BaseSampling

with try_import() as smt_import:
    from smt.sampling_methods import FullFactorial


class FullFactorialSampling(BaseSampling):
    @property
    def model_cls(self) -> type[FullFactorial]:
        smt_import.check()

        return FullFactorial

    def get_extra_parameters(self, name: str, input_features: Any, **kwargs: Any) -> list[dict[str, Any]]:
        params: list = super().get_extra_parameters(name, input_features)

        return params + [
            {
                'model': name,
                'name': 'clip',
                'input_type': 'bool',
                'default': False,
                'description': 'Round number of samples to the sampling number product of each nx dimensions (> asked nt)',
            },
        ]
