import random

from lazy_imports import try_import

from smile_ml_core.models.sampling_methods import BaseSampling

with try_import() as smt_import:
    from smt.sampling_methods import LHS, FullFactorial, Random
    from smt.sampling_methods.sampling_method import ScaledSamplingMethod


class Mixed(ScaledSamplingMethod):
    def __init__(self, **kwargs):
        self.sampling_methods: list[type[ScaledSamplingMethod]] = [LHS, FullFactorial, Random]
        super().__init__(**kwargs)

    def _compute(self, nt):
        sampling_method = random.choice(self.sampling_methods)

        xlimits = self.options['xlimits']
        instance = sampling_method(xlimits=xlimits)

        return instance._compute(nt)


class MixedSampling(BaseSampling):
    @property
    def model_cls(self) -> type[Mixed]:
        smt_import.check()

        return Mixed
