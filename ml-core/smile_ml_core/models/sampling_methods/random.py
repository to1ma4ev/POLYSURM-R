from lazy_imports import try_import

from smile_ml_core.models.sampling_methods import BaseSampling

with try_import() as smt_import:
    from smt.sampling_methods import Random


class RandomSampling(BaseSampling):
    @property
    def model_cls(self) -> type[Random]:
        smt_import.check()

        return Random
