import dataclasses
from abc import ABC
from typing import Type, Optional, Dict

import numpy as np

@dataclasses.dataclass
class EffectSettings:
    setting_name: str
    data_type: Type
    range_min: Optional[float] = None
    range_max: Optional[float] = None

class SampleEffect(ABC):
    def __init__(self):
        self.sensitivity = 1.0

    def apply_step(self, sample_slice: np.ndarray, slice_offset: int, resolution: int, magnitude: float):
        raise NotImplementedError()

    def get_settings(self) -> Dict[str, EffectSettings]:
        return {'sensitivity': EffectSettings('Sensitivity', float, 0.0, None)}