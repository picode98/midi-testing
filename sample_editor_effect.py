import dataclasses
from abc import ABC
from typing import Type, Optional, Dict, List

import numpy as np
import dataclasses_json

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
    
@dataclasses.dataclass
class LoopedRegion(dataclasses_json.DataClassJsonMixin):
    start: Optional[float]
    end: Optional[float]
    loop_duration: Optional[float]
    sub_loops: List['LoopedRegion']

    def deep_copy(self):
        return dataclasses.replace(self, sub_loops=[loop.deep_copy() for loop in self.sub_loops])